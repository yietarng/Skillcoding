"""The skill bank: storage, retrieval, versioning, and compaction.

Mirrors the paper's description of a bank that (a) stays auditable --
every mutation is versioned and every exclusion has a logged reason -- and
(b) stays compact -- redundant or unhelpful skills get merged/dropped so the
bank doesn't grow without bound across rounds of iterative construction.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from codeskill.schema import Granularity, Provenance, Skill


@dataclass
class RetrievalResult:
    skill: Skill
    score: float


@dataclass
class ExclusionRecord:
    """Why a candidate was *not* added, or why a skill was dropped."""

    skill_name: str
    reason: str
    round_num: int


@dataclass
class BankSnapshot:
    """Frozen, immutable view of the bank at the end of a trial round."""

    round_num: int
    skills: dict[str, dict]


class SkillBank:
    """Mutable store of `Skill` objects with add/update/merge/drop operations."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._snapshots: list[BankSnapshot] = []
        self._exclusions: list[ExclusionRecord] = []
        self.current_round: int = 0

    # -- basic access -----------------------------------------------------

    def get(self, skill_id: str) -> Optional[Skill]:
        return self._skills.get(skill_id)

    def active_skills(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.active]

    def all_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self.active_skills())

    # -- mutations ----------------------------------------------------------

    def _duplicate_of(self, skill: Skill) -> Optional[Skill]:
        key = skill.content_key()
        for existing in self.active_skills():
            if existing.content_key() == key:
                return existing
        return None

    def add(self, skill: Skill) -> tuple[bool, Optional[str]]:
        """Add a new skill candidate.

        Returns `(accepted, exclusion_reason)`. A candidate is rejected (not
        added) if it duplicates an already-active skill's content -- the
        caller should route such cases to `update`/`merge` instead if new
        provenance should still be recorded.
        """
        dup = self._duplicate_of(skill)
        if dup is not None:
            reason = f"duplicate_of:{dup.id}"
            self._exclusions.append(ExclusionRecord(skill.name, reason, self.current_round))
            return False, reason
        self._skills[skill.id] = skill
        return True, None

    def update(self, skill_id: str, *, new_provenance: Optional[Provenance] = None, **field_updates) -> Skill:
        """Apply an evolution step to an existing skill: revise its content
        and/or attach new evidence, bumping its version."""
        skill = self._skills[skill_id]
        for field_name, value in field_updates.items():
            if not hasattr(skill, field_name):
                raise AttributeError(f"Skill has no field {field_name!r}")
            setattr(skill, field_name, value)
        if new_provenance is not None:
            skill.provenance.append(new_provenance)
        skill.version += 1
        import time

        skill.updated_at = time.time()
        return skill

    def merge(self, primary_id: str, absorbed_ids: list[str], rationale: str = "") -> Skill:
        """Fold `absorbed_ids` into `primary_id`: union their steps/tags and
        provenance, then deactivate the absorbed skills (kept for audit, not
        deleted)."""
        primary = self._skills[primary_id]
        for other_id in absorbed_ids:
            if other_id == primary_id:
                continue
            other = self._skills.get(other_id)
            if other is None or not other.active:
                continue
            for step in other.steps:
                if step not in primary.steps:
                    primary.steps.append(step)
            for tag in other.tags:
                if tag not in primary.tags:
                    primary.tags.append(tag)
            primary.provenance.extend(other.provenance)
            other.active = False
            self._exclusions.append(
                ExclusionRecord(other.name, f"merged_into:{primary_id} ({rationale})", self.current_round)
            )
        primary.version += 1
        import time

        primary.updated_at = time.time()
        return primary

    def drop(self, skill_id: str, reason: str) -> None:
        skill = self._skills[skill_id]
        skill.active = False
        self._exclusions.append(ExclusionRecord(skill.name, f"dropped:{reason}", self.current_round))

    # -- retrieval ------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.strip(".,:;()[]{}").lower() for t in text.split() if len(t) > 2}

    def _skill_tokens(self, skill: Skill) -> set[str]:
        body = " ".join([skill.name, skill.description, *skill.steps, *skill.tags])
        return self._tokenize(body)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        granularity: Optional[Granularity] = None,
    ) -> list[RetrievalResult]:
        """Rank active skills by lexical overlap with `query`.

        Deliberately dependency-free (no embedding model required) so the
        bank is usable offline; swap in a real embedding-similarity scorer
        by subclassing and overriding this method for production use.
        """
        query_tokens = self._tokenize(query)
        candidates = self.active_skills()
        if granularity is not None:
            candidates = [s for s in candidates if s.granularity == granularity]

        scored: list[RetrievalResult] = []
        for skill in candidates:
            skill_tokens = self._skill_tokens(skill)
            if not query_tokens or not skill_tokens:
                overlap = 0.0
            else:
                intersection = query_tokens & skill_tokens
                union = query_tokens | skill_tokens
                overlap = len(intersection) / len(union)
            # Blend in a small prior for skills with a proven track record so
            # that, all else equal, reliable skills surface first.
            score = overlap + 0.05 * skill.success_rate()
            scored.append(RetrievalResult(skill, score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # -- compaction -------------------------------------------------------

    def compact(self, min_uses: int = 3, min_success_rate: float = 0.2) -> list[str]:
        """Drop skills that have been tried enough times to trust the signal
        but consistently fail to help the downstream agent.

        This is what keeps the bank at a "stable size during iterative
        construction" instead of growing unboundedly: it's applied after
        each round on skills old enough to have a meaningful usage count.
        """
        dropped: list[str] = []
        for skill in self.active_skills():
            if skill.usage_count >= min_uses and skill.success_rate() < min_success_rate:
                self.drop(skill.id, f"low_success_rate={skill.success_rate():.2f}")
                dropped.append(skill.id)
        return dropped

    # -- rounds / snapshots -------------------------------------------------

    def advance_round(self) -> BankSnapshot:
        """Freeze the current bank state and start a new trial round.

        Snapshots give the "explicit version control of bank state across
        rounds" the reconstruction notes describe: you can always look back
        at what the bank looked like before a given round's mutations.
        """
        snapshot = BankSnapshot(
            round_num=self.current_round,
            skills={sid: copy.deepcopy(s.to_dict()) for sid, s in self._skills.items()},
        )
        self._snapshots.append(snapshot)
        self.current_round += 1
        return snapshot

    def snapshot_at(self, round_num: int) -> Optional[BankSnapshot]:
        for snap in self._snapshots:
            if snap.round_num == round_num:
                return snap
        return None

    def exclusions(self) -> list[ExclusionRecord]:
        return list(self._exclusions)

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        return json.dumps(
            {
                "current_round": self.current_round,
                "skills": [s.to_dict() for s in self._skills.values()],
            },
            indent=2,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def from_json(cls, text: str) -> "SkillBank":
        data = json.loads(text)
        bank = cls()
        bank.current_round = data.get("current_round", 0)
        for skill_dict in data.get("skills", []):
            skill = Skill.from_dict(skill_dict)
            bank._skills[skill.id] = skill
        return bank

    @classmethod
    def load(cls, path: str | Path) -> "SkillBank":
        return cls.from_json(Path(path).read_text())
