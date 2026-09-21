"""The skill bank: storage, retrieval, versioning, and compaction.

Mirrors the paper's description of a bank that (a) stays auditable --
every mutation is versioned and every exclusion has a logged reason -- and
(b) stays compact -- redundant or unhelpful skills get merged/dropped so the
bank doesn't grow without bound across rounds of iterative construction.

Per Figure 9/13 of the appendix, a "merge" decision doesn't mechanically
union two skills' rule lists: the maintenance policy itself produces the
full merged `{title, when_to_apply, rules}` object, which *replaces* the
target skill in place. `SkillBank.replace` implements exactly that.
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

    skill_title: str
    reason: str
    round_num: int


@dataclass
class BankSnapshot:
    """Frozen, immutable view of the bank at the end of a trial round."""

    round_num: int
    skills: dict[str, dict]


class SkillBank:
    """Mutable store of `Skill` objects with add/replace/drop operations."""

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
        """Add a new skill candidate (the maintenance policy's `add` decision).

        Returns `(accepted, exclusion_reason)`. A candidate is rejected (not
        added) if it duplicates an already-active skill's content.
        """
        dup = self._duplicate_of(skill)
        if dup is not None:
            reason = f"duplicate_of:{dup.id}"
            self._exclusions.append(ExclusionRecord(skill.title, reason, self.current_round))
            return False, reason
        self._skills[skill.id] = skill
        return True, None

    def replace(self, target_id: str, replacement: Skill, *, rationale: str = "") -> Skill:
        """Replace `target_id`'s content in place with `replacement` (used
        for both the maintenance policy's `merge` decision and the
        evolution policy's `evolve` decision), bumping its version and
        keeping its id/usage history/provenance."""
        target = self._skills[target_id]
        target.title = replacement.title
        target.granularity = replacement.granularity
        target.when_to_apply = replacement.when_to_apply
        target.rules = replacement.rules
        target.provenance.extend(replacement.provenance)
        target.version += 1
        import time

        target.updated_at = time.time()
        if rationale:
            target.provenance.append(Provenance(trajectory_id="", step_indices=[], note=rationale))
        return target

    def drop(self, skill_id: str, reason: str) -> None:
        skill = self._skills[skill_id]
        skill.active = False
        self._exclusions.append(ExclusionRecord(skill.title, f"dropped:{reason}", self.current_round))

    # -- retrieval ------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {t.strip(".,:;()[]{}").lower() for t in text.split() if len(t) > 2}

    def _skill_tokens(self, skill: Skill) -> set[str]:
        body = " ".join([skill.title, skill.when_to_apply, *skill.rules])
        return self._tokenize(body)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        granularity: Optional[Granularity] = None,
        exclude_trajectory_ids: Optional[set[str]] = None,
    ) -> list[RetrievalResult]:
        """Rank active skills by lexical overlap with `query`.

        Deliberately dependency-free (no embedding model required) so the
        bank is usable offline; the paper (Appendix C) encodes retrieval
        documents built from a skill's title/when_to_apply/rules with
        `sentence-transformers/all-MiniLM-L6-v2` and builds separate dense
        indexes per benchmark and skill granularity -- swap in a real
        embedding-similarity scorer by subclassing and overriding this
        method for production use.

        `exclude_trajectory_ids`, when given, drops any skill whose
        provenance traces back to one of those trajectories -- the same
        "skills generated from the same evaluation instance are filtered
        out to avoid same-instance leakage" rule Appendix C describes for
        retrieval at task-solving time.
        """
        query_tokens = self._tokenize(query)
        candidates = self.active_skills()
        if granularity is not None:
            candidates = [s for s in candidates if s.granularity == granularity]
        if exclude_trajectory_ids:
            candidates = [
                s for s in candidates if not any(p.trajectory_id in exclude_trajectory_ids for p in s.provenance)
            ]

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

    def retrieve_similar(
        self,
        skill: Skill,
        top_k: int = 5,
        exclude_trajectory_ids: Optional[set[str]] = None,
    ) -> list[RetrievalResult]:
        """Like `retrieve`, but queries with a *skill's* own full content
        (title + when_to_apply + rules) rather than a bare query string --
        the same document construction Appendix C describes for indexed
        skills, applied symmetrically to the query side. This is what
        maintenance (Fig 9) should use to find "retrieved similar skills"
        for a candidate: matching on the candidate's full content, not just
        its `when_to_apply`.
        """
        query = " ".join([skill.title, skill.when_to_apply, *skill.rules])
        return self.retrieve(query, top_k=top_k, granularity=skill.granularity, exclude_trajectory_ids=exclude_trajectory_ids)

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
        """Freeze the current bank state and start a new trial round."""
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
