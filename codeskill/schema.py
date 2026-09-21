"""Core data structures: trajectories, skills, provenance, and stage outcomes.

The `Skill` schema and the three `*Outcome` dataclasses deliberately mirror
the JSON schemas given in the CODESKILL paper's appendix (Figures 6-9, see
`codeskill/prompts.py` for the verbatim prompts and source citation):

  - generate/skip (task- and event-level extraction, Fig 6-7)
  - evolve/skip (skill evolution, Fig 8)
  - add/merge/drop (skill-bank maintenance, Fig 9)

A skill is always `{title, granularity, when_to_apply, rules}` -- there is
no separate "description" or "steps" field beyond what the paper's own
schema defines. `Provenance.step_indices` is this codebase's own addition
(not in the paper): a machine-checkable pointer to which trajectory steps a
skill's evidence should trace back to, used by `codeskill/grounding.py` as a
cheap automated safeguard alongside the paper's LLM-judged "groundedness"
rubric dimension.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Granularity(str, Enum):
    """Matches the paper's `granularity` field values exactly."""

    GENERAL = "general"
    EVENT_DRIVEN = "event-driven"


class Decision(str, Enum):
    """The action vocabulary across all three manager-policy stages."""

    GENERATE = "generate"  # extraction (Fig 6/7)
    EVOLVE = "evolve"  # evolution (Fig 8)
    ADD = "add"  # maintenance (Fig 9)
    MERGE = "merge"  # maintenance (Fig 9)
    DROP = "drop"  # maintenance (Fig 9)
    SKIP = "skip"  # extraction or evolution


@dataclass
class TrajectoryStep:
    """One turn of an agent trajectory.

    `tool_result` is `None` until the tool has actually executed -- a step
    with `assistant_action` set but `tool_result` still `None` is *not*
    grounded, and `codeskill.grounding` never treats it as evidence.
    """

    index: int
    assistant_action: str
    tool_name: Optional[str] = None
    tool_input: Optional[dict[str, Any]] = None
    tool_result: Optional[str] = None
    observed: bool = False

    def is_grounded(self) -> bool:
        return bool(self.assistant_action) and self.observed and self.tool_result is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "assistant_action": self.assistant_action,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_result": self.tool_result,
            "observed": self.observed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrajectoryStep":
        return cls(
            index=d["index"],
            assistant_action=d["assistant_action"],
            tool_name=d.get("tool_name"),
            tool_input=d.get("tool_input"),
            tool_result=d.get("tool_result"),
            observed=d.get("observed", False),
        )


@dataclass
class Trajectory:
    """A full agent episode, successful or not."""

    task_id: str
    task_description: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    success: Optional[bool] = None
    id: str = field(default_factory=lambda: _new_id("traj"))
    metadata: dict[str, Any] = field(default_factory=dict)

    def grounded_step_indices(self) -> list[int]:
        return [s.index for s in self.steps if s.is_grounded()]

    def is_grounded_span(self, indices: list[int]) -> bool:
        """True iff every given step index is a grounded (observed) step."""
        if not indices:
            return False
        grounded = set(self.grounded_step_indices())
        return all(i in grounded for i in indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        return cls(
            id=d.get("id", _new_id("traj")),
            task_id=d["task_id"],
            task_description=d["task_description"],
            steps=[TrajectoryStep.from_dict(s) for s in d.get("steps", [])],
            success=d.get("success"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Provenance:
    """Links a skill back to the trajectory (and, as a runtime addition
    beyond the paper, the specific steps) that justified it."""

    trajectory_id: str
    step_indices: list[int] = field(default_factory=list)
    trial_round: int = 0
    extracted_at: float = field(default_factory=time.time)
    note: str = ""


@dataclass
class Skill:
    """A reusable procedural skill, in the paper's own schema:
    `{title, granularity, when_to_apply, rules}`."""

    title: str
    granularity: Granularity
    when_to_apply: str
    rules: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    id: str = field(default_factory=lambda: _new_id("skill"))
    version: int = 1
    usage_count: int = 0
    success_count: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active: bool = True

    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count

    def record_usage(self, succeeded: bool) -> None:
        self.usage_count += 1
        if succeeded:
            self.success_count += 1
        self.updated_at = time.time()

    def content_key(self) -> str:
        """Cheap lexical fingerprint used for duplicate/merge-candidate detection."""
        body = " ".join([self.title.lower(), self.when_to_apply.lower(), *[r.lower() for r in self.rules]])
        tokens = sorted(set(t.strip(".,:;()") for t in body.split() if len(t) > 2))
        return "|".join(tokens)

    def to_paper_dict(self) -> dict[str, Any]:
        """The exact `skill` object shape the appendix prompts specify --
        used when serializing this skill back into a prompt (e.g. as a
        `PROPOSED_OUTPUT` or `EXISTING_PRIOR_KNOWLEDGE` for a judge)."""
        return {
            "title": self.title,
            "granularity": self.granularity.value,
            "when_to_apply": self.when_to_apply,
            "rules": self.rules,
        }

    def to_dict(self) -> dict[str, Any]:
        d = self.to_paper_dict()
        d.update(
            {
                "id": self.id,
                "provenance": [vars(p) for p in self.provenance],
                "version": self.version,
                "usage_count": self.usage_count,
                "success_count": self.success_count,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "active": self.active,
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        prov = [Provenance(**p) for p in d.get("provenance", [])]
        return cls(
            id=d["id"],
            title=d["title"],
            granularity=Granularity(d["granularity"]),
            when_to_apply=d["when_to_apply"],
            rules=d.get("rules", []),
            provenance=prov,
            version=d.get("version", 1),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            active=d.get("active", True),
        )


@dataclass
class ExtractionOutcome:
    """Result of a task- or event-level extraction call (Fig 6/7): `generate`|`skip`."""

    decision: Decision
    skill: Optional[Skill] = None
    reason: str = ""


@dataclass
class EvolutionOutcome:
    """Result of an evolution call (Fig 8): `evolve`|`skip`."""

    decision: Decision
    target_skill_id: Optional[str] = None
    skill: Optional[Skill] = None
    reason: str = ""


@dataclass
class MaintenanceOutcome:
    """Result of a skill-bank maintenance call (Fig 9): `add`|`merge`|`drop`."""

    decision: Decision
    merge_target_skill_id: Optional[str] = None
    skill: Optional[Skill] = None
    reason: str = ""


def load_trajectories(path: str | Path) -> list[Trajectory]:
    data = json.loads(Path(path).read_text())
    return [Trajectory.from_dict(d) for d in data]


def save_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    Path(path).write_text(json.dumps([t.to_dict() for t in trajectories], indent=2))
