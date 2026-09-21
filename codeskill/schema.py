"""Core data structures: trajectories, skills, provenance, and manager operations.

Design notes (mapped from the CODESKILL paper description):
  - Skills are extracted at two granularities: TASK (a whole successful/failed
    episode, e.g. "how to fix a failing pytest import error") and EVENT (a
    single reusable action pattern inside an episode, e.g. "run `pip show`
    before editing setup.py to confirm the installed version").
  - Every skill carries provenance: which trajectory and which steps within
    it justify the skill's existence, so the bank stays auditable.
  - Grounding rule ("C-only" task extraction, from the reconstruction notes):
    a task-level skill is only accepted if each of its rule/steps cites an
    assistant action that was followed by an *observed* tool result -- not a
    hallucinated or predicted outcome. `Trajectory.is_grounded_span` enforces
    this at the data-structure level so extractors can't silently skip it.
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
    """Level of abstraction a skill was extracted at."""

    TASK = "task"
    EVENT = "event"


class OperationType(str, Enum):
    """The management actions the learnable skill-manager policy can take."""

    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    DROP = "drop"
    NOOP = "noop"


@dataclass
class TrajectoryStep:
    """One turn of an agent trajectory.

    `tool_result` is `None` until the tool has actually executed -- a step
    with `assistant_action` set but `tool_result` still `None` is *not*
    grounded and must not back a task-level skill's citation.
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
        """True iff every cited step index is a grounded (observed) step.

        This is the data-level enforcement of the paper's "C-only" rule:
        a skill may only cite assistant actions that were actually followed
        by an observed tool result, never a predicted/hallucinated one.
        """
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
    """Links a skill back to the exact evidence that justified it."""

    trajectory_id: str
    step_indices: list[int]
    trial_round: int = 0
    extracted_at: float = field(default_factory=time.time)
    note: str = ""


@dataclass
class Skill:
    """A reusable procedural skill stored in the skill bank."""

    name: str
    description: str
    steps: list[str]
    granularity: Granularity
    preconditions: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    id: str = field(default_factory=lambda: _new_id("skill"))
    version: int = 1
    usage_count: int = 0
    success_count: int = 0
    quality_score: float = 0.0
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
        """Cheap lexical fingerprint used for duplicate/merge detection."""
        body = " ".join([self.name.lower(), self.description.lower(), *[s.lower() for s in self.steps]])
        tokens = sorted(set(t.strip(".,:;()") for t in body.split() if len(t) > 2))
        return "|".join(tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": self.steps,
            "granularity": self.granularity.value,
            "preconditions": self.preconditions,
            "tags": self.tags,
            "provenance": [vars(p) for p in self.provenance],
            "version": self.version,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "quality_score": self.quality_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active": self.active,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        prov = [Provenance(**p) for p in d.get("provenance", [])]
        return cls(
            id=d["id"],
            name=d["name"],
            description=d["description"],
            steps=d["steps"],
            granularity=Granularity(d["granularity"]),
            preconditions=d.get("preconditions", []),
            tags=d.get("tags", []),
            provenance=prov,
            version=d.get("version", 1),
            usage_count=d.get("usage_count", 0),
            success_count=d.get("success_count", 0),
            quality_score=d.get("quality_score", 0.0),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            active=d.get("active", True),
        )


@dataclass
class SkillOperation:
    """One decision emitted by the skill-manager policy for a candidate skill.

    `target_skill_id` is set for UPDATE/DROP; `merge_with` lists the other
    skill ids folded into this one for MERGE.
    """

    op_type: OperationType
    skill: Optional[Skill] = None
    target_skill_id: Optional[str] = None
    merge_with: list[str] = field(default_factory=list)
    rationale: str = ""
    exclusion_reason: Optional[str] = None


def load_trajectories(path: str | Path) -> list[Trajectory]:
    data = json.loads(Path(path).read_text())
    return [Trajectory.from_dict(d) for d in data]


def save_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    Path(path).write_text(json.dumps([t.to_dict() for t in trajectories], indent=2))
