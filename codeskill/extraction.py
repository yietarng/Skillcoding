"""The four manager-policy stages, each backed by its verbatim appendix
prompt (see `codeskill/prompts.py`): task-level extraction (Fig 6),
event-driven extraction (Fig 7), evolution (Fig 8), and skill-bank
maintenance (Fig 9).

Note what each stage does and doesn't see, per the paper's own description:
extraction (Fig 6/7) sees only trajectories, never the existing bank;
evolution (Fig 8) sees a set of already-relevant existing skills plus one
new trajectory; maintenance (Fig 9) sees a candidate skill and retrieved
similar skills but *no* trajectory at all ("maintain decisions are made
without a trajectory" -- Fig 13's judge description). That separation is
preserved here rather than collapsed into a single combined prompt.

The user-message formatting below (how a trajectory or a skill gets
serialized into text) is this codebase's own design -- the appendix
specifies the system prompt and what the user message must contain, but not
its exact layout.
"""
from __future__ import annotations

import json

from codeskill.llm import LLMClient, classify_output, extract_json
from codeskill.prompts import (
    EVENT_EXTRACTION_SYSTEM_PROMPT,
    SKILL_EVOLUTION_SYSTEM_PROMPT,
    SKILL_MAINTENANCE_SYSTEM_PROMPT,
    TASK_EXTRACTION_SYSTEM_PROMPT,
)
from codeskill.schema import (
    Decision,
    EvolutionOutcome,
    ExtractionOutcome,
    Granularity,
    MaintenanceOutcome,
    Provenance,
    Skill,
    Trajectory,
)

# -- user-message formatting (this codebase's own design) -------------------


def format_trajectory(trajectory: Trajectory) -> str:
    lines = [
        f"Task context: {trajectory.task_description}",
        f"Outcome: {'success' if trajectory.success else 'failure'}",
        "Trajectory:",
    ]
    for step in trajectory.steps:
        marker = "observed" if step.is_grounded() else "unobserved"
        lines.append(f"  [{step.index}] ({marker}) action: {step.assistant_action}")
        if step.tool_name:
            lines.append(f"        tool: {step.tool_name}({step.tool_input or {}})")
        if step.tool_result is not None:
            lines.append(f"        result: {step.tool_result}")
    return "\n".join(lines)


def format_skill_ref(skill: Skill) -> str:
    return json.dumps({"id": skill.id, **skill.to_paper_dict()})


def build_task_extraction_prompt(trajectories: list[Trajectory]) -> str:
    header = trajectories[0].task_description if trajectories else ""
    parts = [f"Task context: {header}", ""]
    for i, trajectory in enumerate(trajectories, start=1):
        parts.append(f"--- Trajectory {i} ---")
        parts.append(format_trajectory(trajectory))
        parts.append("")
    return "\n".join(parts)


def build_event_extraction_prompt(trajectory: Trajectory) -> str:
    return format_trajectory(trajectory)


def build_evolution_prompt(existing_skills: list[Skill], trajectory: Trajectory) -> str:
    parts = ["Relevant existing skills:"]
    for skill in existing_skills:
        parts.append(f"  {format_skill_ref(skill)}")
    parts.append("")
    parts.append(format_trajectory(trajectory))
    return "\n".join(parts)


def build_maintenance_prompt(candidate: Skill, retrieved: list[Skill]) -> str:
    parts = [f"Candidate skill:\n  {json.dumps(candidate.to_paper_dict())}", "", "Retrieved similar skills:"]
    for skill in retrieved:
        parts.append(f"  {format_skill_ref(skill)}")
    return "\n".join(parts)


# -- response parsing (shared by live LLM calls and codeskill.grpo rollouts) -


def _parse_skill_payload(skill_data: dict) -> tuple[Skill | None, str]:
    try:
        granularity = Granularity(skill_data.get("granularity"))
    except ValueError:
        return None, f"invalid_granularity:{skill_data.get('granularity')!r}"
    title = skill_data.get("title")
    when_to_apply = skill_data.get("when_to_apply")
    if not title or not when_to_apply:
        return None, "missing_required_field:title_or_when_to_apply"
    skill = Skill(title=title, granularity=granularity, when_to_apply=when_to_apply, rules=list(skill_data.get("rules", [])))
    return skill, ""


def parse_extraction_response(raw: str) -> ExtractionOutcome:
    """Shared parser for Fig 6 (task) and Fig 7 (event) responses: both use
    the identical `{"action": "generate"|"skip", ...}` schema."""
    status = classify_output(raw)
    if status != "looks_json":
        return ExtractionOutcome(decision=Decision.SKIP, reason=f"malformed_output:{status}")
    try:
        payload = extract_json(raw)
    except ValueError as exc:
        return ExtractionOutcome(decision=Decision.SKIP, reason=f"unparseable_json:{exc}")

    action = payload.get("action")
    if action == "skip":
        return ExtractionOutcome(decision=Decision.SKIP, reason=payload.get("reason", ""))
    if action == "generate":
        skill, error = _parse_skill_payload(payload.get("skill", {}))
        if skill is None:
            return ExtractionOutcome(decision=Decision.SKIP, reason=error)
        return ExtractionOutcome(decision=Decision.GENERATE, skill=skill, reason=payload.get("reason", ""))
    return ExtractionOutcome(decision=Decision.SKIP, reason=f"unknown_action:{action!r}")


def parse_evolution_response(raw: str) -> EvolutionOutcome:
    status = classify_output(raw)
    if status != "looks_json":
        return EvolutionOutcome(decision=Decision.SKIP, reason=f"malformed_output:{status}")
    try:
        payload = extract_json(raw)
    except ValueError as exc:
        return EvolutionOutcome(decision=Decision.SKIP, reason=f"unparseable_json:{exc}")

    action = payload.get("action")
    if action == "skip":
        return EvolutionOutcome(decision=Decision.SKIP, reason=payload.get("reason", ""))
    if action == "evolve":
        skill, error = _parse_skill_payload(payload.get("skill", {}))
        if skill is None:
            return EvolutionOutcome(decision=Decision.SKIP, reason=error)
        return EvolutionOutcome(
            decision=Decision.EVOLVE,
            target_skill_id=payload.get("target_skill_id"),
            skill=skill,
            reason=payload.get("reason", ""),
        )
    return EvolutionOutcome(decision=Decision.SKIP, reason=f"unknown_action:{action!r}")


def parse_maintenance_response(raw: str) -> MaintenanceOutcome:
    status = classify_output(raw)
    if status != "looks_json":
        return MaintenanceOutcome(decision=Decision.DROP, reason=f"malformed_output:{status}")
    try:
        payload = extract_json(raw)
    except ValueError as exc:
        return MaintenanceOutcome(decision=Decision.DROP, reason=f"unparseable_json:{exc}")

    action = payload.get("action")
    if action == "add":
        return MaintenanceOutcome(decision=Decision.ADD, reason=payload.get("reason", ""))
    if action == "drop":
        return MaintenanceOutcome(decision=Decision.DROP, reason=payload.get("reason", ""))
    if action == "merge":
        skill, error = _parse_skill_payload(payload.get("skill", {}))
        if skill is None:
            return MaintenanceOutcome(decision=Decision.DROP, reason=f"invalid_merge_payload:{error}")
        return MaintenanceOutcome(
            decision=Decision.MERGE,
            merge_target_skill_id=payload.get("merge_target_skill_id"),
            skill=skill,
            reason=payload.get("reason", ""),
        )
    return MaintenanceOutcome(decision=Decision.DROP, reason=f"unknown_action:{action!r}")


# -- the four stages ----------------------------------------------------


class TaskSkillExtractor:
    """Figure 6: task-level extraction from 2-3 related trajectories."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, trajectories: list[Trajectory]) -> ExtractionOutcome:
        prompt = build_task_extraction_prompt(trajectories)
        raw = self.llm.complete(TASK_EXTRACTION_SYSTEM_PROMPT, prompt)
        outcome = parse_extraction_response(raw)
        if outcome.decision == Decision.GENERATE and outcome.skill is not None:
            for trajectory in trajectories:
                outcome.skill.provenance.append(
                    Provenance(trajectory_id=trajectory.id, step_indices=trajectory.grounded_step_indices())
                )
        return outcome


class EventSkillExtractor:
    """Figure 7: event-driven extraction from a single full trajectory."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, trajectory: Trajectory) -> ExtractionOutcome:
        prompt = build_event_extraction_prompt(trajectory)
        raw = self.llm.complete(EVENT_EXTRACTION_SYSTEM_PROMPT, prompt)
        outcome = parse_extraction_response(raw)
        if outcome.decision == Decision.GENERATE and outcome.skill is not None:
            outcome.skill.provenance.append(
                Provenance(trajectory_id=trajectory.id, step_indices=trajectory.grounded_step_indices())
            )
        return outcome


class SkillEvolver:
    """Figure 8: revise one existing skill given new trajectory evidence."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def propose(self, existing_skills: list[Skill], trajectory: Trajectory) -> EvolutionOutcome:
        prompt = build_evolution_prompt(existing_skills, trajectory)
        raw = self.llm.complete(SKILL_EVOLUTION_SYSTEM_PROMPT, prompt)
        outcome = parse_evolution_response(raw)
        if outcome.decision == Decision.EVOLVE and outcome.skill is not None:
            outcome.skill.provenance.append(
                Provenance(trajectory_id=trajectory.id, step_indices=trajectory.grounded_step_indices())
            )
        return outcome


class SkillMaintainer:
    """Figure 9: decide add/merge/drop for a candidate against retrieved
    similar skills. Deliberately takes no trajectory -- the paper is
    explicit that maintenance decisions are judged on skill text alone."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decide(self, candidate: Skill, retrieved: list[Skill]) -> MaintenanceOutcome:
        prompt = build_maintenance_prompt(candidate, retrieved)
        raw = self.llm.complete(SKILL_MAINTENANCE_SYSTEM_PROMPT, prompt)
        return parse_maintenance_response(raw)
