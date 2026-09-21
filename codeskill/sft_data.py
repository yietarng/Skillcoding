"""Warm-start SFT dataset construction.

Per the paper, training starts by warm-starting the manager policy with
supervised data built from coding-agent trajectories paired with
teacher-generated skill operations (a stronger model's decisions), before
the RL (GRPO) stage takes over. This module builds that dataset separately
for each of the four appendix-prompt stages (task extraction, event
extraction, evolution, maintenance), in the same (system, prompt) ->
completion shape used at inference time, so a base model fine-tuned on it
can be dropped in as that stage's policy directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from codeskill.extraction import (
    EventSkillExtractor,
    SkillEvolver,
    SkillMaintainer,
    TaskSkillExtractor,
    build_event_extraction_prompt,
    build_evolution_prompt,
    build_maintenance_prompt,
    build_task_extraction_prompt,
)
from codeskill.prompts import (
    EVENT_EXTRACTION_SYSTEM_PROMPT,
    SKILL_EVOLUTION_SYSTEM_PROMPT,
    SKILL_MAINTENANCE_SYSTEM_PROMPT,
    TASK_EXTRACTION_SYSTEM_PROMPT,
)
from codeskill.schema import Decision, EvolutionOutcome, ExtractionOutcome, MaintenanceOutcome, Skill, Trajectory


@dataclass
class SFTExample:
    system: str
    prompt: str
    completion: str  # the target JSON string the policy should learn to produce


def extraction_outcome_to_dict(outcome: ExtractionOutcome) -> dict:
    if outcome.decision == Decision.GENERATE and outcome.skill is not None:
        return {"action": "generate", "skill": outcome.skill.to_paper_dict()}
    return {"action": "skip", "reason": outcome.reason}


def evolution_outcome_to_dict(outcome: EvolutionOutcome) -> dict:
    if outcome.decision == Decision.EVOLVE and outcome.skill is not None:
        return {
            "action": "evolve",
            "target_skill_id": outcome.target_skill_id,
            "reason": outcome.reason,
            "skill": outcome.skill.to_paper_dict(),
        }
    return {"action": "skip", "reason": outcome.reason}


def maintenance_outcome_to_dict(outcome: MaintenanceOutcome) -> dict:
    if outcome.decision == Decision.MERGE and outcome.skill is not None:
        return {
            "action": "merge",
            "merge_target_skill_id": outcome.merge_target_skill_id,
            "reason": outcome.reason,
            "skill": outcome.skill.to_paper_dict(),
        }
    return {"action": outcome.decision.value, "reason": outcome.reason}


def _keep(drop_pure_skip: bool, decision: Decision) -> bool:
    return not (drop_pure_skip and decision in (Decision.SKIP, Decision.DROP))


def build_event_extraction_dataset(
    trajectories: list[Trajectory], teacher: EventSkillExtractor, *, drop_pure_skip: bool = True
) -> list[SFTExample]:
    examples = []
    for trajectory in trajectories:
        outcome = teacher.propose(trajectory)
        if not _keep(drop_pure_skip, outcome.decision):
            continue
        prompt = build_event_extraction_prompt(trajectory)
        examples.append(SFTExample(EVENT_EXTRACTION_SYSTEM_PROMPT, prompt, json.dumps(extraction_outcome_to_dict(outcome))))
    return examples


def build_task_extraction_dataset(
    trajectory_groups: list[list[Trajectory]], teacher: TaskSkillExtractor, *, drop_pure_skip: bool = True
) -> list[SFTExample]:
    examples = []
    for group in trajectory_groups:
        outcome = teacher.propose(group)
        if not _keep(drop_pure_skip, outcome.decision):
            continue
        prompt = build_task_extraction_prompt(group)
        examples.append(SFTExample(TASK_EXTRACTION_SYSTEM_PROMPT, prompt, json.dumps(extraction_outcome_to_dict(outcome))))
    return examples


def build_evolution_dataset(
    cases: list[tuple[list[Skill], Trajectory]], teacher: SkillEvolver, *, drop_pure_skip: bool = True
) -> list[SFTExample]:
    examples = []
    for existing_skills, trajectory in cases:
        outcome = teacher.propose(existing_skills, trajectory)
        if not _keep(drop_pure_skip, outcome.decision):
            continue
        prompt = build_evolution_prompt(existing_skills, trajectory)
        examples.append(SFTExample(SKILL_EVOLUTION_SYSTEM_PROMPT, prompt, json.dumps(evolution_outcome_to_dict(outcome))))
    return examples


def build_maintenance_dataset(
    cases: list[tuple[Skill, list[Skill]]], teacher: SkillMaintainer, *, drop_pure_skip: bool = False
) -> list[SFTExample]:
    """`drop_pure_skip` defaults to False here since `drop` is itself a
    meaningful, common maintenance decision (unlike `skip` in extraction),
    not a "nothing to learn from" case."""
    examples = []
    for candidate, retrieved in cases:
        outcome = teacher.decide(candidate, retrieved)
        if not _keep(drop_pure_skip, outcome.decision):
            continue
        prompt = build_maintenance_prompt(candidate, retrieved)
        examples.append(
            SFTExample(SKILL_MAINTENANCE_SYSTEM_PROMPT, prompt, json.dumps(maintenance_outcome_to_dict(outcome)))
        )
    return examples


def export_jsonl(examples: list[SFTExample], path: str | Path) -> None:
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps({"system": ex.system, "prompt": ex.prompt, "completion": ex.completion}) + "\n")


def load_jsonl(path: str | Path) -> list[SFTExample]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            examples.append(SFTExample(**d))
    return examples
