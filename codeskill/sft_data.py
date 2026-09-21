"""Warm-start SFT dataset construction.

Per the paper, training starts by warm-starting the manager policy with
supervised data built from coding-agent trajectories paired with
teacher-generated skill operations (i.e. a stronger model's extract/update/
merge/drop decisions), before the RL (GRPO) stage takes over. This module
builds that dataset in the same (system, prompt) -> completion shape used at
inference time by `codeskill.extraction`, so a base model fine-tuned on it
can be dropped in as the manager policy directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from codeskill.extraction import SYSTEM_PROMPT, SkillExtractor, build_extraction_prompt
from codeskill.schema import OperationType, SkillOperation, Trajectory


@dataclass
class SFTExample:
    system: str
    prompt: str
    completion: str  # the target JSON string the policy should learn to produce


def operation_to_dict(op: SkillOperation) -> dict:
    d: dict = {"op_type": op.op_type.value, "rationale": op.rationale}
    if op.skill is not None:
        d.update(
            {
                "name": op.skill.name,
                "description": op.skill.description,
                "steps": op.skill.steps,
                "granularity": op.skill.granularity.value,
                "cited_step_indices": op.skill.provenance[0].step_indices if op.skill.provenance else [],
                "preconditions": op.skill.preconditions,
                "tags": op.skill.tags,
            }
        )
    if op.target_skill_id is not None:
        d["target_skill_id"] = op.target_skill_id
    if op.merge_with:
        d["merge_with"] = op.merge_with
    return d


def build_example(trajectory: Trajectory, operations: list[SkillOperation], bank_summary: str = "") -> SFTExample:
    prompt = build_extraction_prompt(trajectory, bank_summary)
    completion = json.dumps({"operations": [operation_to_dict(op) for op in operations]}, indent=2)
    return SFTExample(system=SYSTEM_PROMPT, prompt=prompt, completion=completion)


def generate_teacher_dataset(
    trajectories: list[Trajectory],
    teacher_extractor: SkillExtractor,
    bank_summary: str = "",
) -> list[tuple[Trajectory, list[SkillOperation]]]:
    """Have a (presumably stronger) teacher model label each trajectory with
    the skill operations it would take; only grounded, well-formed
    operations survive `SkillExtractor.propose_operations`'s own checks."""
    labeled = []
    for trajectory in trajectories:
        ops = teacher_extractor.propose_operations(trajectory)
        labeled.append((trajectory, ops))
    return labeled


def build_warm_start_dataset(
    labeled: list[tuple[Trajectory, list[SkillOperation]]],
    bank_summary: str = "",
    *,
    drop_pure_noop: bool = True,
) -> list[SFTExample]:
    """Turn teacher-labeled (trajectory, operations) pairs into SFT examples.

    `drop_pure_noop` skips trajectories where the teacher produced only
    NOOP/rejected operations -- they carry no positive supervision signal
    for what a *good* extraction looks like, only that nothing was worth
    extracting, which is a much rarer class worth downweighting rather than
    dominating the warm-start set.
    """
    examples = []
    for trajectory, ops in labeled:
        if drop_pure_noop and all(op.op_type == OperationType.NOOP for op in ops):
            continue
        examples.append(build_example(trajectory, ops, bank_summary))
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
