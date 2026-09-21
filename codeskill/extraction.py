"""Skill extraction: turn a trajectory into candidate skill operations.

This is the "propose" half of the learnable manager policy: given a
trajectory (and, optionally, a summary of the current bank so the model can
prefer UPDATE/MERGE over ADD when something similar already exists), an LLM
is prompted to emit structured skill operations. Two safety nets sit between
the raw LLM output and anything reaching the bank:

  1. `classify_output` -- cheap, pre-parse classification of the response
     (empty / clearly not JSON / looks like JSON) so malformed manager
     output is logged with a reason instead of raising deep inside a JSON
     parser.
  2. The grounding rule -- any operation that cites trajectory steps must
     cite only steps that were actually *observed* (assistant action
     followed by a real tool result), never a predicted/hallucinated one.
     Ungrounded proposals are rejected here, before they ever reach the
     bank, and are recorded as `SkillOperation(op_type=NOOP, exclusion_reason=...)`.
"""
from __future__ import annotations

from typing import Optional

from codeskill.bank import SkillBank
from codeskill.llm import LLMClient, extract_json
from codeskill.schema import Granularity, OperationType, Provenance, Skill, SkillOperation, Trajectory

SYSTEM_PROMPT = """You are the skill-management policy of a self-evolving coding agent.
Given one agent trajectory (and, optionally, a summary of skills already in the bank),
decide what to do with it: extract new reusable skills, update an existing skill with
new evidence, merge redundant skills, or drop a skill that keeps failing.

Rules:
- Only cite step indices that appear in the trajectory's grounded (observed) steps.
  Never cite a step whose tool result was not actually observed.
- Prefer UPDATE or MERGE over ADD when a very similar skill already exists.
- Emit EVENT-granularity skills for a single reusable action pattern, and
  TASK-granularity skills for a whole reusable episode strategy.
- Respond with a single JSON object: {"operations": [ ... ]}. Each operation has:
  "op_type" (add|update|merge|drop|noop), "name", "description", "steps" (list[str]),
  "granularity" (task|event), "cited_step_indices" (list[int]), "preconditions" (list[str]),
  "tags" (list[str]), "target_skill_id" (for update/drop), "merge_with" (list[str], for merge),
  "rationale" (str)."""


def classify_output(text: str) -> str:
    """Cheap pre-parse triage of a manager LLM response."""
    stripped = text.strip()
    if not stripped:
        return "empty"
    if "{" not in stripped or "}" not in stripped:
        return "not_json"
    return "looks_json"


def build_extraction_prompt(trajectory: Trajectory, bank_summary: str = "") -> str:
    lines = [f"Task: {trajectory.task_description}", f"Outcome: {'success' if trajectory.success else 'failure'}", ""]
    for step in trajectory.steps:
        grounded = "OBSERVED" if step.is_grounded() else "UNOBSERVED"
        lines.append(f"[{step.index}] ({grounded}) assistant: {step.assistant_action}")
        if step.tool_name:
            lines.append(f"      tool_call: {step.tool_name}({step.tool_input or {}})")
        if step.tool_result is not None:
            lines.append(f"      tool_result: {step.tool_result}")
    if bank_summary:
        lines.append("\nExisting skill bank summary:\n" + bank_summary)
    return "\n".join(lines)


class SkillExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def summarize_bank(self, bank: SkillBank, limit: int = 20) -> str:
        skills = bank.active_skills()[:limit]
        return "\n".join(f"- ({s.id}) [{s.granularity.value}] {s.name}: {s.description}" for s in skills)

    def propose_operations(self, trajectory: Trajectory, bank: Optional[SkillBank] = None) -> list[SkillOperation]:
        bank_summary = self.summarize_bank(bank) if bank is not None else ""
        prompt = build_extraction_prompt(trajectory, bank_summary)
        raw = self.llm.complete(SYSTEM_PROMPT, prompt)
        return parse_operations(raw, trajectory)


def _build_operation(raw_op: dict, trajectory: Trajectory) -> SkillOperation:
    try:
        op_type = OperationType(raw_op.get("op_type", "noop"))
    except ValueError:
        return SkillOperation(op_type=OperationType.NOOP, exclusion_reason=f"unknown_op_type:{raw_op.get('op_type')}")

    cited = [int(i) for i in raw_op.get("cited_step_indices", [])]
    rationale = raw_op.get("rationale", "")

    if op_type in (OperationType.ADD, OperationType.UPDATE) and cited:
        if not trajectory.is_grounded_span(cited):
            return SkillOperation(
                op_type=OperationType.NOOP,
                rationale=rationale,
                exclusion_reason=f"ungrounded_citation:{cited}",
            )

    if op_type == OperationType.DROP:
        return SkillOperation(
            op_type=op_type,
            target_skill_id=raw_op.get("target_skill_id"),
            rationale=rationale,
        )

    if op_type == OperationType.MERGE:
        return SkillOperation(
            op_type=op_type,
            target_skill_id=raw_op.get("target_skill_id"),
            merge_with=list(raw_op.get("merge_with", [])),
            rationale=rationale,
        )

    if op_type == OperationType.NOOP:
        return SkillOperation(op_type=op_type, rationale=rationale)

    # ADD or UPDATE with a concrete skill payload.
    try:
        granularity = Granularity(raw_op.get("granularity", "event"))
    except ValueError:
        granularity = Granularity.EVENT

    skill = Skill(
        name=raw_op.get("name", "unnamed_skill"),
        description=raw_op.get("description", ""),
        steps=list(raw_op.get("steps", [])),
        granularity=granularity,
        preconditions=list(raw_op.get("preconditions", [])),
        tags=list(raw_op.get("tags", [])),
        provenance=[Provenance(trajectory_id=trajectory.id, step_indices=cited, note=rationale)],
    )
    return SkillOperation(
        op_type=op_type,
        skill=skill,
        target_skill_id=raw_op.get("target_skill_id"),
        rationale=rationale,
    )


def parse_operations(raw: str, trajectory: Trajectory) -> list[SkillOperation]:
    """Parse a raw manager-policy completion into `SkillOperation`s, applying
    the same pre-parse classification and grounding checks used at
    inference time. Shared by `SkillExtractor.propose_operations` (live LLM
    call) and `codeskill.grpo` (scoring sampled rollouts during training),
    so training-time and inference-time validation can never drift apart.
    """
    status = classify_output(raw)
    if status != "looks_json":
        return [SkillOperation(op_type=OperationType.NOOP, rationale=raw, exclusion_reason=f"malformed_output:{status}")]

    try:
        payload = extract_json(raw)
    except ValueError as exc:
        return [SkillOperation(op_type=OperationType.NOOP, rationale=str(exc), exclusion_reason="unparseable_json")]

    return [_build_operation(raw_op, trajectory) for raw_op in payload.get("operations", [])]
