"""The learnable skill-manager policy: extraction + bank maintenance.

At inference time this is "extractor proposes, manager applies": the
`SkillExtractor` LLM call proposes operations, and `SkillManagerPolicy`
applies them to a `SkillBank`, handling the bookkeeping (rejecting
duplicates, resolving merge/update targets, running compaction at the end
of a round) that keeps the bank both useful and bounded.

During RL training (see `codeskill.grpo`), it's this same propose+apply
policy whose *proposals* are sampled, scored by the hybrid reward, and
optimized -- the apply-side bookkeeping here is deterministic and not
itself learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from codeskill.bank import SkillBank
from codeskill.extraction import SkillExtractor
from codeskill.schema import OperationType, SkillOperation, Trajectory


@dataclass
class OperationOutcome:
    operation: SkillOperation
    applied: bool
    detail: str = ""


@dataclass
class RoundReport:
    round_num: int
    outcomes: list[OperationOutcome] = field(default_factory=list)
    compacted_skill_ids: list[str] = field(default_factory=list)
    bank_size_before: int = 0
    bank_size_after: int = 0


class SkillManagerPolicy:
    def __init__(self, extractor: SkillExtractor, bank: SkillBank):
        self.extractor = extractor
        self.bank = bank

    def apply_operation(self, op: SkillOperation) -> OperationOutcome:
        if op.op_type == OperationType.NOOP:
            return OperationOutcome(op, applied=False, detail=op.exclusion_reason or "noop")

        if op.op_type == OperationType.ADD:
            assert op.skill is not None
            accepted, reason = self.bank.add(op.skill)
            return OperationOutcome(op, applied=accepted, detail=reason or "added")

        if op.op_type == OperationType.UPDATE:
            assert op.skill is not None
            if op.target_skill_id is None or self.bank.get(op.target_skill_id) is None:
                # No valid target: fall back to treating this as a new skill
                # rather than silently dropping the (grounded) evidence.
                accepted, reason = self.bank.add(op.skill)
                return OperationOutcome(op, applied=accepted, detail=reason or "added_as_fallback")
            new_provenance = op.skill.provenance[0] if op.skill.provenance else None
            self.bank.update(
                op.target_skill_id,
                new_provenance=new_provenance,
                description=op.skill.description or self.bank.get(op.target_skill_id).description,
                steps=list(dict.fromkeys(self.bank.get(op.target_skill_id).steps + op.skill.steps)),
            )
            return OperationOutcome(op, applied=True, detail=f"updated:{op.target_skill_id}")

        if op.op_type == OperationType.MERGE:
            if op.target_skill_id is None or self.bank.get(op.target_skill_id) is None or not op.merge_with:
                return OperationOutcome(op, applied=False, detail="invalid_merge_target")
            self.bank.merge(op.target_skill_id, op.merge_with, rationale=op.rationale)
            return OperationOutcome(op, applied=True, detail=f"merged_into:{op.target_skill_id}")

        if op.op_type == OperationType.DROP:
            if op.target_skill_id is None or self.bank.get(op.target_skill_id) is None:
                return OperationOutcome(op, applied=False, detail="invalid_drop_target")
            self.bank.drop(op.target_skill_id, op.rationale or "manager_decision")
            return OperationOutcome(op, applied=True, detail=f"dropped:{op.target_skill_id}")

        return OperationOutcome(op, applied=False, detail=f"unhandled_op_type:{op.op_type}")

    def process_trajectory(self, trajectory: Trajectory) -> list[OperationOutcome]:
        proposals = self.extractor.propose_operations(trajectory, bank=self.bank)
        return [self.apply_operation(op) for op in proposals]

    def run_round(
        self,
        trajectories: list[Trajectory],
        *,
        compact_after: bool = True,
        min_uses: int = 3,
        min_success_rate: float = 0.2,
    ) -> RoundReport:
        report = RoundReport(round_num=self.bank.current_round, bank_size_before=len(self.bank))
        for trajectory in trajectories:
            report.outcomes.extend(self.process_trajectory(trajectory))
        if compact_after:
            report.compacted_skill_ids = self.bank.compact(min_uses=min_uses, min_success_rate=min_success_rate)
        report.bank_size_after = len(self.bank)
        self.bank.advance_round()
        return report
