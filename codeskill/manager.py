"""The learnable skill-manager policy: extraction/evolution + bank maintenance.

At inference time this is "extract or evolve, then maintain": an
extraction (Fig 6/7) or evolution (Fig 8) call proposes a candidate skill,
and a maintenance call (Fig 9) decides how it enters the bank
(add/merge/drop) by comparing it against retrieved similar skills. The
paper is explicit that this applies to *both* sources -- "each newly
extracted or evolved candidate skill is further passed to a maintenance
stage" (Section 3.2, repeated in Appendix C) -- so `process_evolution`
below routes its revised candidate through the same `_maintain` step as
extraction, rather than writing the revision straight to the bank.

For an evolution-sourced candidate, maintenance's `add` decision commits
the revision (replacing the skill the evolver named as its target);
`merge` still means "fold into whichever retrieved skill the maintainer
chose" (which may or may not be that same target); `drop` rejects the
revision and leaves the target unchanged.

During RL training (see `codeskill.grpo`), it's these same LLM calls whose
*proposals* are sampled, scored by the hybrid reward, and optimized -- the
bank-mutation bookkeeping here (dedup, replace-in-place, compaction) is
deterministic and not itself learned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from codeskill.bank import SkillBank
from codeskill.extraction import EventSkillExtractor, SkillEvolver, SkillMaintainer, TaskSkillExtractor
from codeskill.llm import LLMClient
from codeskill.schema import Decision, Skill, Trajectory


@dataclass
class StageOutcome:
    stage: str  # "extract_event" | "extract_task" | "maintain_add" | "maintain_merge" | "maintain_drop" | "evolve"
    detail: str
    applied: bool
    skill_id: Optional[str] = None


@dataclass
class RoundReport:
    round_num: int
    stage_outcomes: list[StageOutcome] = field(default_factory=list)
    compacted_skill_ids: list[str] = field(default_factory=list)
    bank_size_before: int = 0
    bank_size_after: int = 0


class SkillManagerPolicy:
    def __init__(
        self,
        bank: SkillBank,
        event_extractor: EventSkillExtractor,
        task_extractor: Optional[TaskSkillExtractor] = None,
        evolver: Optional[SkillEvolver] = None,
        maintainer: Optional[SkillMaintainer] = None,
        retrieval_top_k: int = 5,
    ):
        self.bank = bank
        self.event_extractor = event_extractor
        self.task_extractor = task_extractor
        self.evolver = evolver
        self.maintainer = maintainer
        self.retrieval_top_k = retrieval_top_k

    @classmethod
    def from_llm(cls, llm: LLMClient, bank: SkillBank, retrieval_top_k: int = 5) -> "SkillManagerPolicy":
        """Convenience constructor: one LLM client backing all four stages
        (a `MockLLMClient` dispatches by matching each call's system prompt;
        a real client just gets called four different ways)."""
        return cls(
            bank=bank,
            event_extractor=EventSkillExtractor(llm),
            task_extractor=TaskSkillExtractor(llm),
            evolver=SkillEvolver(llm),
            maintainer=SkillMaintainer(llm),
            retrieval_top_k=retrieval_top_k,
        )

    def _maintain(self, candidate: Skill, *, replace_target_id: Optional[str] = None) -> list[StageOutcome]:
        """Runs the maintenance decision (Fig 9) for `candidate`.

        `replace_target_id`, when set, marks this candidate as an
        evolution's revision rather than a fresh extraction: an `add`
        decision then commits the revision onto that target (via
        `bank.replace`) instead of inserting a brand-new skill, and a
        `drop` decision is reported as keeping the target unchanged rather
        than as rejecting a would-be-new skill.
        """
        if self.maintainer is None:
            raise ValueError("no SkillMaintainer configured")
        retrieved = [r.skill for r in self.bank.retrieve_similar(candidate, top_k=self.retrieval_top_k)]
        decision = self.maintainer.decide(candidate, retrieved)

        if decision.decision == Decision.ADD:
            if replace_target_id is not None:
                if self.bank.get(replace_target_id) is None:
                    return [StageOutcome("maintain_add", f"invalid_replace_target:{replace_target_id}", applied=False)]
                self.bank.replace(replace_target_id, candidate, rationale=decision.reason)
                return [
                    StageOutcome(
                        "maintain_add", f"committed_revision:{replace_target_id}", applied=True, skill_id=replace_target_id
                    )
                ]
            accepted, reason = self.bank.add(candidate)
            return [
                StageOutcome(
                    "maintain_add", reason or "added", applied=accepted, skill_id=candidate.id if accepted else None
                )
            ]

        if decision.decision == Decision.MERGE:
            if (
                decision.merge_target_skill_id is None
                or self.bank.get(decision.merge_target_skill_id) is None
                or decision.skill is None
            ):
                return [StageOutcome("maintain_merge", "invalid_merge_target", applied=False)]
            self.bank.replace(decision.merge_target_skill_id, decision.skill, rationale=decision.reason)
            return [
                StageOutcome(
                    "maintain_merge",
                    f"merged_into:{decision.merge_target_skill_id}",
                    applied=True,
                    skill_id=decision.merge_target_skill_id,
                )
            ]

        # DROP (or a malformed response, which parse_maintenance_response also maps to DROP)
        detail = decision.reason or "dropped_candidate"
        if replace_target_id is not None:
            detail = f"kept_existing_unchanged:{replace_target_id} ({detail})"
        return [StageOutcome("maintain_drop", detail, applied=False)]

    def process_event_trajectory(self, trajectory: Trajectory) -> list[StageOutcome]:
        outcome = self.event_extractor.propose(trajectory)
        if outcome.decision != Decision.GENERATE or outcome.skill is None:
            return [StageOutcome("extract_event", outcome.reason or "skip", applied=False)]
        return [StageOutcome("extract_event", "generated candidate", applied=True)] + self._maintain(outcome.skill)

    def process_task_trajectories(self, trajectories: list[Trajectory]) -> list[StageOutcome]:
        if self.task_extractor is None:
            raise ValueError("no TaskSkillExtractor configured")
        outcome = self.task_extractor.propose(trajectories)
        if outcome.decision != Decision.GENERATE or outcome.skill is None:
            return [StageOutcome("extract_task", outcome.reason or "skip", applied=False)]
        return [StageOutcome("extract_task", "generated candidate", applied=True)] + self._maintain(outcome.skill)

    def process_evolution(self, existing_skills: list[Skill], trajectory: Trajectory) -> list[StageOutcome]:
        if self.evolver is None:
            raise ValueError("no SkillEvolver configured")
        outcome = self.evolver.propose(existing_skills, trajectory)
        if outcome.decision != Decision.EVOLVE or outcome.skill is None or outcome.target_skill_id is None:
            return [StageOutcome("evolve", outcome.reason or "skip", applied=False)]
        if self.bank.get(outcome.target_skill_id) is None:
            return [StageOutcome("evolve", f"invalid_target:{outcome.target_skill_id}", applied=False)]
        proposed = StageOutcome("evolve", "revised candidate", applied=True, skill_id=outcome.target_skill_id)
        return [proposed] + self._maintain(outcome.skill, replace_target_id=outcome.target_skill_id)

    def run_round(
        self,
        trajectories: list[Trajectory],
        *,
        compact_after: bool = True,
        min_uses: int = 3,
        min_success_rate: float = 0.2,
    ) -> RoundReport:
        """Runs event-level extraction+maintenance over every trajectory in
        the batch. Use `process_task_trajectories` directly (with your own
        grouping of related trajectories) for task-level extraction, and
        `process_evolution` for revising a specific existing skill --
        neither is folded into this default round since both need
        information (a trajectory grouping, or a target skill) this method
        doesn't have.
        """
        report = RoundReport(round_num=self.bank.current_round, bank_size_before=len(self.bank))
        for trajectory in trajectories:
            report.stage_outcomes.extend(self.process_event_trajectory(trajectory))
        if compact_after:
            report.compacted_skill_ids = self.bank.compact(min_uses=min_uses, min_success_rate=min_success_rate)
        report.bank_size_after = len(self.bank)
        self.bank.advance_round()
        return report
