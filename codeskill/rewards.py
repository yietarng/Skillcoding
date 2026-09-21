"""The hybrid reward: dense rubric-judge quality scores + sparse execution
feedback, matching the paper's five appendix judge prompts (Figures 10-14,
see `codeskill/prompts.py`) plus a verifiable pass/fail signal from the
frozen downstream agent.

Each judge prompt asks the *judge LLM itself* to count yes-answers per
dimension and emit the resulting integer score directly (not raw yes/no
answers for us to tally) -- so `JudgeScore.overall()` just normalizes the
returned per-dimension integers by their stated max.

- `TaskQualityJudge` / `EventQualityJudge` -- score a freshly extracted
  candidate skill against the trajectory evidence it came from (Fig 10/11).
- `EvolutionQualityJudge` -- score a proposed revision against the existing
  skill and new trajectory evidence (Fig 12).
- `MergeQualityJudge` -- score a proposed merge purely on skill text, no
  trajectory (Fig 13).
- `BehaviorAlignmentJudge` -- score whether a downstream rollout actually
  reflects the provided skill, not just competent behavior (Fig 14). This
  is the "behavior-skill alignment" signal the paper's abstract references.
- `ExecutionReward` -- sparse, verifiable: the frozen downstream agent's
  pass rate on held-out tasks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from codeskill.downstream_agent import FrozenDownstreamAgent
from codeskill.extraction import format_skill_ref, format_trajectory
from codeskill.llm import LLMClient, extract_json
from codeskill.prompts import (
    BEHAVIOR_ALIGNMENT_DIMENSIONS,
    BEHAVIOR_ALIGNMENT_JUDGE_PROMPT,
    EVENT_QUALITY_DIMENSIONS,
    EVENT_QUALITY_JUDGE_PROMPT,
    EVOLUTION_QUALITY_DIMENSIONS,
    EVOLUTION_QUALITY_JUDGE_PROMPT,
    MERGE_QUALITY_DIMENSIONS,
    MERGE_JUDGE_PROMPT,
    SKILL_EVOLUTION_SYSTEM_PROMPT,
    TASK_QUALITY_DIMENSIONS,
    TASK_QUALITY_JUDGE_PROMPT,
)
from codeskill.schema import Skill, Trajectory


@dataclass
class JudgeScore:
    dimensions: dict[str, int]
    dimension_maxes: dict[str, int]
    reason: str = ""

    def overall(self) -> float:
        max_possible = sum(self.dimension_maxes.values())
        if max_possible == 0:
            return 0.0
        achieved = sum(min(self.dimensions.get(k, 0), m) for k, m in self.dimension_maxes.items())
        return achieved / max_possible


class _RubricJudge:
    system_prompt: str = ""
    dimension_maxes: dict[str, int] = {}

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _score_from_raw(self, raw: str) -> JudgeScore:
        payload = extract_json(raw)
        dims = {k: int(payload.get(k, 0)) for k in self.dimension_maxes}
        return JudgeScore(dimensions=dims, dimension_maxes=self.dimension_maxes, reason=payload.get("reason", ""))


class TaskQualityJudge(_RubricJudge):
    """Figure 10."""

    system_prompt = TASK_QUALITY_JUDGE_PROMPT
    dimension_maxes = TASK_QUALITY_DIMENSIONS

    def score(self, trajectories: list[Trajectory], proposed_skill: Skill) -> JudgeScore:
        evidence = "\n\n".join(format_trajectory(t) for t in trajectories)
        prompt = f"TRAJECTORY_EVIDENCE:\n{evidence}\n\nPROPOSED_OUTPUT:\n{json.dumps(proposed_skill.to_paper_dict())}"
        raw = self.llm.complete(self.system_prompt, prompt)
        return self._score_from_raw(raw)


class EventQualityJudge(_RubricJudge):
    """Figure 11."""

    system_prompt = EVENT_QUALITY_JUDGE_PROMPT
    dimension_maxes = EVENT_QUALITY_DIMENSIONS

    def score(self, trajectory: Trajectory, proposed_skill: Skill) -> JudgeScore:
        prompt = (
            f"TRAJECTORY_EVIDENCE:\n{format_trajectory(trajectory)}\n\n"
            f"PROPOSED_OUTPUT:\n{json.dumps(proposed_skill.to_paper_dict())}"
        )
        raw = self.llm.complete(self.system_prompt, prompt)
        return self._score_from_raw(raw)


class EvolutionQualityJudge(_RubricJudge):
    """Figure 12. Notably takes the evolution system prompt itself as part
    of its evaluation input, per the appendix's `{{SYSTEM_PROMPT}}` slot."""

    system_prompt = EVOLUTION_QUALITY_JUDGE_PROMPT
    dimension_maxes = EVOLUTION_QUALITY_DIMENSIONS

    def score(self, existing_skill: Skill, trajectory: Trajectory, proposed_skill: Skill) -> JudgeScore:
        prompt = (
            f"SYSTEM_PROMPT:\n{SKILL_EVOLUTION_SYSTEM_PROMPT}\n\n"
            f"EXISTING_PRIOR_KNOWLEDGE:\n{format_skill_ref(existing_skill)}\n\n"
            f"TRAJECTORY_EVIDENCE:\n{format_trajectory(trajectory)}\n\n"
            f"PROPOSED_OUTPUT:\n{json.dumps(proposed_skill.to_paper_dict())}"
        )
        raw = self.llm.complete(self.system_prompt, prompt)
        return self._score_from_raw(raw)


class MergeQualityJudge(_RubricJudge):
    """Figure 13. Deliberately takes no trajectory: "maintain decisions are
    made without a trajectory" per the appendix."""

    system_prompt = MERGE_JUDGE_PROMPT
    dimension_maxes = MERGE_QUALITY_DIMENSIONS

    def score(self, candidate: Skill, existing_skills: list[Skill], merge_target_skill_id: str, merged_skill: Skill) -> JudgeScore:
        existing_block = "\n".join(f"  {format_skill_ref(s)}" for s in existing_skills)
        proposed_output = {
            "action": "merge",
            "merge_target_skill_id": merge_target_skill_id,
            "skill": merged_skill.to_paper_dict(),
        }
        prompt = (
            f"CANDIDATE_PRIOR_KNOWLEDGE:\n{json.dumps(candidate.to_paper_dict())}\n\n"
            f"EXISTING_PRIOR_KNOWLEDGE:\n{existing_block}\n\n"
            f"PROPOSED_OUTPUT:\n{json.dumps(proposed_output)}"
        )
        raw = self.llm.complete(self.system_prompt, prompt)
        return self._score_from_raw(raw)


class BehaviorAlignmentJudge(_RubricJudge):
    """Figure 14: does a downstream rollout actually reflect the provided
    skill, versus merely being competent."""

    system_prompt = BEHAVIOR_ALIGNMENT_JUDGE_PROMPT
    dimension_maxes = BEHAVIOR_ALIGNMENT_DIMENSIONS

    def score(
        self,
        task_context: str,
        user_prompt: str,
        prior_knowledge: Skill,
        trajectory: Trajectory,
        result_summary: str,
    ) -> JudgeScore:
        prompt = (
            f"TASK_CONTEXT:\n{task_context}\n\n"
            f"USER_PROMPT:\n{user_prompt}\n\n"
            f"PRIOR_KNOWLEDGE:\n{format_skill_ref(prior_knowledge)}\n\n"
            f"TRAJECTORY_EVIDENCE:\n{format_trajectory(trajectory)}\n\n"
            f"RESULT_SUMMARY:\n{result_summary}"
        )
        raw = self.llm.complete(self.system_prompt, prompt)
        return self._score_from_raw(raw)


class ExecutionReward:
    """Runs the frozen downstream agent on a fixed set of held-out tasks and
    reports the pass rate: the sparse, verifiable half of the hybrid reward."""

    def __init__(self, agent: FrozenDownstreamAgent, eval_tasks: list[str]):
        self.agent = agent
        self.eval_tasks = eval_tasks

    def score(self) -> float:
        if not self.eval_tasks:
            return 0.0
        results = [self.agent.attempt(t, record_usage=False) for t in self.eval_tasks]
        return sum(1.0 for r in results if r.success) / len(results)


@dataclass
class RewardBreakdown:
    quality_reward: float
    execution_reward: Optional[float]
    alignment_reward: Optional[float]
    total: float
    quality_detail: Optional[JudgeScore] = None
    alignment_detail: Optional[JudgeScore] = None


class HybridReward:
    """Combines a stage-appropriate quality `JudgeScore` with the sparse
    execution pass rate and (optionally) a behavior-alignment `JudgeScore`.

    The paper's abstract describes "a hybrid reward that combines dense
    rubric-based skill-quality feedback with sparse verifiable execution
    feedback from the frozen downstream agent" -- the exact weighting
    between the two is not given in what was accessible of the paper, so
    the defaults here are a reasonable choice, not a reproduction of a
    reported hyperparameter.
    """

    def __init__(self, quality_weight: float = 0.5, execution_weight: float = 0.4, alignment_weight: float = 0.1):
        self.quality_weight = quality_weight
        self.execution_weight = execution_weight
        self.alignment_weight = alignment_weight

    def combine(
        self,
        quality: JudgeScore,
        execution: Optional[float] = None,
        alignment: Optional[JudgeScore] = None,
    ) -> RewardBreakdown:
        total = self.quality_weight * quality.overall()
        if execution is not None:
            total += self.execution_weight * execution
        if alignment is not None:
            total += self.alignment_weight * alignment.overall()
        return RewardBreakdown(
            quality_reward=quality.overall(),
            execution_reward=execution,
            alignment_reward=alignment.overall() if alignment else None,
            total=total,
            quality_detail=quality,
            alignment_detail=alignment,
        )
