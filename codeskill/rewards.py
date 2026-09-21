"""The hybrid reward: dense rubric-judge quality scores + sparse execution
feedback, matching the paper's five appendix judge prompts (Figures 10-14,
see `codeskill/prompts.py`) and Section 3.3.2 / Algorithm 1's reward formula,
now verified against the primary PDF (not just a third-party transcription).

Each judge prompt asks the *judge LLM itself* to count yes-answers per
dimension and emit the resulting integer score directly (not raw yes/no
answers for us to tally) -- so `JudgeScore.overall()` just normalizes the
returned per-dimension integers by their stated max, matching the paper's
"number of satisfied yes/no questions divided by total number of yes/no
questions in the rubric" (Appendix B).

- `TaskQualityJudge` / `EventQualityJudge` -- score a freshly extracted
  candidate skill against the trajectory evidence it came from (Fig 10/11).
- `EvolutionQualityJudge` -- score a proposed revision against the existing
  skill and new trajectory evidence (Fig 12).
- `MergeQualityJudge` -- score a proposed merge purely on skill text, no
  trajectory (Fig 13).
- `BehaviorAlignmentJudge` -- score whether a downstream rollout actually
  reflects the provided skill, not just competent behavior (Fig 14).
- `NoSkillBaselineCache` / `ExecutionReward` -- Algorithm 1's pre-cached
  no-skill baseline (average of `n` no-skill rollouts per task) and the
  reverse-retrieval execution reward `V(skill-conditioned rollout) - baseline`.
- `HybridReward` -- Algorithm 1's exact combination rule, not an
  independently-weighted sum (see its docstring for the formula and why an
  earlier version of this file got it wrong).
"""
from __future__ import annotations

import json
import random
import statistics
from dataclasses import dataclass
from typing import Optional

from codeskill.downstream_agent import SolveFn
from codeskill.eval.harness import BenchmarkAdapter, BenchmarkTask
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


class NoSkillBaselineCache:
    """Algorithm 1, lines 1-4: `b_pi(x) = (1/n) * sum_i V(tau_0_i(x))`.

    Precomputes and caches, per task, the average verifier score over `n`
    no-skill rollouts (the paper uses n=4, Appendix B). Cached so the same
    baseline is reused however many times different candidate skills'
    reverse retrieval lands on that task.
    """

    def __init__(self, no_skill_solve_fn: SolveFn, adapter: BenchmarkAdapter, n: int = 4):
        self.no_skill_solve_fn = no_skill_solve_fn
        self.adapter = adapter
        self.n = n
        self._cache: dict[str, float] = {}

    def baseline(self, task: BenchmarkTask) -> float:
        if task.task_id not in self._cache:
            scores = []
            for _ in range(self.n):
                result = self.no_skill_solve_fn(task.description, [])
                scores.append(1.0 if self.adapter.verify(task, result) else 0.0)
            self._cache[task.task_id] = statistics.fmean(scores)
        return self._cache[task.task_id]


def _skill_query_text(skill: Skill) -> str:
    return " ".join([skill.title, skill.when_to_apply, *skill.rules])


def _tokenize(text: str) -> set[str]:
    return {t.strip(".,:;()[]{}").lower() for t in text.split() if len(t) > 2}


def reverse_retrieve(skill: Skill, task_pool: list[BenchmarkTask], top_k: int = 5) -> list[BenchmarkTask]:
    """`x_u ~ TopK(s_u, D_task)`: rank the task pool by lexical overlap with
    the *skill's* content (title/when_to_apply/rules) rather than the usual
    task-queries-skills direction -- this is what makes it "reverse"
    retrieval. Stands in for the paper's dense-embedding retrieval (it uses
    `sentence-transformers/all-MiniLM-L6-v2`, Appendix C); swap in a real
    embedding scorer for production use.
    """
    skill_tokens = _tokenize(_skill_query_text(skill))

    def score(task: BenchmarkTask) -> float:
        task_tokens = _tokenize(task.description)
        if not skill_tokens or not task_tokens:
            return 0.0
        return len(skill_tokens & task_tokens) / len(skill_tokens | task_tokens)

    ranked = sorted(task_pool, key=score, reverse=True)
    return ranked[:top_k]


class ExecutionReward:
    """Algorithm 1, lines 9-12: `R_E(u; x_u, pi) = V(tau_u_pi) - b_pi(x_u)`,
    where `x_u` is chosen by reverse retrieval over a task pool, not a fixed
    per-run eval set averaged into a pass rate (an earlier version of this
    file did the latter, which is not what the paper describes).
    """

    def __init__(
        self,
        agent_solve_fn: SolveFn,
        adapter: BenchmarkAdapter,
        baseline_cache: NoSkillBaselineCache,
        top_k: int = 5,
        rng: Optional[random.Random] = None,
    ):
        self.agent_solve_fn = agent_solve_fn
        self.adapter = adapter
        self.baseline_cache = baseline_cache
        self.top_k = top_k
        self.rng = rng or random.Random()

    def score(self, skill: Skill, task_pool: list[BenchmarkTask]) -> float:
        candidates = reverse_retrieve(skill, task_pool, top_k=self.top_k)
        if not candidates:
            return 0.0
        task = self.rng.choice(candidates)
        baseline = self.baseline_cache.baseline(task)
        result = self.agent_solve_fn(task.description, [skill])
        verified = 1.0 if self.adapter.verify(task, result) else 0.0
        return verified - baseline


@dataclass
class RewardBreakdown:
    quality_reward: float
    execution_reward: Optional[float]
    alignment_reward: Optional[float]
    total: float
    quality_detail: Optional[JudgeScore] = None
    alignment_detail: Optional[JudgeScore] = None


class HybridReward:
    """Section 3.3.2 / Algorithm 1's exact reward combination:

        R(u; q) = lam * R_Q(u; q) + R_A(u; tau_u_pi) * R_E(u; x_u, pi)   if u produces an injectable skill
        R(u; q) = lam_dec * R_Q(u; q)                                    otherwise (add / drop / skip)

    `lam = 0.25` is the paper's own reported value ("Quality reward weight
    lambda", Table 4). Note this is *not* three independently-weighted
    terms summed together -- alignment multiplies execution reward, acting
    as a credit-assignment gate (a skill only gets execution credit when the
    agent's behavior actually reflects it), rather than being an additive
    bonus on top. `lam_dec` (the weight for skip/add/drop operations, which
    never reach a skill-conditioned rollout) is referenced in Algorithm 1
    but no distinct numeric value is given anywhere the paper makes
    accessible; it defaults to the same value as `lam` here as a documented
    assumption, not a reported hyperparameter.
    """

    def __init__(self, lam: float = 0.25, lam_dec: Optional[float] = None):
        self.lam = lam
        self.lam_dec = lam_dec if lam_dec is not None else lam

    def combine(
        self,
        quality: JudgeScore,
        *,
        execution: Optional[float] = None,
        alignment: Optional[JudgeScore] = None,
    ) -> RewardBreakdown:
        if execution is not None and alignment is not None:
            total = self.lam * quality.overall() + alignment.overall() * execution
        else:
            total = self.lam_dec * quality.overall()
        return RewardBreakdown(
            quality_reward=quality.overall(),
            execution_reward=execution,
            alignment_reward=alignment.overall() if alignment else None,
            total=total,
            quality_detail=quality,
            alignment_detail=alignment,
        )
