import json
import random

from codeskill.downstream_agent import SolveResult
from codeskill.eval.harness import BenchmarkAdapter, BenchmarkTask
from codeskill.llm import MockLLMClient
from codeskill.rewards import (
    BehaviorAlignmentJudge,
    EventQualityJudge,
    EvolutionQualityJudge,
    ExecutionReward,
    HybridReward,
    JudgeScore,
    MergeQualityJudge,
    NoSkillBaselineCache,
    TaskQualityJudge,
    reverse_retrieve,
)
from codeskill.schema import Granularity, Skill, Trajectory, TrajectoryStep


def make_skill(title="install requests") -> Skill:
    return Skill(title=title, granularity=Granularity.EVENT_DRIVEN, when_to_apply="ModuleNotFoundError for requests", rules=["pip install requests"])


def make_trajectory() -> Trajectory:
    return Trajectory(
        task_id="t1",
        task_description="fix it",
        steps=[TrajectoryStep(0, "pip install requests", tool_name="shell", tool_result="installed", observed=True)],
        success=True,
    )


def judge_llm(payload: dict) -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(lambda system, prompt: True, lambda system, prompt: json.dumps(payload))
    return llm


def test_judge_score_overall_normalizes_by_max():
    score = JudgeScore(dimensions={"a": 2, "b": 1}, dimension_maxes={"a": 3, "b": 1})
    assert score.overall() == 0.75


def test_judge_score_clamps_out_of_range_values():
    score = JudgeScore(dimensions={"a": 99}, dimension_maxes={"a": 3})
    assert score.overall() == 1.0


def test_task_quality_judge_scores_from_llm_output():
    payload = {
        "groundedness": 3,
        "reusability": 2,
        "specificity": 2,
        "format_validity": 1,
        "when_to_apply_quality": 3,
        "task_level_granularity": 3,
        "reason": "solid",
    }
    judge = TaskQualityJudge(judge_llm(payload))
    score = judge.score([make_trajectory()], make_skill())
    assert score.dimensions["groundedness"] == 3
    assert 0.0 < score.overall() <= 1.0


def test_event_quality_judge_scores_from_llm_output():
    payload = {
        "groundedness": 3,
        "reusability": 3,
        "specificity": 3,
        "format_validity": 1,
        "when_to_apply_quality": 3,
        "event_level_granularity": 3,
        "reason": "perfect",
    }
    judge = EventQualityJudge(judge_llm(payload))
    score = judge.score(make_trajectory(), make_skill())
    assert score.overall() == 1.0


def test_evolution_quality_judge_scores_from_llm_output():
    payload = {
        "groundedness": 1,
        "reusability": 1,
        "specificity": 1,
        "format_validity": 0,
        "failure_responsiveness": 1,
        "update_quality": 1,
        "reason": "weak update",
    }
    judge = EvolutionQualityJudge(judge_llm(payload))
    score = judge.score(make_skill(), make_trajectory(), make_skill(title="install requests (revised)"))
    assert score.overall() < 0.5


def test_merge_quality_judge_scores_from_llm_output():
    payload = {
        "target_choice_correct": 3,
        "target_overlap_substantive": 3,
        "merge_integration_quality": 3,
        "identity_preserved": 3,
        "no_critical_loss_target": 3,
        "no_critical_loss_candidate": 3,
        "merged_when_to_apply_tightness": 3,
        "format_validity": 1,
        "reason": "clean merge",
    }
    judge = MergeQualityJudge(judge_llm(payload))
    existing = make_skill()
    score = judge.score(make_skill(title="candidate"), [existing], existing.id, make_skill(title="merged"))
    assert score.overall() == 1.0


def test_behavior_alignment_judge_scores_from_llm_output():
    payload = {"when_to_apply_match": 3, "rule_specificity": 0, "trajectory_evidence": 0, "reason": "only the trigger matched"}
    judge = BehaviorAlignmentJudge(judge_llm(payload))
    score = judge.score("task context", "user prompt", make_skill(), make_trajectory(), "result summary")
    assert score.dimensions["when_to_apply_match"] == 3
    assert score.overall() == 3 / 9


class _FixedAdapter(BenchmarkAdapter):
    """A BenchmarkAdapter stand-in whose `verify` outcome per task_id is
    fully scripted, so baseline-averaging and reverse-retrieval scoring are
    each deterministic to test against."""

    name = "fixed"

    def __init__(self, outcomes_by_task_id: dict[str, list[bool]]):
        self.outcomes_by_task_id = outcomes_by_task_id
        self._call_index: dict[str, int] = {}

    def load_tasks(self):
        raise NotImplementedError

    def verify(self, task: BenchmarkTask, solve_result: SolveResult) -> bool:
        i = self._call_index.get(task.task_id, 0)
        outcomes = self.outcomes_by_task_id[task.task_id]
        outcome = outcomes[min(i, len(outcomes) - 1)]
        self._call_index[task.task_id] = i + 1
        return outcome


def test_reverse_retrieve_ranks_tasks_by_overlap_with_skill_content():
    skill = make_skill(title="install requests")  # when_to_apply mentions ModuleNotFoundError, requests
    matching_task = BenchmarkTask("t-match", "ModuleNotFoundError for the requests package")
    unrelated_task = BenchmarkTask("t-other", "add a pytest fixture for a temporary git repository")

    ranked = reverse_retrieve(skill, [unrelated_task, matching_task], top_k=2)

    assert ranked[0].task_id == "t-match"


def test_no_skill_baseline_cache_averages_n_rollouts_and_caches():
    task = BenchmarkTask("t1", "fix it")
    # 3 of 4 no-skill rollouts pass -> baseline 0.75
    adapter = _FixedAdapter({"t1": [True, True, True, False]})
    calls = []

    def no_skill_solve(task_description, skills):
        calls.append(task_description)
        return SolveResult(success=True, trace="ok")

    cache = NoSkillBaselineCache(no_skill_solve, adapter, n=4)

    assert cache.baseline(task) == 0.75
    assert len(calls) == 4
    # second call for the same task must hit the cache, not re-run rollouts
    assert cache.baseline(task) == 0.75
    assert len(calls) == 4


def test_execution_reward_matches_algorithm_1_formula():
    matching_task = BenchmarkTask("t-match", "ModuleNotFoundError for the requests package")
    baseline_adapter = _FixedAdapter({"t-match": [True, False, False, False]})  # baseline = 0.25
    cache = NoSkillBaselineCache(lambda desc, skills: SolveResult(success=True), baseline_adapter, n=4)

    skill_conditioned_adapter = _FixedAdapter({"t-match": [True]})  # skill-conditioned rollout passes -> V=1.0

    def agent_solve(task_description, skills):
        return SolveResult(success=True, trace="applied skill")

    reward = ExecutionReward(agent_solve, skill_conditioned_adapter, cache, top_k=1, rng=random.Random(0))
    skill = make_skill(title="install requests")

    score = reward.score(skill, [matching_task])

    assert score == 1.0 - 0.25  # V(skill-conditioned) - baseline


def test_execution_reward_empty_pool_is_zero():
    adapter = _FixedAdapter({})
    cache = NoSkillBaselineCache(lambda d, s: SolveResult(success=True), adapter, n=4)
    reward = ExecutionReward(lambda d, s: SolveResult(success=True), adapter, cache)
    assert reward.score(make_skill(), []) == 0.0


def test_hybrid_reward_skill_output_multiplies_alignment_by_execution():
    # Algorithm 1: R = lam*RQ + RA*RE for operations that produce a skill.
    hybrid = HybridReward(lam=0.25)
    quality = JudgeScore(dimensions={"a": 2}, dimension_maxes={"a": 4})  # overall 0.5
    alignment = JudgeScore(dimensions={"a": 3}, dimension_maxes={"a": 3})  # overall 1.0

    breakdown = hybrid.combine(quality, execution=0.6, alignment=alignment)

    assert breakdown.total == 0.25 * 0.5 + 1.0 * 0.6


def test_hybrid_reward_zero_alignment_gates_out_execution_credit():
    hybrid = HybridReward(lam=0.25)
    quality = JudgeScore(dimensions={"a": 4}, dimension_maxes={"a": 4})  # overall 1.0
    alignment = JudgeScore(dimensions={"a": 0}, dimension_maxes={"a": 3})  # overall 0.0

    # Even a large execution reward earns nothing when alignment is zero --
    # the agent's success wasn't attributable to this skill.
    breakdown = hybrid.combine(quality, execution=1.0, alignment=alignment)

    assert breakdown.total == 0.25 * 1.0


def test_hybrid_reward_no_skill_output_uses_lam_dec_quality_only():
    # Algorithm 1: R = lam_dec*RQ for add/drop/skip (no execution/alignment available).
    hybrid = HybridReward(lam=0.25, lam_dec=0.4)
    quality = JudgeScore(dimensions={"a": 3}, dimension_maxes={"a": 3})  # overall 1.0

    breakdown = hybrid.combine(quality)

    assert breakdown.execution_reward is None
    assert breakdown.alignment_reward is None
    assert breakdown.total == 0.4


def test_hybrid_reward_lam_dec_defaults_to_lam():
    hybrid = HybridReward(lam=0.25)
    assert hybrid.lam_dec == 0.25
