import json

from codeskill.bank import SkillBank
from codeskill.downstream_agent import FrozenDownstreamAgent, SolveResult
from codeskill.llm import MockLLMClient
from codeskill.rewards import (
    BehaviorAlignmentJudge,
    EventQualityJudge,
    EvolutionQualityJudge,
    ExecutionReward,
    HybridReward,
    JudgeScore,
    MergeQualityJudge,
    TaskQualityJudge,
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


def test_execution_reward_uses_downstream_agent_pass_rate():
    bank = SkillBank()

    def always_pass(task, skills):
        return SolveResult(success=True, trace="ok")

    agent = FrozenDownstreamAgent(bank, always_pass, top_k=3)
    reward = ExecutionReward(agent, eval_tasks=["task a", "task b"])
    assert reward.score() == 1.0


def test_execution_reward_empty_tasks_is_zero():
    bank = SkillBank()
    agent = FrozenDownstreamAgent(bank, lambda t, s: SolveResult(success=True), top_k=3)
    reward = ExecutionReward(agent, eval_tasks=[])
    assert reward.score() == 0.0


def test_hybrid_reward_combines_quality_and_execution():
    hybrid = HybridReward(quality_weight=0.5, execution_weight=0.5, alignment_weight=0.0)
    quality = JudgeScore(dimensions={"a": 3}, dimension_maxes={"a": 3})  # overall 1.0
    breakdown = hybrid.combine(quality, execution=0.5)

    assert breakdown.quality_reward == 1.0
    assert breakdown.execution_reward == 0.5
    assert breakdown.total == 0.75


def test_hybrid_reward_includes_alignment_when_provided():
    hybrid = HybridReward(quality_weight=0.5, execution_weight=0.3, alignment_weight=0.2)
    quality = JudgeScore(dimensions={"a": 3}, dimension_maxes={"a": 3})
    alignment = JudgeScore(dimensions={"a": 0}, dimension_maxes={"a": 3})  # overall 0.0
    breakdown = hybrid.combine(quality, execution=1.0, alignment=alignment)

    assert breakdown.alignment_reward == 0.0
    assert breakdown.total == 0.5 * 1.0 + 0.3 * 1.0 + 0.2 * 0.0
