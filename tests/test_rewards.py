import json

from codeskill.bank import SkillBank
from codeskill.downstream_agent import FrozenDownstreamAgent, SolveResult
from codeskill.llm import MockLLMClient
from codeskill.rewards import ExecutionReward, HybridReward, RubricReward
from codeskill.schema import Granularity, OperationType, Skill, SkillOperation, Trajectory, TrajectoryStep


def make_skill() -> Skill:
    return Skill(name="install requests", description="pip install requests", steps=["pip install requests"], granularity=Granularity.EVENT)


def make_trajectory() -> Trajectory:
    return Trajectory(
        task_id="t1",
        task_description="fix it",
        steps=[TrajectoryStep(0, "pip install requests", tool_name="shell", tool_result="installed", observed=True)],
        success=True,
    )


def rubric_llm(grounding=0.9, clarity=0.8, non_redundancy=0.7, alignment=0.6) -> MockLLMClient:
    llm = MockLLMClient()
    llm.register(
        lambda system, prompt: True,
        lambda system, prompt: json.dumps(
            {"grounding": grounding, "clarity": clarity, "non_redundancy": non_redundancy, "alignment": alignment, "explanation": "ok"}
        ),
    )
    return llm


def test_rubric_reward_scores_and_averages():
    reward = RubricReward(rubric_llm(1.0, 1.0, 0.0, 1.0))
    score = reward.score(make_skill(), make_trajectory())
    assert score.overall() == 0.75


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


def test_hybrid_reward_combines_rubric_and_execution():
    bank = SkillBank()

    def half_pass(task, skills):
        return SolveResult(success=(task == "task a"), trace="ok")

    agent = FrozenDownstreamAgent(bank, half_pass, top_k=3)
    execution = ExecutionReward(agent, eval_tasks=["task a", "task b"])  # 1 of 2 passes -> 0.5
    rubric = RubricReward(rubric_llm(1.0, 1.0, 1.0, 1.0))  # overall 1.0

    hybrid = HybridReward(rubric, execution, rubric_weight=0.5, execution_weight=0.5)
    op = SkillOperation(op_type=OperationType.ADD, skill=make_skill())

    breakdown = hybrid.score_operation(op, make_trajectory())

    assert breakdown.rubric_reward == 1.0
    assert breakdown.execution_reward == 0.5
    assert breakdown.total == 0.75


def test_hybrid_reward_without_skill_falls_back_to_execution_only():
    bank = SkillBank()
    agent = FrozenDownstreamAgent(bank, lambda t, s: SolveResult(success=True), top_k=3)
    execution = ExecutionReward(agent, eval_tasks=["x"])
    hybrid = HybridReward(RubricReward(rubric_llm()), execution)

    op = SkillOperation(op_type=OperationType.NOOP)
    breakdown = hybrid.score_operation(op, make_trajectory())

    assert breakdown.rubric_reward == 0.0
    assert breakdown.total == 1.0
