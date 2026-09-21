import json

from codeskill.grpo import GRPOSample, GRPOTrainer, LoggingOnlyPolicy, compute_group_advantages
from codeskill.llm import MockLLMClient
from codeskill.rewards import HybridReward, RubricReward
from codeskill.schema import Trajectory, TrajectoryStep


def make_trajectory() -> Trajectory:
    return Trajectory(
        task_id="t1",
        task_description="fix ModuleNotFoundError for requests",
        steps=[
            TrajectoryStep(0, "run tests", tool_name="shell", tool_result="ModuleNotFoundError", observed=True),
            TrajectoryStep(1, "pip install requests", tool_name="shell", tool_result="installed", observed=True),
        ],
        success=True,
    )


def _op_json(quality: str) -> str:
    steps = ["pip install requests"] if quality == "good" else []
    granularity = "event"
    cited = [1] if quality == "good" else [1]
    return json.dumps(
        {
            "operations": [
                {
                    "op_type": "add",
                    "name": f"{quality} skill",
                    "description": f"a {quality} skill",
                    "steps": steps or ["vague step"],
                    "granularity": granularity,
                    "cited_step_indices": cited,
                    "rationale": quality,
                }
            ]
        }
    )


def test_compute_group_advantages_zero_mean():
    samples = [
        GRPOSample(system="s", prompt="p", completion="a", operations=[], reward=1.0),
        GRPOSample(system="s", prompt="p", completion="b", operations=[], reward=0.0),
    ]
    compute_group_advantages(samples)
    assert samples[0].advantage > 0
    assert samples[1].advantage < 0
    assert abs(samples[0].advantage + samples[1].advantage) < 1e-6


def test_compute_group_advantages_handles_single_sample():
    samples = [GRPOSample(system="s", prompt="p", completion="a", operations=[], reward=0.5)]
    compute_group_advantages(samples)
    assert samples[0].advantage == 0.0


def _rubric_llm(scores_by_quality):
    llm = MockLLMClient()

    def responder(system, prompt):
        quality = "good" if "good skill" in prompt else "bad"
        s = scores_by_quality[quality]
        return json.dumps({"grounding": s, "clarity": s, "non_redundancy": s, "alignment": s, "explanation": "x"})

    llm.register(lambda system, prompt: True, responder)
    return llm


def test_grpo_trainer_rollout_group_scores_and_ranks_samples():
    trajectory = make_trajectory()

    def sampler(system, prompt, n, temperature):
        return [_op_json("good"), _op_json("bad")][:n]

    policy = LoggingOnlyPolicy(sampler)
    reward_fn = HybridReward(RubricReward(_rubric_llm({"good": 1.0, "bad": 0.1})), execution=None, rubric_weight=1.0, execution_weight=0.0)
    trainer = GRPOTrainer(policy, reward_fn, group_size=2, temperature=1.0)

    group = trainer.rollout_group(trajectory)

    assert len(group.samples) == 2
    good_sample = next(s for s in group.samples if "good" in s.completion)
    bad_sample = next(s for s in group.samples if "bad" in s.completion)
    assert good_sample.reward > bad_sample.reward
    assert good_sample.advantage > bad_sample.advantage


def test_grpo_trainer_train_step_invokes_policy_update():
    trajectory = make_trajectory()

    def sampler(system, prompt, n, temperature):
        return [_op_json("good")] * n

    policy = LoggingOnlyPolicy(sampler)
    reward_fn = HybridReward(RubricReward(_rubric_llm({"good": 1.0, "bad": 0.1})), execution=None, rubric_weight=1.0, execution_weight=0.0)
    trainer = GRPOTrainer(policy, reward_fn, group_size=3)

    stats = trainer.train_step([trajectory])

    assert stats["num_groups"] == 1
    assert stats["num_samples"] == 3
    assert len(policy.update_history) == 1
