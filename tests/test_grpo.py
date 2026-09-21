import json

from codeskill.extraction import build_event_extraction_prompt, parse_extraction_response
from codeskill.grpo import GRPOSample, GRPOTrainer, LoggingOnlyPolicy, compute_group_advantages
from codeskill.prompts import EVENT_EXTRACTION_SYSTEM_PROMPT
from codeskill.rewards import EventQualityJudge, HybridReward
from codeskill.llm import MockLLMClient
from codeskill.schema import Decision, Trajectory, TrajectoryStep


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


def _completion(quality: str) -> str:
    rules = ["pip install requests"] if quality == "good" else ["do something vague"]
    return json.dumps(
        {
            "action": "generate",
            "skill": {
                "title": f"{quality} skill",
                "granularity": "event-driven",
                "when_to_apply": "ModuleNotFoundError for a pip package",
                "rules": rules,
            },
        }
    )


def test_compute_group_advantages_zero_mean():
    samples = [
        GRPOSample(system="s", prompt="p", completion="a", parsed=None, reward=1.0),
        GRPOSample(system="s", prompt="p", completion="b", parsed=None, reward=0.0),
    ]
    compute_group_advantages(samples)
    assert samples[0].advantage > 0
    assert samples[1].advantage < 0
    assert abs(samples[0].advantage + samples[1].advantage) < 1e-6


def test_compute_group_advantages_handles_single_sample():
    samples = [GRPOSample(system="s", prompt="p", completion="a", parsed=None, reward=0.5)]
    compute_group_advantages(samples)
    assert samples[0].advantage == 0.0


def _judge_llm(scores_by_quality: dict) -> MockLLMClient:
    llm = MockLLMClient()

    def responder(system, prompt):
        quality = "good" if "good skill" in prompt else "bad"
        s = scores_by_quality[quality]
        return json.dumps(
            {
                "groundedness": s,
                "reusability": s,
                "specificity": s,
                "format_validity": 1 if s else 0,
                "when_to_apply_quality": s,
                "event_level_granularity": s,
                "reason": "x",
            }
        )

    llm.register(lambda system, prompt: True, responder)
    return llm


def test_grpo_trainer_rollout_group_scores_and_ranks_samples():
    trajectory = make_trajectory()

    def sampler(system, prompt, n, temperature):
        return [_completion("good"), _completion("bad")][:n]

    policy = LoggingOnlyPolicy(sampler)
    judge = EventQualityJudge(_judge_llm({"good": 3, "bad": 0}))
    hybrid = HybridReward(quality_weight=1.0, execution_weight=0.0, alignment_weight=0.0)

    def score(parsed, trajectory):
        if parsed.decision != Decision.GENERATE or parsed.skill is None:
            return 0.0
        quality = judge.score(trajectory, parsed.skill)
        return hybrid.combine(quality).total

    trainer = GRPOTrainer(
        policy=policy,
        system_prompt=EVENT_EXTRACTION_SYSTEM_PROMPT,
        build_prompt=build_event_extraction_prompt,
        parse=lambda raw: parse_extraction_response(raw),
        score=lambda parsed, ctx: score(parsed, ctx),
        group_size=2,
        temperature=1.0,
    )

    group = trainer.rollout_group(trajectory)

    assert len(group.samples) == 2
    good_sample = next(s for s in group.samples if "good" in s.completion)
    bad_sample = next(s for s in group.samples if "bad" in s.completion)
    assert good_sample.reward > bad_sample.reward
    assert good_sample.advantage > bad_sample.advantage


def test_grpo_trainer_train_step_invokes_policy_update():
    trajectory = make_trajectory()

    def sampler(system, prompt, n, temperature):
        return [_completion("good")] * n

    policy = LoggingOnlyPolicy(sampler)
    judge = EventQualityJudge(_judge_llm({"good": 3, "bad": 0}))
    hybrid = HybridReward(quality_weight=1.0, execution_weight=0.0, alignment_weight=0.0)

    def score(parsed, ctx):
        if parsed.decision != Decision.GENERATE or parsed.skill is None:
            return 0.0
        return hybrid.combine(judge.score(ctx, parsed.skill)).total

    trainer = GRPOTrainer(
        policy=policy,
        system_prompt=EVENT_EXTRACTION_SYSTEM_PROMPT,
        build_prompt=build_event_extraction_prompt,
        parse=parse_extraction_response,
        score=score,
        group_size=3,
    )

    stats = trainer.train_step([trajectory])

    assert stats["num_groups"] == 1
    assert stats["num_samples"] == 3
    assert len(policy.update_history) == 1
