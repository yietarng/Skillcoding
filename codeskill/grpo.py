"""GRPO training skeleton for the skill-manager policy.

Group Relative Policy Optimization needs no learned critic: for each
trajectory, sample a *group* of candidate skill-operation completions from
the current policy, score each with the hybrid reward, and turn the
within-group reward distribution into an advantage (score minus the group's
own mean, scaled by the group's own std). This module implements that
rollout + advantage bookkeeping precisely, and leaves the actual gradient
step behind the `PolicyModel.update` interface so it can be backed by
whatever training stack is available (HF `transformers` + a custom loss,
TRL, or a remote fine-tuning API) without this module needing to depend on
any of them.

Running a real gradient step requires a local/accessible base model and a
training loop far too heavy to embed here; what's implemented is everything
paper-shaped *around* that step -- sampling, reward scoring shared with the
inference-time validation path, and advantage computation -- so plugging in
a concrete `PolicyModel` is the only thing left to do to actually train.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Protocol

from codeskill.extraction import SYSTEM_PROMPT, build_extraction_prompt, parse_operations
from codeskill.rewards import HybridReward
from codeskill.schema import SkillOperation, Trajectory


class PolicyModel(Protocol):
    """The model being trained. Kept separate from `codeskill.llm.LLMClient`
    because training needs `n` correlated samples at a given temperature
    (for group-relative advantages) and a hook to actually apply gradients,
    neither of which a plain inference client exposes."""

    def sample(self, system: str, prompt: str, n: int, temperature: float) -> list[str]:
        ...

    def update(self, batch: list["GRPOSample"]) -> dict:
        """Apply one optimizer step given the scored batch and return a
        dict of training stats (e.g. loss, KL) for logging."""
        ...


@dataclass
class GRPOSample:
    system: str
    prompt: str
    completion: str
    operations: list[SkillOperation]
    reward: float
    advantage: float = 0.0


@dataclass
class GRPOGroupResult:
    trajectory: Trajectory
    samples: list[GRPOSample] = field(default_factory=list)

    def mean_reward(self) -> float:
        return statistics.fmean(s.reward for s in self.samples) if self.samples else 0.0

    def std_reward(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return statistics.pstdev(s.reward for s in self.samples)


def compute_group_advantages(samples: list[GRPOSample], eps: float = 1e-4) -> None:
    """In-place group-relative advantage: (reward - group_mean) / (group_std + eps).

    This is the core of GRPO -- no value/critic network is needed because
    the group itself supplies the baseline.
    """
    if not samples:
        return
    mean = statistics.fmean(s.reward for s in samples)
    std = statistics.pstdev(s.reward for s in samples) if len(samples) > 1 else 0.0
    for s in samples:
        s.advantage = (s.reward - mean) / (std + eps)


class GRPOTrainer:
    def __init__(
        self,
        policy: PolicyModel,
        reward_fn: HybridReward,
        group_size: int = 4,
        temperature: float = 1.0,
    ):
        self.policy = policy
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.temperature = temperature

    def rollout_group(self, trajectory: Trajectory, bank_summary: str = "") -> GRPOGroupResult:
        prompt = build_extraction_prompt(trajectory, bank_summary)
        completions = self.policy.sample(SYSTEM_PROMPT, prompt, n=self.group_size, temperature=self.temperature)

        result = GRPOGroupResult(trajectory=trajectory)
        for completion in completions:
            operations = parse_operations(completion, trajectory)
            reward = self._score_operations(operations, trajectory)
            result.samples.append(
                GRPOSample(
                    system=SYSTEM_PROMPT,
                    prompt=prompt,
                    completion=completion,
                    operations=operations,
                    reward=reward,
                )
            )
        compute_group_advantages(result.samples)
        return result

    def _score_operations(self, operations: list[SkillOperation], trajectory: Trajectory) -> float:
        if not operations:
            return 0.0
        # A completion can propose several operations; the sample's reward
        # is their mean so a policy can't game the reward by padding one
        # good operation with many low-effort ones.
        scores = [self.reward_fn.score_operation(op, trajectory).total for op in operations]
        return statistics.fmean(scores)

    def train_step(self, trajectories: list[Trajectory], bank_summary: str = "") -> dict:
        batch: list[GRPOSample] = []
        group_stats = []
        for trajectory in trajectories:
            group = self.rollout_group(trajectory, bank_summary)
            batch.extend(group.samples)
            group_stats.append({"mean_reward": group.mean_reward(), "std_reward": group.std_reward()})

        update_stats = self.policy.update(batch)
        return {
            "num_groups": len(trajectories),
            "num_samples": len(batch),
            "mean_reward": statistics.fmean(s.reward for s in batch) if batch else 0.0,
            "groups": group_stats,
            **update_stats,
        }


class LoggingOnlyPolicy:
    """No-op `PolicyModel` that records what it would train on instead of
    updating any weights. Useful for wiring/testing `GRPOTrainer` end to end
    (and as a template for a real backend) without a training stack."""

    def __init__(self, sampler):
        """`sampler(system, prompt, n, temperature) -> list[str]` -- typically
        an `LLMClient.complete` called `n` times, or a `MockLLMClient` fed
        several registered variants for the same prompt."""
        self._sampler = sampler
        self.update_history: list[list[GRPOSample]] = []

    def sample(self, system: str, prompt: str, n: int, temperature: float) -> list[str]:
        return self._sampler(system, prompt, n, temperature)

    def update(self, batch: list[GRPOSample]) -> dict:
        self.update_history.append(batch)
        return {"applied_update": False, "batch_size": len(batch)}
