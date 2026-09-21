"""GRPO training skeleton for the skill-manager policy.

Group Relative Policy Optimization needs no learned critic: for each
context (a trajectory, a trajectory group, an evolution case, or a
maintenance case -- whichever the stage being trained needs), sample a
*group* of candidate completions from the current policy, score each with
the hybrid reward, and turn the within-group reward distribution into an
advantage (score minus the group's own mean, scaled by the group's own
std). This module implements that rollout + advantage bookkeeping exactly
once, generically over all four appendix-prompt stages (see
`codeskill.extraction` and `codeskill.prompts`) via injected
`build_prompt`/`parse`/`score` callables, rather than duplicating it per
stage.

Running a real gradient step requires a local/accessible base model and a
training loop far too heavy to embed here; what's implemented is everything
paper-shaped *around* that step -- sampling, reward scoring reusing the
exact parsers used at inference time, and advantage computation -- so
plugging in a concrete `PolicyModel` is the only thing left to do to
actually train.

`GRPOTrainer`'s `group_size=6` and `temperature=0.7` defaults match the
paper's own reported RL settings (Table 4: "Group size 6 generations per
prompt", "Rollout temperature 0.7"); the reward each sample should be
scored with is `codeskill.rewards.HybridReward`, whose `combine()`
implements Algorithm 1's `R = lam*R_Q + R_A*R_E` (skill-producing ops) /
`R = lam_dec*R_Q` (add/drop/skip) exactly, with `lam=0.25` per Table 4.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


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
    parsed: Any
    reward: float
    advantage: float = 0.0


@dataclass
class GRPOGroupResult:
    context: Any
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
    """Stage-agnostic: pass the system prompt and stage-specific
    `build_prompt`/`parse`/`score` for whichever of the four appendix
    prompts (task extraction, event extraction, evolution, maintenance)
    you're training. `parse` should reuse the exact parser
    `codeskill.extraction` uses at inference time (e.g.
    `parse_extraction_response`), so training-time and inference-time
    validation can never drift apart.
    """

    def __init__(
        self,
        policy: PolicyModel,
        system_prompt: str,
        build_prompt: Callable[[Any], str],
        parse: Callable[[str], Any],
        score: Callable[[Any, Any], float],
        group_size: int = 6,
        temperature: float = 0.7,
    ):
        self.policy = policy
        self.system_prompt = system_prompt
        self.build_prompt = build_prompt
        self.parse = parse
        self.score = score
        self.group_size = group_size
        self.temperature = temperature

    def rollout_group(self, context: Any) -> GRPOGroupResult:
        prompt = self.build_prompt(context)
        completions = self.policy.sample(self.system_prompt, prompt, n=self.group_size, temperature=self.temperature)

        result = GRPOGroupResult(context=context)
        for completion in completions:
            parsed = self.parse(completion)
            reward = self.score(parsed, context)
            result.samples.append(
                GRPOSample(system=self.system_prompt, prompt=prompt, completion=completion, parsed=parsed, reward=reward)
            )
        compute_group_advantages(result.samples)
        return result

    def train_step(self, contexts: list[Any]) -> dict:
        batch: list[GRPOSample] = []
        group_stats = []
        for context in contexts:
            group = self.rollout_group(context)
            batch.extend(group.samples)
            group_stats.append({"mean_reward": group.mean_reward(), "std_reward": group.std_reward()})

        update_stats = self.policy.update(batch)
        return {
            "num_groups": len(contexts),
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
