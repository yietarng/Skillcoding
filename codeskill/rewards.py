"""Hybrid reward: dense rubric-based skill quality + sparse execution feedback.

This is the training signal the paper describes for optimizing the
skill-manager policy with GRPO:

  - `RubricReward`  -- dense, always available. An LLM judge scores a
    proposed skill against a fixed rubric (grounding, clarity,
    non-redundancy, and alignment between the skill's steps and the
    behavior it was distilled from).
  - `ExecutionReward` -- sparse, verifiable. Measures whether the skill
    actually helps the *frozen* downstream agent solve held-out tasks --
    the ground truth signal, but expensive and only meaningful in
    aggregate/at rollout granularity.
  - `HybridReward` linearly combines the two so the policy gets a gradient
    signal even on operations too cheap/early to execution-test, while
    still being anchored to real task outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from codeskill.downstream_agent import FrozenDownstreamAgent
from codeskill.llm import LLMClient, extract_json
from codeskill.schema import Skill, SkillOperation, Trajectory

RUBRIC_SYSTEM_PROMPT = """You are grading a candidate coding-agent skill on four criteria, each 0.0-1.0:
- grounding: are the steps actually supported by the cited trajectory evidence (not invented)?
- clarity: is the skill specific and actionable, not vague boilerplate?
- non_redundancy: does it add information beyond generic common sense?
- alignment: do the skill's steps match the behavior/outcome of the trajectory it came from?
Respond with a single JSON object: {"grounding": x, "clarity": x, "non_redundancy": x, "alignment": x, "explanation": "..."}"""


@dataclass
class RubricScore:
    grounding: float
    clarity: float
    non_redundancy: float
    alignment: float
    explanation: str = ""

    def overall(self) -> float:
        return (self.grounding + self.clarity + self.non_redundancy + self.alignment) / 4.0


@dataclass
class RewardBreakdown:
    rubric_reward: float
    execution_reward: Optional[float]
    total: float
    rubric_detail: Optional[RubricScore] = None


class RubricReward:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def score(self, skill: Skill, trajectory: Optional[Trajectory] = None) -> RubricScore:
        evidence = ""
        if trajectory is not None:
            evidence = "\n".join(
                f"[{s.index}] {s.assistant_action} -> {s.tool_result}"
                for s in trajectory.steps
                if s.is_grounded()
            )
        prompt = (
            f"Skill name: {skill.name}\nDescription: {skill.description}\n"
            f"Steps:\n" + "\n".join(f"- {step}" for step in skill.steps) + "\n\n"
            f"Cited trajectory evidence:\n{evidence}"
        )
        raw = self.llm.complete(RUBRIC_SYSTEM_PROMPT, prompt)
        payload = extract_json(raw)
        return RubricScore(
            grounding=float(payload.get("grounding", 0.0)),
            clarity=float(payload.get("clarity", 0.0)),
            non_redundancy=float(payload.get("non_redundancy", 0.0)),
            alignment=float(payload.get("alignment", 0.0)),
            explanation=payload.get("explanation", ""),
        )


class ExecutionReward:
    """Runs the frozen downstream agent on a fixed set of held-out tasks and
    reports the pass rate. Meant to be measured before/after a batch of
    skill operations to see whether they actually helped."""

    def __init__(self, agent: FrozenDownstreamAgent, eval_tasks: list[str]):
        self.agent = agent
        self.eval_tasks = eval_tasks

    def score(self) -> float:
        if not self.eval_tasks:
            return 0.0
        results = [self.agent.attempt(t, record_usage=False) for t in self.eval_tasks]
        return sum(1.0 for r in results if r.success) / len(results)


class HybridReward:
    def __init__(
        self,
        rubric: RubricReward,
        execution: Optional[ExecutionReward] = None,
        rubric_weight: float = 0.5,
        execution_weight: float = 0.5,
    ):
        self.rubric = rubric
        self.execution = execution
        self.rubric_weight = rubric_weight
        self.execution_weight = execution_weight

    def score_operation(self, operation: SkillOperation, trajectory: Trajectory) -> RewardBreakdown:
        if operation.skill is None:
            # NOOP / drop / merge without a concrete skill payload: rubric
            # reward doesn't apply, fall back to execution signal alone.
            exec_score = self.execution.score() if self.execution is not None else 0.0
            return RewardBreakdown(rubric_reward=0.0, execution_reward=exec_score, total=exec_score)

        rubric_score = self.rubric.score(operation.skill, trajectory)
        exec_score = self.execution.score() if self.execution is not None else None
        total = self.rubric_weight * rubric_score.overall()
        if exec_score is not None:
            total += self.execution_weight * exec_score
        return RewardBreakdown(
            rubric_reward=rubric_score.overall(),
            execution_reward=exec_score,
            total=total,
            rubric_detail=rubric_score,
        )
