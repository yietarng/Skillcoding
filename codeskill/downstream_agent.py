"""The frozen downstream coding agent.

Central to the paper's training signal: the coding agent itself is never
updated. Only the skill-manager policy is trained, using this agent's
pass/fail outcome (plus rubric judgments) as reward. That separation is
what lets a fixed, possibly closed-source coding agent still benefit from a
learned skill-management policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from codeskill.bank import SkillBank
from codeskill.llm import LLMClient
from codeskill.schema import Skill


@dataclass
class SolveResult:
    success: bool
    trace: str = ""
    used_skill_ids: list[str] = field(default_factory=list)


SolveFn = Callable[[str, list[Skill]], SolveResult]


class FrozenDownstreamAgent:
    """Retrieves relevant skills for a task, then delegates solving to a
    pluggable `solve_fn` -- the agent's own weights/prompting are never
    touched by CODESKILL training."""

    def __init__(self, bank: SkillBank, solve_fn: SolveFn, top_k: int = 5):
        self.bank = bank
        self.solve_fn = solve_fn
        self.top_k = top_k

    def attempt(
        self,
        task_description: str,
        *,
        record_usage: bool = True,
        exclude_trajectory_ids: Optional[set[str]] = None,
    ) -> SolveResult:
        """`exclude_trajectory_ids` implements the paper's same-instance
        leakage guard (Appendix C): pass the current evaluation instance's
        own trajectory id(s) so a skill extracted from this very instance
        can't be retrieved to help solve it."""
        retrieved = self.bank.retrieve(task_description, top_k=self.top_k, exclude_trajectory_ids=exclude_trajectory_ids)
        skills = [r.skill for r in retrieved]
        result = self.solve_fn(task_description, skills)
        result.used_skill_ids = [s.id for s in skills]
        if record_usage:
            for skill in skills:
                skill.record_usage(result.success)
        return result


def make_llm_solver(llm: LLMClient, verifier: Optional[Callable[[str, str], bool]] = None) -> SolveFn:
    """Build a `solve_fn` that asks an LLM to solve the task given the
    retrieved skills as context, then checks the answer with `verifier`
    (e.g. running the task's tests). If no verifier is given, the agent's
    own self-reported completion is used -- useful for smoke-testing, not
    for training reward (which should use a real verifiable check)."""

    def solve(task_description: str, skills: list[Skill]) -> SolveResult:
        skill_block = "\n\n".join(
            f"Skill: {s.title}\nWhen to apply: {s.when_to_apply}\nRules:\n"
            + "\n".join(f"  - {rule}" for rule in s.rules)
            for s in skills
        )
        system = (
            "You are a coding agent. Use the following retrieved skills if relevant, "
            "then solve the task. End your answer with SOLUTION: <your solution>."
        )
        prompt = f"Relevant skills:\n{skill_block}\n\nTask:\n{task_description}"
        response = llm.complete(system, prompt)
        success = verifier(task_description, response) if verifier else "SOLUTION:" in response
        return SolveResult(success=success, trace=response)

    return solve
