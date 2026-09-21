"""Generic benchmark harness.

`BenchmarkAdapter` is the seam between CODESKILL and any concrete benchmark.
The paper evaluates on EnvBench, SWE-Bench Verified, and Terminal-Bench 2;
those are large external datasets that need their own sandboxed execution
environments (a checked-out repo + test runner, a live terminal container,
etc.), which this repository does not vendor. Implementing a real adapter
means: (1) loading that benchmark's tasks into `BenchmarkTask`, and (2)
implementing `verify()` to actually run the benchmark's own checker
(pytest subset, dependency-resolution check, terminal session grader, ...)
against the downstream agent's output. See `codeskill/eval/mock_benchmark.py`
for a fully working (synthetic) example of the shape a real adapter takes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from codeskill.downstream_agent import FrozenDownstreamAgent, SolveResult


@dataclass
class BenchmarkTask:
    task_id: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    task: BenchmarkTask
    solve_result: SolveResult
    passed: bool


@dataclass
class BenchmarkResult:
    name: str
    task_results: list[TaskResult] = field(default_factory=list)

    def pass_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return sum(1.0 for r in self.task_results if r.passed) / len(self.task_results)


class BenchmarkAdapter(ABC):
    """Implement this once per benchmark to plug it into `run_benchmark`."""

    name: str = "unnamed_benchmark"

    @abstractmethod
    def load_tasks(self) -> list[BenchmarkTask]:
        """Return the tasks to evaluate on."""

    @abstractmethod
    def verify(self, task: BenchmarkTask, solve_result: SolveResult) -> bool:
        """Return True iff the agent's attempt actually solves `task`.

        Should be a *real* verifiable check (run tests, check environment
        state, grade the terminal session) -- never the agent's own
        self-reported success -- since this is exactly the sparse execution
        signal the paper's hybrid reward and the final reported pass rates
        depend on.

        The paper defines its verifier as `V(tau) in [0,1]`, i.e.
        potentially a continuous/partial-credit score (e.g. EnvBench's
        per-instance issue count suggests fractional resolution is
        meaningful there), not necessarily a hard pass/fail. This interface
        simplifies to a bool for the benchmarks that are genuinely binary
        (SWE-Bench-Verified's FAIL_TO_PASS/PASS_TO_PASS gate, Terminal-
        Bench-2's task grader); `codeskill.rewards.NoSkillBaselineCache` and
        `ExecutionReward` already average several such bools into a
        continuous baseline the way Algorithm 1 does. A benchmark that
        genuinely needs fractional per-rollout credit should widen this
        return type rather than force it through a bool.
        """


def run_benchmark(adapter: BenchmarkAdapter, agent: FrozenDownstreamAgent) -> BenchmarkResult:
    result = BenchmarkResult(name=adapter.name)
    for task in adapter.load_tasks():
        solve_result = agent.attempt(task.description)
        passed = adapter.verify(task, solve_result)
        result.task_results.append(TaskResult(task=task, solve_result=solve_result, passed=passed))
    return result


class UnimplementedBenchmarkAdapter(BenchmarkAdapter):
    """Placeholder for a real benchmark that needs external assets/sandboxes
    this repo doesn't vendor (dataset files, container images, test
    harnesses). Raises clearly instead of silently returning empty/fake
    results, so a caller can't mistake "not wired up" for "0% pass rate"."""

    def __init__(self, name: str, setup_hint: str):
        self.name = name
        self.setup_hint = setup_hint

    def load_tasks(self) -> list[BenchmarkTask]:
        raise NotImplementedError(f"{self.name} is not wired up yet: {self.setup_hint}")

    def verify(self, task: BenchmarkTask, solve_result: SolveResult) -> bool:
        raise NotImplementedError(f"{self.name} is not wired up yet: {self.setup_hint}")


def env_bench_adapter() -> BenchmarkAdapter:
    return UnimplementedBenchmarkAdapter(
        "EnvBench",
        "load its environment-setup/dependency-repair tasks and verify by actually "
        "building the target environment and checking the resolved dependency state.",
    )


def swe_bench_verified_adapter() -> BenchmarkAdapter:
    return UnimplementedBenchmarkAdapter(
        "SWE-Bench-Verified",
        "load its repo-level issue tasks (instance patches + FAIL_TO_PASS/PASS_TO_PASS "
        "test lists) and verify by applying the agent's patch in the task's container "
        "image and running those tests.",
    )


def terminal_bench_2_adapter() -> BenchmarkAdapter:
    return UnimplementedBenchmarkAdapter(
        "Terminal-Bench-2",
        "load its terminal task specs and verify with its own session grader running "
        "inside the benchmark's terminal container.",
    )
