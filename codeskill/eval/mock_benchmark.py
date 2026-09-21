"""A small, fully-synthetic benchmark used for tests and the `demo` CLI
command. It exercises the exact same `BenchmarkAdapter` interface a real
EnvBench/SWE-Bench-Verified/Terminal-Bench-2 adapter would implement, just
with cheap string-matching verification instead of running real containers,
so the whole pipeline is demonstrable with no external dataset or sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass

from codeskill.downstream_agent import SolveResult
from codeskill.eval.harness import BenchmarkAdapter, BenchmarkTask


@dataclass(frozen=True)
class _MockCase:
    task_id: str
    description: str
    expected_substring: str


DEFAULT_CASES: list[_MockCase] = [
    _MockCase(
        "mock-1",
        "The test suite fails with ModuleNotFoundError: No module named 'requests'. Fix it.",
        "pip install requests",
    ),
    _MockCase(
        "mock-2",
        "A pytest run fails because conftest.py is missing a fixture named 'tmp_repo'. Add it.",
        # Deliberately not a substring of the task description above: passing
        # this check requires actually knowing the fix, not just echoing the
        # problem statement back.
        "temporary git repository",
    ),
    _MockCase(
        "mock-3",
        "setup.py references a package version that conflicts with requirements.txt. Reconcile them.",
        "match the version pinned",
    ),
]


class MockBenchmarkAdapter(BenchmarkAdapter):
    name = "mock_benchmark"

    def __init__(self, cases: list[_MockCase] = DEFAULT_CASES):
        self.cases = cases

    def load_tasks(self) -> list[BenchmarkTask]:
        return [
            BenchmarkTask(task_id=c.task_id, description=c.description, metadata={"expected_substring": c.expected_substring})
            for c in self.cases
        ]

    def verify(self, task: BenchmarkTask, solve_result: SolveResult) -> bool:
        expected = task.metadata["expected_substring"].lower()
        return expected in solve_result.trace.lower()
