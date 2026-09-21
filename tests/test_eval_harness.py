import pytest

from codeskill.bank import SkillBank
from codeskill.downstream_agent import FrozenDownstreamAgent, SolveResult
from codeskill.eval.harness import env_bench_adapter, run_benchmark, swe_bench_verified_adapter, terminal_bench_2_adapter
from codeskill.eval.mock_benchmark import MockBenchmarkAdapter


def test_mock_benchmark_pass_when_solver_mentions_expected_substring():
    bank = SkillBank()

    def solve(task, skills):
        return SolveResult(success=True, trace="I ran pip install requests to fix it")

    agent = FrozenDownstreamAgent(bank, solve, top_k=1)
    result = run_benchmark(MockBenchmarkAdapter(), agent)

    mock1 = next(r for r in result.task_results if r.task.task_id == "mock-1")
    assert mock1.passed is True


def test_mock_benchmark_fail_when_solver_output_is_generic():
    bank = SkillBank()

    def solve(task, skills):
        return SolveResult(success=False, trace="I tried something generic")

    agent = FrozenDownstreamAgent(bank, solve, top_k=1)
    result = run_benchmark(MockBenchmarkAdapter(), agent)

    assert result.pass_rate() == 0.0


@pytest.mark.parametrize("factory", [env_bench_adapter, swe_bench_verified_adapter, terminal_bench_2_adapter])
def test_real_benchmark_adapters_are_explicit_placeholders(factory):
    adapter = factory()
    with pytest.raises(NotImplementedError):
        adapter.load_tasks()
