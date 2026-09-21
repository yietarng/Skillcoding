"""End-to-end orchestration: trajectories -> skill bank -> benchmark comparison.

Ties the pieces together the way the paper's evaluation does: build/evolve a
skill bank from a batch of trajectories (via `SkillManagerPolicy.run_round`,
which runs event-level extraction (Fig 7) followed by maintenance (Fig 9)
per trajectory), then compare the frozen downstream agent's benchmark pass
rate with an empty bank ("no-skill baseline") against the same agent using
the evolved bank.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from codeskill.bank import SkillBank
from codeskill.downstream_agent import FrozenDownstreamAgent, SolveFn
from codeskill.eval.harness import BenchmarkAdapter, BenchmarkResult, run_benchmark
from codeskill.llm import LLMClient
from codeskill.manager import RoundReport, SkillManagerPolicy
from codeskill.schema import Trajectory

SolveFnFactory = Callable[[SkillBank], SolveFn]


@dataclass
class PipelineReport:
    round_report: RoundReport
    baseline_result: BenchmarkResult
    with_skills_result: BenchmarkResult
    bank: SkillBank

    def summary(self) -> str:
        baseline = self.baseline_result.pass_rate()
        with_skills = self.with_skills_result.pass_rate()
        return (
            f"bank size after round: {len(self.bank)}\n"
            f"no-skill baseline pass rate:  {baseline:.1%}\n"
            f"with-skill-bank pass rate:    {with_skills:.1%}\n"
            f"delta:                        {with_skills - baseline:+.1%}"
        )


def run_pipeline(
    trajectories: list[Trajectory],
    llm: LLMClient,
    benchmark_adapter: BenchmarkAdapter,
    solve_fn_factory: SolveFnFactory,
    bank: Optional[SkillBank] = None,
    top_k: int = 5,
) -> PipelineReport:
    bank = bank if bank is not None else SkillBank()
    manager = SkillManagerPolicy.from_llm(llm, bank, retrieval_top_k=top_k)
    round_report = manager.run_round(trajectories)

    baseline_bank = SkillBank()  # deliberately empty: the no-skill baseline
    baseline_agent = FrozenDownstreamAgent(baseline_bank, solve_fn_factory(baseline_bank), top_k=top_k)
    baseline_result = run_benchmark(benchmark_adapter, baseline_agent)

    with_skills_agent = FrozenDownstreamAgent(bank, solve_fn_factory(bank), top_k=top_k)
    with_skills_result = run_benchmark(benchmark_adapter, with_skills_agent)

    return PipelineReport(
        round_report=round_report,
        baseline_result=baseline_result,
        with_skills_result=with_skills_result,
        bank=bank,
    )
