from pathlib import Path

from codeskill.cli import _demo_solve_fn_factory
from codeskill.demo_mock import build_demo_llm
from codeskill.eval.mock_benchmark import MockBenchmarkAdapter
from codeskill.pipeline import run_pipeline
from codeskill.schema import load_trajectories

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "sample_trajectories.json"


def test_full_pipeline_end_to_end_with_mock_llm():
    trajectories = load_trajectories(EXAMPLES)
    llm = build_demo_llm()
    benchmark = MockBenchmarkAdapter()

    report = run_pipeline(
        trajectories=trajectories,
        llm=llm,
        benchmark_adapter=benchmark,
        solve_fn_factory=_demo_solve_fn_factory,
        top_k=3,
    )

    # Skills were actually extracted and added for all three trajectories
    # (the demo bank starts empty, so maintenance always decides "add").
    assert len(report.bank) == 3
    assert all(o.applied for o in report.round_report.stage_outcomes)

    # A populated bank should out-perform the no-skill baseline on this
    # benchmark, since the mock benchmark's expected fixes are exactly what
    # the extracted skills encode.
    assert report.with_skills_result.pass_rate() > report.baseline_result.pass_rate()
    assert report.with_skills_result.pass_rate() == 1.0
    assert report.baseline_result.pass_rate() == 0.0


def test_pipeline_summary_is_human_readable():
    trajectories = load_trajectories(EXAMPLES)
    llm = build_demo_llm()
    benchmark = MockBenchmarkAdapter()

    report = run_pipeline(
        trajectories=trajectories,
        llm=llm,
        benchmark_adapter=benchmark,
        solve_fn_factory=_demo_solve_fn_factory,
    )

    summary = report.summary()
    assert "pass rate" in summary
    assert "delta" in summary
