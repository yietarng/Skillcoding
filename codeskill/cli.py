"""Command-line entry points for the CODESKILL reimplementation.

    python -m codeskill.cli demo
    python -m codeskill.cli extract --trajectories examples/sample_trajectories.json --out-bank bank.json
    python -m codeskill.cli build-sft --trajectories examples/sample_trajectories.json --out sft.jsonl
    python -m codeskill.cli evaluate --bank bank.json
"""
from __future__ import annotations

import argparse
import sys

from codeskill.bank import SkillBank
from codeskill.demo_mock import build_demo_llm
from codeskill.downstream_agent import SolveResult
from codeskill.eval.mock_benchmark import MockBenchmarkAdapter
from codeskill.extraction import EventSkillExtractor
from codeskill.llm import AnthropicLLMClient, LLMClient
from codeskill.manager import SkillManagerPolicy
from codeskill.pipeline import run_pipeline
from codeskill.schema import Skill, load_trajectories
from codeskill.sft_data import build_event_extraction_dataset, export_jsonl


def _make_llm(use_anthropic: bool) -> LLMClient:
    if use_anthropic:
        return AnthropicLLMClient()
    return build_demo_llm()


def _demo_solve_fn_factory(bank: SkillBank):
    """Illustrative, non-LLM solver for the demo/mock benchmark: it "solves"
    a task by writing out whatever retrieved skills say to do. This isolates
    the one thing the demo is meant to show -- that a populated skill bank
    changes the agent's behavior/outcome relative to an empty one -- from
    any LLM-sampling randomness.
    """

    def solve(task_description: str, skills: list[Skill]) -> SolveResult:
        if not skills:
            trace = f"Attempting task with no prior skills: {task_description}\nSOLUTION: generic attempt, no specific fix applied."
            return SolveResult(success=False, trace=trace)
        rule_lines = "\n".join(f"- {rule}" for s in skills for rule in s.rules)
        trace = f"Attempting task: {task_description}\nApplying retrieved skill rules:\n{rule_lines}\nSOLUTION: applied retrieved skill rules."
        return SolveResult(success=True, trace=trace)

    return solve


def cmd_demo(args: argparse.Namespace) -> None:
    trajectories = load_trajectories(args.trajectories)
    llm = build_demo_llm()
    benchmark = MockBenchmarkAdapter()

    report = run_pipeline(
        trajectories=trajectories,
        llm=llm,
        benchmark_adapter=benchmark,
        solve_fn_factory=_demo_solve_fn_factory,
        top_k=args.top_k,
    )

    print(f"Processed {len(trajectories)} trajectories.")
    applied = sum(1 for o in report.round_report.stage_outcomes if o.applied)
    print(f"Stage outcomes applied: {applied} / {len(report.round_report.stage_outcomes)}")
    for outcome in report.round_report.stage_outcomes:
        status = "OK" if outcome.applied else "SKIPPED"
        print(f"  [{status}] {outcome.stage}: {outcome.detail}")
    print()
    print("Skill bank:")
    for skill in report.bank.active_skills():
        print(f"  - ({skill.granularity.value}) {skill.title}: {skill.when_to_apply}")
    print()
    print(report.summary())


def cmd_extract(args: argparse.Namespace) -> None:
    trajectories = load_trajectories(args.trajectories)
    bank = SkillBank.load(args.bank) if args.bank else SkillBank()
    llm = _make_llm(args.anthropic)
    manager = SkillManagerPolicy.from_llm(llm, bank)
    report = manager.run_round(trajectories)

    for outcome in report.stage_outcomes:
        status = "OK" if outcome.applied else "SKIPPED"
        print(f"[{status}] {outcome.stage}: {outcome.detail}")
    print(
        f"Bank size: {report.bank_size_before} -> {report.bank_size_after}"
        f" (compacted {len(report.compacted_skill_ids)})"
    )

    bank.save(args.out_bank)
    print(f"Saved bank to {args.out_bank}")


def cmd_build_sft(args: argparse.Namespace) -> None:
    trajectories = load_trajectories(args.trajectories)
    teacher_llm = _make_llm(args.anthropic)
    teacher_extractor = EventSkillExtractor(teacher_llm)
    examples = build_event_extraction_dataset(trajectories, teacher_extractor)
    export_jsonl(examples, args.out)
    print(f"Wrote {len(examples)} event-extraction SFT examples to {args.out}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    bank = SkillBank.load(args.bank)
    benchmark = MockBenchmarkAdapter()
    from codeskill.downstream_agent import FrozenDownstreamAgent
    from codeskill.eval.harness import run_benchmark

    agent = FrozenDownstreamAgent(bank, _demo_solve_fn_factory(bank), top_k=args.top_k)
    result = run_benchmark(benchmark, agent)
    print(f"{result.name}: pass rate {result.pass_rate():.1%} ({len(result.task_results)} tasks)")
    for task_result in result.task_results:
        status = "PASS" if task_result.passed else "FAIL"
        print(f"  [{status}] {task_result.task.task_id}: {task_result.task.description}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codeskill", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser("demo", help="Run the full pipeline end-to-end with a deterministic mock LLM.")
    demo_p.add_argument("--trajectories", default="examples/sample_trajectories.json")
    demo_p.add_argument("--top-k", type=int, default=3)
    demo_p.set_defaults(func=cmd_demo)

    extract_p = sub.add_parser("extract", help="Extract/maintain skills from trajectories into a skill bank.")
    extract_p.add_argument("--trajectories", required=True)
    extract_p.add_argument("--bank", default=None, help="Existing bank JSON to load and evolve.")
    extract_p.add_argument("--out-bank", required=True)
    extract_p.add_argument("--anthropic", action="store_true", help="Use a real Anthropic model instead of the mock LLM.")
    extract_p.set_defaults(func=cmd_extract)

    sft_p = sub.add_parser("build-sft", help="Build a warm-start SFT dataset (event-extraction stage) from teacher-labeled trajectories.")
    sft_p.add_argument("--trajectories", required=True)
    sft_p.add_argument("--out", required=True)
    sft_p.add_argument("--anthropic", action="store_true", help="Use a real Anthropic model as the teacher.")
    sft_p.set_defaults(func=cmd_build_sft)

    eval_p = sub.add_parser("evaluate", help="Evaluate a saved skill bank on the mock benchmark.")
    eval_p.add_argument("--bank", required=True)
    eval_p.add_argument("--top-k", type=int, default=3)
    eval_p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
