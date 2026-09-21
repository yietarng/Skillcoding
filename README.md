# codeskill

An implementation of the architecture described in **"CODESKILL: Learning
Self-Evolving Skills for Coding Agents"** (Li, Zhang, Zhang, Liu, Liu;
arXiv:2605.25430), using the paper's **actual appendix prompts**.

## Prompt provenance

`codeskill/prompts.py` contains the nine system prompts from the paper's
appendix, Figures 6-14 (pages 16-24): task-level extraction, event-driven
extraction, skill evolution, skill-bank maintenance, and five rubric-judge
prompts (task/event/evolution quality, merge quality, behavior alignment).

**How they got here, honestly:** arxiv.org, huggingface.co,
researchgate.net, and alphaxiv.org were all blocked by network egress
policy in the environment this was written in, so these were not
transcribed directly from the primary PDF by this codebase's author. They
were pulled from the public `prompts/paper/` directory of a third-party
reconstruction project, `rayyichen310/codeskill-rebuild`, whose
`docs/REPRODUCTION_SPEC.md` states they are "faithful, line-normalized
transcriptions" of the appendix and records a SHA-256 checksum of the
source PDF it transcribed from. That's a reasonable chain of custody, but
it is *not* independently verified byte-for-byte against the primary source
here -- treat `codeskill/prompts.py`'s docstring citation as "best-effort
reproduction," not "verified quote."

Where the paper's own code, exact hyperparameters, retrieval embedding
model, or reward weighting were never observed (the appendix gives the
*prompts*, not the training or serving code), this repo makes a reasonable,
clearly-documented engineering choice rather than guessing at unobserved
specifics. Those choices are called out inline, in the module docstrings
below.

## Architecture

The appendix specifies four manager-policy stages, each its own prompt --
not one combined "extract and maintain" call:

```
                    ┌─────────────────────┐
2-3 related         │ TaskSkillExtractor   │  generate | skip
trajectories ──────►│ (Fig 6)              │──┐
                    └─────────────────────┘  │
                                              │
one full trajectory ┌─────────────────────┐  │  candidate Skill
              ──────►│ EventSkillExtractor  │──┤  {title, granularity,
                    │ (Fig 7)              │  │   when_to_apply, rules}
                    └─────────────────────┘  │
                                              ▼
                                     ┌──────────────────┐        add ──► SkillBank.add
existing skill(s) ┌───────────────┐  │ SkillMaintainer  │──────► merge ─► SkillBank.replace
+ 1 trajectory ───►│ SkillEvolver  │  │ (Fig 9)          │        drop ──► (no-op)
              ────►│ (Fig 8)      │  │ vs. retrieved     │
                   └───────┬───────┘  │ similar skills   │
                   evolve  │          └──────────────────┘
                   (replace │
                    target) ▼
                       SkillBank.replace
```

- Extraction (Fig 6/7) sees only trajectories, **never the existing bank**.
- Evolution (Fig 8) sees relevant existing skills plus one new trajectory,
  and revises a named target directly -- no separate maintenance step.
- Maintenance (Fig 9) sees a candidate skill and retrieved similar skills
  but **no trajectory at all** -- the appendix (Fig 13's judge description)
  is explicit that "maintain decisions are made without a trajectory."
  `SkillBank.replace` implements `merge` faithfully: the maintenance policy
  produces the full merged `{title, when_to_apply, rules}` object, which
  *replaces* the target skill in place, rather than a mechanical union of
  two rule lists.

## What's this codebase's own addition, not the paper's

- **`codeskill/grounding.py`** -- a dependency-free lexical-overlap check
  between a candidate skill's text and the trajectory steps that were
  actually observed. The paper relies entirely on an LLM judge's
  qualitative "groundedness" rubric dimension to catch hallucination; this
  adds a cheap automated safeguard *before* a candidate even reaches that
  (expensive) judge call, in the same spirit as what the reconstruction
  project calls a `prompts/runtime/` addition layered on `prompts/paper/`.
- The exact **user-message formatting** (how a trajectory or a skill gets
  serialized into text) in `codeskill/extraction.py` -- the appendix
  specifies the system prompt and what the user message must contain, but
  not its exact layout.
- **`HybridReward`'s quality/execution/alignment weighting** -- the
  abstract describes combining dense rubric feedback with sparse execution
  feedback; the exact weights were never observed and are a reasonable
  default here, not a reported hyperparameter.
- The **GRPO training loop** (`codeskill/grpo.py`) and **SFT dataset
  builder** (`codeskill/sft_data.py`) are original scaffolding built to the
  paper's *description* of warm-start SFT + GRPO with hybrid reward --
  correct in shape, not a port of the authors' training code.

## Modules

- **`codeskill/prompts.py`** -- the nine verbatim appendix prompts (see
  Provenance above), plus each judge's dimension/max-score table.
- **`codeskill/schema.py`** -- `Skill` is exactly the paper's
  `{title, granularity, when_to_apply, rules}`; `Trajectory`/`TrajectoryStep`
  (with grounding helpers); `ExtractionOutcome`/`EvolutionOutcome`/
  `MaintenanceOutcome` mirror the `generate|skip`, `evolve|skip`,
  `add|merge|drop` action vocabularies from Fig 6-9.
- **`codeskill/extraction.py`** -- `TaskSkillExtractor`, `EventSkillExtractor`,
  `SkillEvolver`, `SkillMaintainer`: one class per appendix prompt, sharing
  parsers (`parse_extraction_response`, `parse_evolution_response`,
  `parse_maintenance_response`) between live inference and GRPO rollouts.
- **`codeskill/bank.py`** -- `SkillBank`: add/replace/drop with
  content-hash dedup, lexical retrieval ranking, `compact()` to prune
  skills with a proven-bad track record, and round snapshots for
  auditability.
- **`codeskill/manager.py`** -- `SkillManagerPolicy`: orchestrates
  extract-then-maintain (`process_event_trajectory`,
  `process_task_trajectories`) and evolution (`process_evolution`); `run_round`
  runs event-level extraction+maintenance over a batch of trajectories.
- **`codeskill/downstream_agent.py`** -- `FrozenDownstreamAgent`: retrieves
  relevant skills for a task and delegates solving to a pluggable
  `solve_fn`. Never updated by CODESKILL training -- only the manager
  policy is.
- **`codeskill/rewards.py`** -- `TaskQualityJudge`, `EventQualityJudge`,
  `EvolutionQualityJudge`, `MergeQualityJudge`, `BehaviorAlignmentJudge`
  (Fig 10-14), `ExecutionReward` (sparse pass-rate), and `HybridReward`
  combining them.
- **`codeskill/eval/`** -- `BenchmarkAdapter` interface plus a fully-working
  synthetic `MockBenchmarkAdapter` for tests/demo. `env_bench_adapter()`,
  `swe_bench_verified_adapter()`, and `terminal_bench_2_adapter()` are
  explicit placeholders: real adapters need each benchmark's dataset and
  sandboxed execution environment, which aren't vendored here.

## Try it

```bash
pip install -r requirements.txt   # just pytest; no network/API key needed
python -m codeskill.cli demo
pytest
```

`demo` runs the whole pipeline -- event-driven extraction (Fig 7), bank
maintenance (Fig 9), and a benchmark comparison -- against
`examples/sample_trajectories.json` using a fully deterministic mock LLM
(`codeskill/demo_mock.py`) that dispatches on which of the four real system
prompts it's called with, so it needs no API key. It extracts one skill per
trajectory, then shows that a frozen downstream agent retrieving those
skills clears the (synthetic) benchmark's checks that the same agent misses
with an empty skill bank.

To use a real model instead of the mock:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python -m codeskill.cli extract --trajectories examples/sample_trajectories.json --out-bank bank.json --anthropic
python -m codeskill.cli build-sft --trajectories examples/sample_trajectories.json --out sft.jsonl --anthropic
python -m codeskill.cli evaluate --bank bank.json
```

## Plugging in real benchmarks

`BenchmarkAdapter` is the only interface a real EnvBench / SWE-Bench
Verified / Terminal-Bench 2 integration needs to implement:

```python
class MyAdapter(BenchmarkAdapter):
    name = "swe-bench-verified"
    def load_tasks(self) -> list[BenchmarkTask]: ...       # load real task instances
    def verify(self, task, solve_result) -> bool: ...       # apply patch + run FAIL_TO_PASS/PASS_TO_PASS tests
```

`run_benchmark(adapter, agent)` then works unchanged.
