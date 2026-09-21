# codeskill

An implementation of the architecture described in **"CODESKILL: Learning
Self-Evolving Skills for Coding Agents"** (Li, Zhang, Zhang, Liu, Liu;
arXiv:2605.25430), using the paper's **actual appendix prompts and reward
formula**, verified directly against the primary PDF.

## Verification status

This codebase went through two passes:

1. **First pass** was built from the abstract plus a third-party
   reconstruction project's (`rayyichen310/codeskill-rebuild`) public,
   checksum-cited transcription of the appendix prompts -- arxiv.org,
   huggingface.co, researchgate.net, and alphaxiv.org were all blocked by
   network egress policy at the time, so the primary PDF wasn't
   independently readable.
2. **Second pass** read the actual paper PDF directly (supplied via a
   connected Google Drive) and checked the implementation against it
   line by line. Result:
   - **Figures 6-14 (`codeskill/prompts.py`)**: confirmed verbatim-correct.
     The third-party transcription the first pass relied on matched the
     primary source exactly.
   - **The reward formula (`codeskill/rewards.py`) was wrong** and has
     since been fixed. See "Corrected against the paper" below.
   - Two more paper details -- same-instance retrieval leakage filtering
     and the real retrieval embedding model name -- were added/documented
     from the verified text.

Where the paper's own code, exact hyperparameters not printed in its
tables, or design choices below the level the appendix specifies were never
observed, this repo makes a reasonable, clearly-documented engineering
choice rather than guessing at unobserved specifics. Those choices are
called out inline, in the module docstrings.

## Corrected against the paper

Reading the full PDF surfaced a real bug: `HybridReward` computed
`quality_weight*RQ + execution_weight*RE + alignment_weight*RA` as three
independently-weighted terms. The paper's actual formula (Section 3.3.2,
Algorithm 1) is different in kind, not just in weights:

```
R(u; q) = lam * R_Q(u; q) + R_A(u; tau_u^pi) * R_E(u; x_u, pi)   # u produces an injectable skill (generate/evolve/merge)
R(u; q) = lam_dec * R_Q(u; q)                                     # otherwise (add/drop/skip)
```

with **lam = 0.25** (Table 4's "Quality reward weight"). Alignment
*multiplies* execution reward rather than being added alongside it -- it's
a credit-assignment gate: a skill only earns execution credit when the
agent's behavior is judged to actually reflect it, not merely correlate
with task success. `lam_dec`'s exact value is referenced in Algorithm 1 but
never given numerically anywhere in the paper; it defaults to `lam` here as
a documented assumption. `codeskill/rewards.py`'s `HybridReward.combine()`
now implements this exactly, and `ExecutionReward` was rewritten to match
Algorithm 1's actual mechanic: a pre-cached no-skill baseline (average of
**n=4** rollouts per task, Appendix B) and *reverse retrieval* --
`x_u ~ TopK(s_u, D_task)`, ranking the task pool by the **skill's** content
rather than a query -- instead of a plain pass-rate over a fixed eval set.

Also added from the verified text: `SkillBank.retrieve()` now accepts
`exclude_trajectory_ids` and `FrozenDownstreamAgent.attempt()` exposes it,
implementing Appendix C's "skills generated from the same evaluation
instance are filtered out to avoid same-instance leakage." `GRPOTrainer`'s
`group_size`/`temperature` defaults (6, 0.7) now match Table 4's RL
settings. The real retrieval encoder is confirmed as
`sentence-transformers/all-MiniLM-L6-v2` with separate dense indexes per
benchmark and skill granularity (Appendix C) -- this repo's own
`SkillBank.retrieve()` still uses lexical Jaccard overlap as a
dependency-free stand-in, now correctly cited rather than guessed at.

## Architecture

The appendix specifies four manager-policy stages, each its own prompt --
not one combined "extract and maintain" call:

```
2-3 related          ┌────────────────────┐
trajectories ────────►│ TaskSkillExtractor │  generate | skip
                      │ (Fig 6)            │──┐
                      └────────────────────┘  │
one full trajectory   ┌────────────────────┐  │
              ────────►│ EventSkillExtractor│──┤  candidate Skill
                      │ (Fig 7)            │  │  {title, granularity,
                      └────────────────────┘  │   when_to_apply, rules}
                                               │
existing skill(s)     ┌────────────────────┐  │
+ 1 trajectory ───────►│ SkillEvolver       │──┤  (candidate + its
                      │ (Fig 8) -> evolve   │  │   named target_skill_id)
                      └────────────────────┘  │
                                               ▼
                                      ┌───────────────────┐   add   ──► SkillBank.add,
                                      │ SkillMaintainer    │            or SkillBank.replace(target)
                          ┌──────────►│ (Fig 9)            │─► merge ──► SkillBank.replace(merge_target)
                          │           │ vs. retrieved       │   drop  ──► (no-op; evolution's
                    retrieved         │ similar skills      │            target, if any, unchanged)
                    similar skills    └───────────────────┘
```

- Extraction (Fig 6/7) sees only trajectories, **never the existing bank**.
- Evolution (Fig 8) sees relevant existing skills plus one new trajectory,
  and proposes a revision naming its target skill.
- **Both extraction and evolution outputs then pass through the same
  maintenance stage** (Fig 9) -- the paper is explicit that "each newly
  extracted or evolved candidate skill is further passed to a maintenance
  stage" (Section 3.2, repeated verbatim in Appendix C). An earlier version
  of this codebase wrote evolution's revision straight to the bank,
  skipping maintenance; `SkillManagerPolicy.process_evolution` now routes
  it through `_maintain` like extraction does. For an evolution-sourced
  candidate, maintenance's `add` commits the revision onto the evolver's
  named target; `merge` may still fold it into a *different* retrieved
  skill; `drop` leaves the target unchanged.
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
- **`lam_dec`** in `HybridReward` -- Algorithm 1 references it for the
  add/drop/skip branch but the paper never states a value distinct from
  `lam`; it defaults to `lam` here as a documented assumption.
- **Lexical (Jaccard) retrieval** in `SkillBank.retrieve()` /
  `reverse_retrieve()` -- the paper uses dense embeddings via
  `sentence-transformers/all-MiniLM-L6-v2` with per-benchmark,
  per-granularity indexes (Appendix C); this repo uses a dependency-free
  lexical stand-in instead, now correctly cited rather than guessed at.
- The **online, per-instance streaming evaluation protocol** Appendix C
  describes (collect a no-skill rollout on each incoming instance, extract
  skills from it, maintain the bank, then solve that same instance with
  everything already in the bank *except* its own just-extracted skills) is
  not replicated by `codeskill/pipeline.py`, which instead builds the bank
  from an offline trajectory batch and then evaluates baseline-vs-with-skills
  as two separate passes over a benchmark. The same-instance-leakage
  mechanism (`exclude_trajectory_ids`) is implemented and tested, but
  wiring the full streaming loop end-to-end is left as an exercise.
  Relatedly, Appendix C also notes each real instance produces "about one
  task-level skill and three event-driven skills" -- i.e. the extractor is
  called several times per instance (presumably at nonzero sampling
  temperature) -- while `SkillManagerPolicy.process_event_trajectory`
  calls it once per trajectory; call it multiple times yourself if you want
  that same multi-candidate-per-instance behavior.
- **Retrieval timing differs by granularity in the paper**: task-level
  skills are retrieved once before solving starts, but event-driven skills
  are retrieved *online, mid-rollout*, re-querying with "recent reasoning,
  executed actions, observations, error messages, command outputs" as the
  agent proceeds (Appendix C). `FrozenDownstreamAgent.attempt` does query
  the two granularities as **separate retrieval calls with independent
  `top_k` budgets** now (an earlier version pooled both into one shared
  `top_k`, letting one granularity crowd out the other -- fixed, since
  Appendix C is explicit that "we build separate retrieval indexes for
  task-level and event-driven skills"), but it's still a single-shot
  `solve_fn` abstraction with no multi-turn rollout loop, so both calls use
  the same initial task description rather than a live, evolving query --
  a structural simplification, not a tunable parameter.
- The **GRPO training loop** (`codeskill/grpo.py`) and **SFT dataset
  builder** (`codeskill/sft_data.py`) are original scaffolding built to the
  paper's *description* of warm-start SFT + GRPO with hybrid reward --
  correct in shape (including the group-relative advantage formula and
  Table 4's group size/temperature/lambda), not a port of the authors'
  training code or an implementation of the actual GRPO gradient step.

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
  (Fig 10-14); `NoSkillBaselineCache` + `ExecutionReward` (Algorithm 1's
  pre-cached baseline and reverse-retrieval execution reward); and
  `HybridReward`, implementing Algorithm 1's `R = lam*RQ + RA*RE` /
  `R = lam_dec*RQ` combination rule exactly.
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

## The paper's actual reported numbers

For context, don't confuse these with anything this repo's own `demo`
prints (that's a 3-example synthetic smoke test, not a benchmark result).
The paper reports, with Qwen3.5-4B as the skill-manager backbone and
Qwen3.5-35B-A3B as the frozen downstream coding policy (Table 1): average
pass rate across EnvBench/SWE-Bench-Verified/Terminal-Bench-2 improves from
29.57 (no-skill) to 39.26 (CODESKILL) -- a +9.69 absolute / ~33% relative
gain -- and beats the strongest baseline (GPT-5.4-mini prompt-based skill
management, 35.25) by +4.01. With GPT-5.4-mini as the frozen downstream
policy instead, CODESKILL still wins: 21.80 -> 30.73 average pass rate.
Full lifecycle maintenance (add/merge/drop) shrinks the skill bank from
1252 skills (extraction only) to 676 while costing only ~2% average pass
rate -- the compaction the abstract calls "stable size."

Training used a 3-phase curriculum (extraction-only, +evolution,
+maintenance; 130/120/250 GRPO steps respectively, 500 total) over
trajectories from SWE-Bench Verified, SWE-smith, and EnvBench, with
GPT-5.4-mini as the SFT teacher and RL judge.

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
