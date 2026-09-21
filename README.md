# codeskill

An open, from-scratch reimplementation of the architecture described in
**"CODESKILL: Learning Self-Evolving Skills for Coding Agents"** (Li, Zhang,
Zhang, Liu, Liu; arXiv:2605.25430).

## What this is (and isn't)

The paper's own code is not public. This package was built from:

- the paper's abstract and the summaries surfaced through search (full-text
  access to arXiv/HuggingFace/etc. was blocked in the environment this was
  written in), and
- the public architecture notes of a third-party reconstruction effort
  (`rayyichen310/codeskill-rebuild`), which named concrete design choices
  (task/event granularity, per-trial frozen skill banks with provenance, a
  "C-only" grounding rule for task-level extraction, staged
  maintenance/compaction, explicit output classification before JSON
  parsing) that this implementation adopts directly.

So: this is a faithful-to-the-description reimplementation of the
*architecture*, not a byte-for-byte port of the authors' code, exact
prompts, or hyperparameters. Where the description was genuinely
underspecified (exact rubric wording, exact reward weighting, retrieval
embedding model, RL hyperparameters), this repo makes a reasonable,
clearly-documented design choice rather than guessing at specifics that
were never observed.

## Architecture

```
Trajectory ──► SkillExtractor ──► SkillOperation(s) ──► SkillManagerPolicy ──► SkillBank
  (agent           (LLM call,         (add/update/           (applies ops,        (versioned,
   episode)       grounding-checked)   merge/drop/noop)     runs compaction)     retrievable,
                                                                                   auditable)

SkillBank ──retrieve()──► FrozenDownstreamAgent ──attempt()──► SolveResult ──► BenchmarkAdapter.verify()
                            (coding agent, never
                             updated by training)
```

- **`codeskill/schema.py`** -- `Trajectory`/`TrajectoryStep` (with a
  `TrajectoryStep.is_grounded()` / `Trajectory.is_grounded_span()` pair that
  enforce "never cite a step whose tool result wasn't actually observed"),
  and `Skill`/`Provenance`/`SkillOperation`.
- **`codeskill/bank.py`** -- `SkillBank`: add/update/merge/drop with
  content-hash dedup, lexical retrieval ranking, `compact()` to prune
  skills with a proven-bad track record, and round snapshots for
  auditability.
- **`codeskill/extraction.py`** -- `SkillExtractor`: prompts an LLM to
  propose operations from a trajectory, pre-classifies the raw output
  before attempting to parse it as JSON, and rejects any operation that
  cites an ungrounded trajectory step.
- **`codeskill/manager.py`** -- `SkillManagerPolicy`: applies proposed
  operations to a bank and runs a full "round" (a batch of trajectories,
  followed by compaction).
- **`codeskill/downstream_agent.py`** -- `FrozenDownstreamAgent`: retrieves
  relevant skills for a task and delegates solving to a pluggable
  `solve_fn`. This agent is never updated by CODESKILL training -- only the
  manager policy is.
- **`codeskill/rewards.py`** -- the hybrid reward: `RubricReward` (dense,
  LLM-judged skill quality: grounding/clarity/non-redundancy/alignment) and
  `ExecutionReward` (sparse, verifiable: does the skill actually help the
  frozen agent pass held-out tasks), combined by `HybridReward`.
- **`codeskill/sft_data.py`** -- builds the warm-start SFT dataset from
  teacher-labeled trajectories, in the same prompt/completion shape used at
  inference time.
- **`codeskill/grpo.py`** -- the GRPO training loop shape: sample a group of
  candidate completions per trajectory, score each with the hybrid reward
  (reusing the *exact* grounding/parsing logic used at inference time, so
  training and inference can't drift apart), and compute group-relative
  advantages. The actual gradient step is behind a `PolicyModel.update`
  interface -- running it for real needs a base model and training stack
  this repo doesn't vendor.
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

`demo` runs the whole pipeline -- extraction, grounding checks, bank
maintenance, and a benchmark comparison -- against `examples/sample_trajectories.json`
using a fully deterministic mock LLM (`codeskill/demo_mock.py`), so it needs
no API key. It extracts one skill per trajectory (each trajectory encodes a
concrete fix), then shows that a frozen downstream agent retrieving those
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
