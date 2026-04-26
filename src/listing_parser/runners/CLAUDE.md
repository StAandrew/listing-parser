# CLAUDE.md — runners package

Context for AI assistants editing `src/listing_parser/runners/`. This
is subordinate to the repo-level `CLAUDE.md`; read that first.

## Why this package exists

The scorer must be invariant to which model produced
`predictions.jsonl`. Splitting "call the model" from "grade the output"
gives us:

- **Reproducibility.** Predictions files land in
  `benchmarks/runs/<slug>/predictions.jsonl` and get committed. Future
  sessions can re-score them with a tightened scorer without having to
  pay for inference again.
- **Provider independence.** The scorer doesn't care whether a
  prediction came from Haiku on Bedrock, Ollama on localhost, or a
  merged-weights checkpoint. Adding a new provider is a matter of
  implementing the `Runner` Protocol — no scorer changes.
- **CI compatibility.** `lp-benchmark score` and `lp-benchmark smoke`
  never import `boto3` or hit a network. Only `lp-benchmark run`
  does, and that dependency is lazy-loaded inside the subcommand
  dispatcher.

## Module responsibilities

| File | Responsibility | Touches I/O? |
|---|---|---|
| `base.py` | `Runner` Protocol + `RunnerResult` dataclass. Pure types. | No. |
| `_output.py` | Fence stripping + parse-and-retry. Called by runner impls, not by the pipeline directly. | No. |
| `bedrock.py` | `BedrockHaikuRunner` — wraps `labelling/teacher.py` verbatim. | Yes: Bedrock Converse API. |
| `pipeline.py` | Gold → predictions driver: gold loading, resume, concurrency, sidecar/log writers. | Yes: filesystem. |

## Hard rules

- **Runners do not clean the model output.** The whole point of
  benchmarking is to measure the un-repaired model. Calling
  `cleaning.clean_extraction` from a runner would lie to the scorer.
  `parse_output_json` (fence stripping + `json.loads`) is the only
  pre-processing allowed.
- **Runners do not retry on schema-validity failure.** The scorer's
  `schema_rate` metric exists to catch that. Retrying hides the
  regressions a production inference stack would still have to live
  with. Parse-failure retry is OK (default: 1), but it's controlled
  by `--max-parse-retries`; set 0 to measure the model's first-pass
  rate honestly.
- **The `Runner` Protocol is the only contract.** Don't add fields to
  `RunnerResult` without a concrete consumer — `usage` is an untyped
  dict on purpose because every provider surfaces a different shape
  and the scorer ignores it entirely.
- **`pipeline.py` is the only filesystem writer.** A runner's
  `predict()` returns a `RunnerResult` in memory; the pipeline writes
  JSONL, sidecars, and logs. Runner implementations that write to disk
  themselves break resume semantics.

## When adding a new runner

1. Create `src/listing_parser/runners/<provider>.py`.
2. Implement the `Runner` Protocol:
   - `name: str` — short slug used in the report header and log lines
     (e.g. `"ollama-llama31-8b-q6"`).
   - `async def predict(description, listing_type) -> RunnerResult`.
3. Inside `predict`, call your LLM with `build_system_prompt()` +
   `build_user_message(description, listing_type)`. Use
   `runners._output.parse_with_retry` for fence stripping + retry.
4. Register the runner in `benchmarks/cli.py::_RUNNER_BUILDERS` with a
   short slug. Keep the import inside the builder function so
   `lp-benchmark score`/`smoke` don't pay for provider dependencies.
5. Add unit tests in `tests/test_runners.py` using a fake runner (see
   the existing `_FakeRunner` pattern). Do NOT hit the provider in CI.

## Resume semantics

- The sidecar is `_row_ids.txt` inside the run's out-dir, one integer
  `row_index` per line. Present in the sidecar ⇒ skipped on re-run.
- `predictions.jsonl` is opened in append mode. Two resume runs can
  produce a file where rows appear in any order; the scorer joins by
  `row_index`, so order doesn't matter.
- Corrupt / truncated sidecar lines are tolerated (see
  `_load_resume_ids`). This matters because a Ctrl-C during a write
  can produce a partially-flushed line; rejecting those would wipe
  otherwise-good progress.
- Pass `--no-resume` to disable. Don't delete the sidecar yourself —
  `--no-resume` is the intended signal that survives in `git log`.

## Testing

Runner tests live in `tests/test_runners.py`. We don't invoke real
providers; instead we use a `_FakeRunner` that implements the
Protocol and returns pre-seeded results. Test targets:

- `parse_with_retry` correction-prompt wiring (parse failure -> second
  turn carries the raw + error).
- `_PredictionsWriter` append semantics and sidecar invariants.
- `run_predictions` resume skipping, empty-gold handling, and failure
  rows still landing on disk.
- `lp-benchmark run` CLI wiring — subparser registered, missing-gold
  returns rc=2.
