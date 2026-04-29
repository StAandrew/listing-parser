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
| `bedrock.py` | `_BedrockRunnerBase` + three concrete runners: `BedrockHaikuRunner` (teacher, Converse + caching), `BedrockLlamaRunner` (student base, Converse no caching), `BedrockFineTuneRunner` (CMI import, InvokeModel). Dispatches on the `_USES_INVOKE_MODEL` class flag between `labelling/teacher.py` (Converse) and `labelling/bedrock_invoke.py` (InvokeModel). | Yes: Bedrock Converse + InvokeModel APIs. |
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

### If the new runner is also Bedrock-backed

Subclass `_BedrockRunnerBase` instead of re-implementing from
scratch. Set the class-level defaults:

- `_DEFAULT_NAME` — short slug used in report headers
- `_DEFAULT_MODEL_ID` — set to `None` if caller must always provide
  one (e.g. CMI ARNs are account-specific, so
  `BedrockFineTuneRunner._DEFAULT_MODEL_ID = None` and its
  constructor raises if one isn't passed)
- `_DEFAULT_REGION` — where the model lives; note CMI isn't in
  `eu-west-2` so Frankfurt is the default for `bedrock-ft`
- `_DEFAULT_TEMPERATURE` — 0.0 for baselines (reproducibility),
  0.2 for Haiku (matches labelling)
- `_DEFAULT_USE_CACHE` — True for Anthropic Converse (cachePoint
  supported), False for everything else
- `_USES_INVOKE_MODEL` — False (default) for foundation models on
  Converse; True for CMI imports which reject Converse and require
  InvokeModel with a pre-rendered Llama prompt template

See `BedrockHaikuRunner`, `BedrockLlamaRunner`, and
`BedrockFineTuneRunner` for the pattern — none have a custom
`predict()`, all the protocol dispatching lives in the base class.

Both `_DEFAULT_USE_CACHE` and `_USES_INVOKE_MODEL` are load-bearing
flags that break *every* row of a run if set wrong:

- Using Converse against a CMI-imported model returns "This action
  doesn't support the model that you provided" on every call.
- Using InvokeModel against a foundation model sends a Llama-
  flavoured prompt string to Anthropic, which either errors or
  produces garbage.
- Sending cachePoint to a Llama model returns ValidationException.

These are pinned with wire-level tests in
`tests/test_labelling.py` (`test_sync_converse_*` and
`test_sync_invoke_*`) and default-regression tests in
`tests/test_runners.py` (`test_bedrock_*_defaults_*` and
`test_bedrock_haiku_and_llama_use_converse`).

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
- Bedrock runner default regression tests, including the
  `_USES_INVOKE_MODEL` dispatch flag — flipping it wrong breaks
  every row of a run, so the tests guard both directions (Haiku/
  Llama must NOT use InvokeModel, FineTune MUST use it).

Protocol-shape tests live in `tests/test_labelling.py` because that's
where the low-level Bedrock clients live:

- `test_sync_converse_*` — Converse cachePoint toggle, system-block
  shape.
- `test_render_llama31_prompt_*` — byte-exact Llama chat template
  rendering (must match Unsloth training template).
- `test_sync_invoke_*` — InvokeModel payload/response shape and
  `ModelNotReadyException`-as-retryable-throttle conversion.

## Operational notes on CMI

`BedrockFineTuneRunner` is the only CMI-served runner today.
Specifics that apply to it but not the foundation-model runners:

- **Cold start**: first InvokeModel call after >5 min idle takes
  60–120s. Our `bedrock_invoke.call_imported` treats
  `ModelNotReadyException` as a retryable throttle, so the
  pipeline's existing backoff loop handles it — the user just sees
  "row 0 took 90s" on the progress line. Don't try to "fix" this
  by kicking the endpoint before scoring; Bedrock's pricing window
  starts on the first real call and you'd be paying for warmup.
- **Quota**: default on-demand quota is ~1 req/s. `--concurrency 2`
  or higher triggers sustained throttling; stay at 1 unless you've
  got a quota increase from AWS Service Quotas.
- **Region**: CMI isn't available in `eu-west-2`. The runner
  defaults to `eu-central-1` (Frankfurt); override via `--region`
  if you've imported elsewhere.
- **Economics**: CMI bills per 5-minute window of activity, $0.05718
  per CMU per minute (us-east-1) or $0.07144 (eu-central-1), 2 CMUs
  for Llama 3.1 8B. Scale-to-zero when idle + $3.90/month storage
  per imported model. Roughly 5 hours/day of sustained invocation
  is the break-even vs a SageMaker always-on endpoint; below that,
  CMI is cheaper. See `README.md` §8 for the full breakdown.
