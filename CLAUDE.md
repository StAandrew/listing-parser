# CLAUDE.md — listing-parser

Conventions and context for AI assistants working on this repo. Read
this alongside `README.md` (the human-facing doc); nothing here
contradicts it.

## What this project is (one paragraph)

A small Python codebase that fine-tunes Llama 3.1 8B via Unsloth to
extract structured JSON from UK property listing descriptions. Three
deliverables land here: (1) the canonical prompt + schema, (2) the
benchmarking infrastructure (frozen test set + scorer + reports) that
tells us whether training actually helped, and (3) the runner layer
that calls models on Bedrock — teacher (Haiku, Converse), student
base (Llama, Converse), and the fine-tuned student (CMI-imported
Llama, InvokeModel) — and produces `predictions.jsonl` for the
scorer. The fine-tune itself runs in a Colab notebook
(`Listing_Parser_Fine_Tune_Unsloth.ipynb`) on rented A100 time;
inputs (labelled data) and outputs (merged weights) live on Hugging
Face. The Bedrock CMI deployment is documented in `README.md` §7–8.

## Priorities in order

1. **Don't break the frozen test set.** `benchmarks/test_set.jsonl` is
   the anchor for every historical benchmark number. Regenerating it
   with a different seed silently invalidates all prior `report.md`
   files. The `test_set.py` CLI refuses to overwrite without `--force`
   for this reason; keep that safety rail.
2. **Keep the scorer pure.** `scorer.py` must not do I/O beyond reading
   two JSONL files. No HTTP calls, no database, no subprocess. Its
   output must be byte-identical for a given (gold, predictions) pair.
3. **Keep `prompt.md` and `schema.py` in sync.** The enums in
   `schema.py` mirror the vocabularies listed in `prompt.md`. If you
   add a vocabulary value in one, add it in the other in the same
   commit. A lint / diff helper is TODO.
4. **Every scorer change ships with a test.** The scorer's numbers
   drive training decisions. Regressions in the scorer are more
   expensive than regressions in the model.

## Conventions

### Formatting & linting

- Python 3.11+. Target the syntax, use `from __future__ import
  annotations` when convenient.
- Ruff handles both formatting and linting. Config is in
  `pyproject.toml`. Line length 100. `select = E F W I B UP`.
- No `black` / `isort` / `mypy` on top — ruff is enough at this scale.

### File naming

- Modules: `snake_case.py`.
- Tests: `test_<module>.py` in `tests/`.
- Benchmark runs: `benchmarks/runs/<short-slug>/` with kebab-case slugs
  like `haiku-4.5-teacher`, `ollama-llama31-8b-q6`, `ft-v1-epoch3`.

### Comments

- Keep existing comments. When writing new code, add brief comments
  explaining *why* — non-obvious intent, domain context, or a
  constraint the code can't convey (e.g. "scored order-insensitively
  because descriptions hint at floors non-linearly"). Do not strip
  comments during refactors.
- Don't narrate trivia. "increment the counter" is noise; "5% relative
  tolerance — anything tighter fails on the 1478 sqft conversion" is
  signal.

### Imports

- Stdlib first, third-party next, first-party last. Ruff's `I` rule
  enforces this.
- First-party imports use the package root: `from listing_parser.schema
  import Extraction`, not relative imports.

### Errors

- CLIs return integer exit codes; reserve `2` for usage errors, `1`
  for expected failures (e.g. smoke check failed), `0` for success.
- The scorer never raises on bad predictions — parse failures become
  `parse_ok=False` on the ExampleScore. Raising would stop the run mid
  way through a 500-row predictions file, which is worse than a bad
  number.

## Directory contract

| Path | Writable by | Notes |
|---|---|---|
| `prompt.md` | humans only | Source of truth for the task definition. Mirrored to `src/listing_parser/_assets/prompt.md` for package distribution — keep in sync (they're used interchangeably by `prompting.load_prompt_markdown`). |
| `examples.json` | humans only | Worked examples; referenced from the prompt. Same asset-mirror as above. |
| `src/listing_parser/_assets/` | follows root | Mirror of `prompt.md` + `examples.json` so pip-installed consumers (notably the Colab fine-tune notebook) work without a repo checkout. |
| `src/listing_parser/schema.py` | humans only | Mirrors `prompt.md`; keep in sync. |
| `src/listing_parser/benchmarks/*.py` | code changes | Pure logic, no state (except `cli.py` which does I/O through subcommands). |
| `src/listing_parser/labelling/teacher.py` | code changes | Bedrock **Converse** API client, with cachePoint support for Anthropic models. |
| `src/listing_parser/labelling/bedrock_invoke.py` | code changes | Bedrock **InvokeModel** API client, with Llama-3.1 chat template rendering for CMI-imported models. Byte-identical to the Unsloth-trained template. |
| `src/listing_parser/runners/bedrock.py` | code changes | `_BedrockRunnerBase` + three concrete runners. Dispatches between Converse and InvokeModel on a class flag. |
| `Listing_Parser_Fine_Tune_Unsloth.ipynb` | humans only | Colab fine-tune notebook. Commit with outputs cleared — see "Notebook hygiene" below. |
| `data/` | external tools | Raw CSV exports of listing descriptions. Read-only from code's perspective. |
| `benchmarks/test_set.jsonl` | `test_set.py --force` only | Frozen; regenerating invalidates history. |
| `benchmarks/test_set.jsonl.meta.json` | follows `test_set.jsonl` | Repro metadata. |
| `benchmarks/runs/<slug>/predictions.jsonl` | `lp-benchmark run` | One subdir per run; committed. |
| `benchmarks/runs/<slug>/report.{md,json}` | `lp-benchmark score` | Regenerable from predictions; committed. |
| `benchmarks/runs/<slug>/_row_ids.txt` | `lp-benchmark run` | Resume sidecar; gitignored. |
| `benchmarks/runs/<slug>/_log.jsonl` | `lp-benchmark run` | Per-row diagnostics; gitignored. |
| `scripts/` | code changes | One-off utilities (data prep, HF pushes, augmentation). Not on the runtime path. |
| `tests/` | follows code | Mirror the module structure. |

## Running things

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests
pytest -q

# Freeze the test set (ONCE per labelled-dataset version)
python -m listing_parser.benchmarks.test_set \
    --repo standrey/listing-descriptions --config data --n 60

# Smoke the scorer (expect 1.0 on parse/schema/macro-accuracy)
lp-benchmark smoke --gold benchmarks/test_set.jsonl

# Score predictions
lp-benchmark score \
    --gold benchmarks/test_set.jsonl \
    --predictions benchmarks/runs/<slug>/predictions.jsonl \
    --out-dir benchmarks/runs/<slug> \
    --name "<human-readable run name>"

# Generate predictions from a runner (pick one):
#   --runner bedrock-haiku    — teacher model
#   --runner bedrock-llama    — student base model
#   --runner bedrock-ft       — CMI-imported fine-tune (needs --model-id ARN)
AWS_PROFILE=<profile> lp-benchmark run \
    --runner bedrock-haiku \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/<slug> \
    --concurrency 4
```

Full fine-tune + CMI-deploy workflow is in `README.md` §7–8.

## Scorer internals (short version)

- `FIELD_KINDS` / `NESTED_FIELD_KINDS` in `schema.py` drive everything.
  Each field path maps to a comparator kind (`scalar_enum`,
  `list_set`, `list_dict`, `nested_block`, `evidence_map`, etc.).
- `scorer.score_one(gold, pred, description, index)` returns an
  `ExampleScore`. Aggregation is a separate pass
  (`aggregate(scores) -> AggregateReport`) so the per-example objects
  stay inspectable for later analysis (e.g. finding the worst
  predictions to hand-review).
- `FieldResult.tp / fp / fn / wrong` are derived properties, not
  stored. Adding a new comparator requires only setting
  `correct / gold_present / pred_present`.
- Evidence grounding uses `_norm_text` which lowercases + collapses
  whitespace + maps NBSP → space + maps en-dash → hyphen. Add more
  mappings there if real-world descriptions reveal new noise sources;
  mirror with a test case.

## When adding a new field to the schema

1. Update `prompt.md` with the field name, type, and any enum values.
2. Update `schema.py`:
   - add the Pydantic field to the right model (`Extraction`,
     `RentBlock`, `SaleBlock`, `RoomBlock`, `FlagsBlock`);
   - add an entry to `FIELD_KINDS` or `NESTED_FIELD_KINDS`;
   - if the field is an enum, add the vocabulary tuple + an entry in
     `_ENUM_VOCAB`.
3. Add a unit test in `tests/test_scorer.py` covering the comparator
   for the new field.
4. Do NOT touch `benchmarks/test_set.jsonl` — the frozen set doesn't
   include the new field, which is fine: the scorer treats missing
   gold values as "absent" and won't crash.

## When adding a new runner

The runner Protocol lives in `runners/base.py`. Three concrete
runners ship today (`BedrockHaikuRunner`, `BedrockLlamaRunner`,
`BedrockFineTuneRunner`), all subclassing `_BedrockRunnerBase` which
handles the shared semaphore, system-prompt rendering, and dispatch
between the two Bedrock wire protocols.

To add a new Bedrock-backed runner:

1. Subclass `_BedrockRunnerBase`, override `_DEFAULT_*` class
   constants. Set `_USES_INVOKE_MODEL = True` if it's a CMI import
   (CMI rejects Converse); leave False for foundation models.
2. Register the runner in `benchmarks/cli.py::_RUNNER_BUILDERS`.
3. Add tests to `tests/test_runners.py` pinning the critical defaults
   (region, model_id, use_cache, invoke-model flag). `bedrock-ft`'s
   tests are the cleanest template.

To add a non-Bedrock runner (e.g. Ollama, vLLM):

1. Implement the `Runner` Protocol directly (no base class needed).
2. Provide your own client / concurrency / retry logic. The shared
   `_output.parse_with_retry` handles fence stripping + parse-error
   retry; reuse it.
3. Register in `_RUNNER_BUILDERS`, add tests.

The scorer should need **zero** changes to support any new runner.

## Bedrock wire protocols

Two `labelling/` modules own the low-level Bedrock protocols:

- `teacher.py`: Converse API. Used by Haiku (with cachePoint) and
  Llama base (without). Best for foundation models that accept
  structured `messages`/`system` blocks.
- `bedrock_invoke.py`: InvokeModel API. Used by CMI-imported custom
  models, which Converse rejects with "This action doesn't support
  the model that you provided". Renders the Llama-3.1 chat template
  as a single `prompt` string; matches the Unsloth training
  template byte-for-byte.

Both return the same `TeacherResponse` dataclass so the runner layer
can dispatch on a single flag without shape conversion.

If you add a third protocol (e.g. Anthropic's direct Messages API),
put it in `labelling/` alongside these two and teach
`_BedrockRunnerBase.predict` to dispatch to it.

## Notebook hygiene

`Listing_Parser_Fine_Tune_Unsloth.ipynb` is version-controlled, but
saving from Colab reintroduces two kinds of cruft that make diffs
unreadable:

1. `metadata.widgets['application/vnd.jupyter.widget-state+json']`
   — Colab adds this; GitHub's renderer crashes on it, showing the
   notebook as "Invalid". The widget state has no semantic value.
2. Cell outputs — images, training-loss tables, download prompts
   from `google.colab.files.download`, etc. Some outputs can be
   hundreds of KB and balloon the diff.

Before committing the notebook, clean both:

```bash
python -c "
import json
p = 'Listing_Parser_Fine_Tune_Unsloth.ipynb'
nb = json.load(open(p))
nb.get('metadata', {}).pop('widgets', None)
for c in nb.get('cells', []):
    c['outputs'] = []
    c['execution_count'] = None
json.dump(nb, open(p, 'w'), indent=1, ensure_ascii=False)
"
```

If you find yourself running this more than once or twice, lift it
into `scripts/clean_notebook.py` + wire a pre-commit hook.

## Known gotchas

- The labelled HF dataset sometimes has near-duplicate descriptions
  (same apartment complex listed multiple times). `test_set.py`
  deduplicates by the first 200 chars of description. If you
  disable this, the test set concentrates on a single listing.
- Haiku occasionally produces malformed JSON (truncated, or refusal
  text wrapped in a code fence). `parse_output_json` uses
  `raw_decode` to tolerate trailing prose, matching how a production
  LLM consumer should handle this.
- Some labelled rows violate the schema (e.g. hallucinated field
  names, pluralised key variants). The scorer reports
  `schema_rate < 1.0` on gold-vs-gold when this happens — that's a
  signal to clean the labels, not a scorer bug.
- Bedrock Converse rejects CMI-imported models with "This action
  doesn't support the model that you provided" — use InvokeModel
  instead. `BedrockFineTuneRunner` does this automatically via
  `_USES_INVOKE_MODEL=True`, but if you're testing with raw `aws
  bedrock-runtime` from the CLI, remember to use `invoke-model`, not
  `converse`.
- Bedrock CMI cold start is 60–120s on the first call after >5 min
  idle. `bedrock_invoke.py` converts `ModelNotReadyException` to a
  retryable throttle so the runner backs off instead of failing.
- Bedrock CMI isn't available in `eu-west-2` (London) at time of
  writing. `eu-central-1` (Frankfurt) is the nearest EU region with
  CMI support; runner defaults to Frankfurt for `bedrock-ft`.
- The Llama-3.1 chat template must be byte-identical between
  training (Unsloth notebook) and serving (CMI InvokeModel). Cell
  14 of the fine-tune notebook pins this against the canonical Meta
  template; `bedrock_invoke._render_llama31_prompt` is the runtime
  equivalent. If you change one, verify both.
