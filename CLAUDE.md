# CLAUDE.md — listing-parser

Conventions and context for AI assistants working on this repo. Read
this alongside `README.md` (the human-facing doc); nothing here
contradicts it.

## What this project is (one paragraph)

A small Python codebase that fine-tunes Llama 3.1 8B via Unsloth to
extract structured JSON from UK property listing descriptions. Two
deliverables land here: (1) the canonical prompt + schema, and (2) the
benchmarking infrastructure (frozen test set + scorer + reports) that
tells us whether training actually helped. The fine-tune itself runs on
rented GPU time, not in this repo; its inputs (labelled data) live on
Hugging Face; its outputs (`predictions.jsonl`) are scored here.

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
| `prompt.md` | humans only | Source of truth for the task definition. |
| `examples.json` | humans only | Worked examples; referenced from the prompt. |
| `src/listing_parser/schema.py` | humans only | Mirrors `prompt.md`; keep in sync. |
| `src/listing_parser/benchmarks/*.py` | code changes | Pure logic, no state. |
| `data/` | external tools | Raw CSV exports of listing descriptions. Read-only from code's perspective. |
| `benchmarks/test_set.jsonl` | `test_set.py --force` only | Frozen; regenerating invalidates history. |
| `benchmarks/test_set.jsonl.meta.json` | follows `test_set.jsonl` | Repro metadata. |
| `benchmarks/runs/<slug>/predictions.jsonl` | runners | One subdir per run. |
| `benchmarks/runs/<slug>/report.{md,json}` | `lp-benchmark score` | Regenerable from predictions. |
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

# Score a run
lp-benchmark score \
    --gold benchmarks/test_set.jsonl \
    --predictions benchmarks/runs/<slug>/predictions.jsonl \
    --out-dir benchmarks/runs/<slug> \
    --name "<human-readable run name>"
```

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

## When adding a new runner (future PR)

- Put it in `src/listing_parser/runners/<provider>.py`.
- The runner's only contract with the rest of the system is: given a
  gold JSONL and a destination dir, write `predictions.jsonl` with
  `{row_index, pred, raw}` rows.
- Runners handle their own rate limiting, retries, and fence-stripping
  of raw model output. Move fence-stripping logic into a shared
  helper if more than one runner needs it.
- The scorer should need **zero** changes to support a new runner.

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
