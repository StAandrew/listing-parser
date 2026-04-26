# listing-parser

Fine-tunes a small LLM (Llama 3.1 8B) to extract structured JSON from UK
property listing descriptions. The extracted JSON follows a strict
schema backed by controlled vocabularies for property types, amenities,
parking, tenure, furnishing, and so on — the output is directly
consumable by a downstream relational database.

End-to-end pipeline this repo owns:

```
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │ raw listing  │───▶│  labelling   │───▶│  fine-tune   │───▶│  Bedrock CMI │
 │ descriptions │    │  pipeline    │    │  Llama 8B    │    │  (inference) │
 │ (JSON + CSV) │    │  (Haiku 4.5  │    │  QLoRA run   │    │              │
 │              │    │  on Bedrock) │    │              │    │              │
 └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                            │                    ▲
                            ▼                    │
                      ┌───────────────────────────────┐
                      │  data/labelled/ + HF mirror   │
                      │  + frozen test set + scorer   │
                      │  (this repo)                  │
                      └───────────────────────────────┘
```

## Status

| Stage | State |
|---|---|
| Prompt + schema design | ✅ `prompt.md`, `examples.json` |
| Teacher labelling pipeline | ✅ `src/listing_parser/labelling/` |
| Frozen test set + scorer | ✅ this repo |
| Teacher baseline on gold | ✅ `benchmarks/runs/haiku-4.5-teacher/report.md` |
| Base-model baseline (Llama 3.1 8B via Bedrock) | ✅ `benchmarks/runs/llama-3.1-8b-base/report.md` |
| Fine-tune (Unsloth, Llama 3.1 8B QLoRA on RunPod) | ☐ next |
| Bedrock Custom Model Import | ☐ |

## Repo layout

```
listing-parser/
├── prompt.md                          System prompt + JSON schema (source of truth)
├── examples.json                      Worked examples used as few-shot + docs
├── data/                              Listings in (JSON) and labelled rows out
│   ├── listing_202604241550.csv          ~11.6k listing descriptions (single column)
│   ├── listing_<stamp>.json              Listings pre-tagged with listing_type — input
│   │                                     to the labeller
│   ├── amenity_<stamp>.json              Amenity vocabulary snapshot
│   └── labelled/                         Authoritative label store (git-tracked)
│       ├── <basename>.json               One JSON array per labelling batch
│       ├── _index.json                   Consolidated dedup'd view across batches
│       ├── _row_ids/                     Resume sidecars (gitignored)
│       └── _log/                         Per-attempt diagnostics (gitignored)
├── benchmarks/                        Frozen eval artefacts, git-tracked
│   ├── test_set.jsonl                    Stratified held-out gold (60 rows today)
│   ├── test_set.jsonl.meta.json          Repro metadata: seed, histograms, source
│   └── runs/                             One subdir per scored run
│       └── <run-name>/
│           ├── predictions.jsonl            Produced by a future runner
│           ├── report.md                    Human-readable scorecard
│           └── report.json                  Raw numbers for diffing runs
├── src/listing_parser/
│   ├── schema.py                         Enums + Pydantic model, mirrors prompt.md
│   ├── cleaning.py                       Pure repair rules + parse helpers; single
│   │                                     source of truth for "what valid labels look
│   │                                     like" — used by the labeller AND the migration
│   ├── prompting.py                      Assemble system + user messages; stitches
│   │                                     prompt.md, schema.py, and examples.json
│   ├── labelling/                        Teacher-calling pipeline
│   │   ├── sources.py                       Read data/listing_*.json -> ListingInput
│   │   ├── teacher.py                       Bedrock Converse client, async + caching
│   │   ├── pipeline.py                      Per-row loop: prompt -> teacher -> clean
│   │   │                                    -> validate -> retry -> emit
│   │   └── sinks.py                         Append-only writer + index regeneration
│   ├── runners/                          Gold -> predictions (for benchmarking)
│   │   ├── base.py                          Runner Protocol + RunnerResult
│   │   ├── _output.py                       Shared parse-and-retry (one retry on
│   │   │                                    bad JSON; no schema retry)
│   │   ├── bedrock.py                       BedrockHaikuRunner (reuses teacher.py)
│   │   └── pipeline.py                      Gold JSONL -> predictions.jsonl driver
│   │                                         with resume + per-run sidecar/log
│   └── benchmarks/
│       ├── test_set.py                     Builds benchmarks/test_set.jsonl from HF
│       ├── scorer.py                       Per-field metrics (pure, no I/O)
│       └── cli.py                          `lp-benchmark` entry point (run/score/smoke)
├── tests/                             pytest suite, 62 cases covering cleaning,
│                                      prompting, scorer edge cases, and test-set
│                                      fence-stripping
├── scripts/                           One-off CLIs (not on the runtime path)
│   ├── label_listings.py                 Label a listing_*.json batch via Haiku
│   ├── push_labels_to_hf.py              Mirror data/labelled/_index.json to HF
│   └── clean_hf_labels.py                One-off migration for already-uploaded rows
├── pyproject.toml
├── CLAUDE.md                          AI-assistant conventions for THIS repo
└── README.md                          (this file)
```

## Install

Python 3.11+ required. Mac-friendly — everything here runs on a laptop.

```bash
git clone <repo-url> listing-parser && cd listing-parser
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Smoke-test the install:

```bash
pytest                     # 74 tests, <1s
lp-benchmark smoke --gold benchmarks/test_set.jsonl   # gold-vs-gold
```

`lp-benchmark smoke` is a self-check: it scores the gold test set
against itself, so `parse_rate`, `schema_rate`, and `macro_accuracy`
should all be 1.0. Anything less is a bug in the scorer, not the data.

## Typical workflows

### 1. Freeze (or refresh) the benchmark test set

Runs **once** per labelled-dataset version. Pulls a stratified sample
(Rent / Sale / Room) from a Hugging Face dataset of
`(description, output)` pairs, parses the fenced JSON, deduplicates by
description prefix, and writes the chosen rows to
`benchmarks/test_set.jsonl`.

```bash
python -m listing_parser.benchmarks.test_set \
    --repo standrey/listing-descriptions \
    --config data \
    --split train \
    --n 60 \
    --out benchmarks/test_set.jsonl
```

The command refuses to overwrite an existing file; pass `--force` to
override (which invalidates historical benchmark runs — don't do it
casually).

Commit the output **and** the `.meta.json` sidecar, which records the
HF repo, split, seed, and class histogram. That's enough to reproduce
the exact same test set later.

### 2. Generate predictions for a run

Point a runner at the frozen gold set and let it write
`predictions.jsonl` into a run-specific subdirectory. The runner
re-calls the model for every gold row; the scorer reads the output
offline.

```bash
# Teacher baseline — Haiku 4.5, eu-west-2, uses prompt caching.
AWS_PROFILE=XXXXXXX lp-benchmark run \
    --runner bedrock-haiku \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/haiku-4.5-teacher \
    --concurrency 4

# Student base-model baseline — Llama 3.1 8B Instruct on Bedrock,
# us-west-2, greedy decoding, no prompt caching. Matches what the
# fine-tune will be served as via Bedrock CMI.
AWS_PROFILE=XXXXXXX lp-benchmark run \
    --runner bedrock-llama \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/llama-3.1-8b-base \
    --concurrency 4
```

Two runners ship today:

- `bedrock-haiku` — Claude Haiku 4.5 (the teacher). Defaults: region
  `eu-west-2`, model `global.anthropic.claude-haiku-4-5-20251001-v1:0`,
  temperature 0.2, prompt caching on. The 1h system-prompt cache makes
  repeat runs effectively free after the first ~4 rows.
- `bedrock-llama` — Llama 3.1 8B Instruct on Bedrock (the student base
  model). Defaults: region `us-west-2`, model
  `meta.llama3-1-8b-instruct-v1:0`, temperature 0.0, prompt caching
  **off** (Bedrock's Llama integration rejects the `cachePoint`
  directive). This is the baseline the fine-tune has to beat — same
  weights family, same serving precision, no laptop-quantization
  confounds.

Why not Ollama + `hf.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M`
as the baseline? Q4_K_M routinely drops 1–3 points on structured-
extraction tasks vs fp16/bf16, and the fine-tune target is Bedrock
CMI, which serves bf16. A laptop-quant baseline would under-report
the base model's capability and make the fine-tune look artificially
better. Bedrock model ids are also versioned and stable, so
`report.md` numbers stay reproducible — an HF GGUF digest is not, in
practice (the repo can be re-quantized or renamed).

What lands in `--out-dir`:

- `predictions.jsonl` — `{row_index, pred, raw, [error]}` per line, one
  row per gold row. **Committed to git** so future runs can diff
  against it.
- `_row_ids.txt` — integer row_index sidecar for resume. Gitignored.
- `_log.jsonl` — per-row usage + timing + error diagnostics.
  Gitignored.

Resume is on by default: kill the run with Ctrl-C and re-invoke the
same command; rows already present in the sidecar are skipped. Pass
`--no-resume` to force a full re-run.

Runner options:

- `--runner {bedrock-haiku,bedrock-llama}` — picks the provider.
- `--region` / `--model-id` override the runner's defaults (useful
  for pinning an inference profile like
  `us.meta.llama3-1-8b-instruct-v1:0` when an ON_DEMAND quota is
  saturated).
- `--max-parse-retries` controls how many times a JSON-parse failure
  gets a correction prompt retry. Default 1; set 0 to measure the
  model's first-pass parse rate honestly.
- **No schema-validation retry.** The scorer's `schema_rate` metric
  catches that — retrying would hide regressions that a production
  inference stack would still have to live with.

### 3. Score a run

Read the `predictions.jsonl` produced by step 2 (or any other runner)
and emit the scorecard.

```bash
lp-benchmark score \
    --gold benchmarks/test_set.jsonl \
    --predictions benchmarks/runs/haiku-4.5-teacher/predictions.jsonl \
    --out-dir benchmarks/runs/haiku-4.5-teacher \
    --name "haiku-4.5 (teacher)"
```

Output: a markdown scorecard on stdout plus `report.md` + `report.json`
in the out dir. Both are committed — diffing two checked-in `report.md`
files is the quickest way to answer *"did training make field X better
or worse?"*.

### 4. Label new listings

The repo is the authoritative label store. `data/labelled/*.json` is
the ground truth, the HF hub is a mirror rebuilt on demand. The
pipeline that goes from "a JSON file of pre-classified listings" to "a
labelled batch on disk" lives in `src/listing_parser/labelling/` and
is exposed via `scripts/label_listings.py`.

```bash
# Input: a JSON file shaped as either
#   {"listing": [{"listing_type": "Rent", "description": "..."}, ...]}
# or a bare list with the same row shape.

AWS_PROFILE=XXXXXXX python scripts/label_listings.py \
    --input data/listing_202604261415.json \
    --concurrency 8

# Output (matches HF parquet column-for-column):
#   data/labelled/202604261415.json  — array of {description, listing_type, output}
#   data/labelled/_index.json        — consolidated, dedup'd view + per-type histogram
#   data/labelled/_row_ids/*.txt     — resume sidecar (gitignored)
#   data/labelled/_log/*.jsonl       — per-attempt diagnostics (gitignored)
```

The CLI prints a live progress line with rate and ETA, calls out the
first cache hit/miss of a run, and surfaces retries and throttles inline
so a stuck run never looks silent. Pipe stderr to a file and the
overwrite-with-`\r` is automatically replaced by newline-append, so
`2>run.log` stays readable.

On-disk / HF shape notes:

- `description` and `listing_type` are the raw inputs the labeller
  received; `output` is the label (fenced JSON the cleaner already
  validated). Fine-tune trainers build the user message at load time
  via `listing_parser.prompting.build_user_message(description,
  listing_type)` so the prompt template isn't baked into the dataset.
- `listing_type` is a first-class column (not embedded in `output`) so
  you can filter / stratify without re-parsing the JSON.

What the pipeline does per row:

1. Build a system message from `prompt.md` + machine-rendered closed-
   vocabulary reinforcement (from `schema.py`) + few-shot examples
   (from `examples.json`). Bedrock caches the whole thing with a
   1-hour TTL, so after the first `--concurrency` rows the system
   prompt is free on every subsequent call.
2. Call Haiku 4.5 via Bedrock's Converse API (defaults to the
   `global.anthropic.claude-haiku-4-5-20251001-v1:0` inference
   profile in `eu-west-2`; override via `--model-id` / `--region`).
3. Parse the fenced JSON, run `listing_parser.cleaning.clean_extraction`
   (rename `amenities_indoor` typos, drop unknown sub-keys, strip
   out-of-vocab amenities, drop ungrounded evidence quotes), then
   validate against the Pydantic schema.
4. If parsing or validation fails, retry up to twice with the raw
   output + validator error threaded back in as a correction prompt.
   After that, drop the row and log it.

Resume semantics are automatic: the row_id is `sha1(description)[:16]`,
and any row already present in the sidecar file is skipped on re-run.
Kill the job with Ctrl-C and pick up where you left off.

### 5. Mirror labelled data to Hugging Face

Once a batch lands in `data/labelled/`, push the consolidated index to
the HF hub so Unsloth notebooks (or whoever else) can `load_dataset`:

```bash
python scripts/push_labels_to_hf.py
# uses data/labelled/_index.json by default
# writes a single `data/batch_00000.parquet` to the HF repo in one
# atomic commit, refreshes metadata.json, deletes old batches.
```

Requires `HF_TOKEN` in `.env` with write access.

### 6. Migrate pre-existing HF rows (one-off)

Before the labelling pipeline existed, teacher-labelled rows were
written directly to HF without cleanup. `scripts/clean_hf_labels.py`
is a one-off migration that pulls those rows, runs them through
`listing_parser.cleaning.clean_extraction`, and pushes the repaired
set back to HF. After the pipeline above owns all future writes,
running this script should be a no-op — use it only if you ever
tighten a cleaning rule and need to re-clean the hub.

```bash
python scripts/clean_hf_labels.py --out /tmp/cleaned.jsonl   # dry-run
python scripts/clean_hf_labels.py --push                     # upload
```

The script hard-fails if any row in `benchmarks/test_set.jsonl` would
be **dropped** (that would silently change which rows
`test_set.py --seed 42` reproduces) and warns if any test-set row
would be **content-repaired** (the gold file embeds its own parsed
copy, so past reports stay valid — the HF source just drifts from the
gold until you refresh it on purpose).

## File formats

### `benchmarks/test_set.jsonl` (gold)

One JSON object per line:

```json
{
  "row_index": 42,
  "description": "A beautifully presented two-bedroom...",
  "output": { "property_type": "Flat", "beds": 2, "...": "..." },
  "listing_type": "Rent"
}
```

- `row_index` — stable integer, the join key with predictions
- `description` — raw listing text the model sees
- `output` — the gold extraction (already-parsed JSON, no fences)
- `listing_type` — one of `Rent`, `Sale`, `Room`, inferred from which
  sub-block the output populates

### `benchmarks/runs/*/predictions.jsonl`

One JSON object per line, produced by a runner:

```json
{ "row_index": 42, "pred": { "property_type": "Flat", "...": "..." }, "raw": "..." }
```

- `pred` — the already-parsed JSON (runner strips fences / trailing prose)
- `raw` — optional, the model's untouched output; useful when `pred` is
  `null` because parsing failed

`row_index` is the join key. Predictions **do not** need to be in the
same order as gold — some providers (batch APIs) reorder results.

### `prompt.md`

The canonical system prompt handed to the model. Contains the full
output schema with every enum spelled out. This is the source of truth:
`schema.py` mirrors its vocabularies, and any change to the prompt must
be mirrored there. Hand-sync; the enum lists are short enough that
codegen isn't worth the complexity.

## Schema at a glance

`src/listing_parser/schema.py` defines:

- **Controlled vocabularies** as `tuple[str, ...]` constants:
  `PROPERTY_TYPES`, `AMENITIES_INTERIOR`, `AMENITIES_OUTDOOR`,
  `AMENITIES_FACILITIES`, `AMENITIES_APPLIANCES`, `PARKING_TYPES`,
  `PURCHASE_SCHEMES`, `FURNISH_TYPES`, `LET_TYPES`, `TENURE_TYPES`,
  `ROOM_TYPES`, `BILLS_INCLUDED`, `COMPASS_POINTS`.
- **Pydantic model** `Extraction` — `extra="forbid"` at every level so
  hallucinated keys fail validation rather than sneak through.
- **`FIELD_KINDS` / `NESTED_FIELD_KINDS`** — maps every field path to
  one of `scalar_enum | scalar_int | scalar_float | scalar_bool |
  list_set | list_ordered | nested_block | list_dict | string |
  evidence_map`. The scorer uses this to pick a comparator per field.

### Key design choices

1. **Every top-level field is optional.** The prompt instructs the
   model to *omit* keys rather than emit `null` when a fact isn't
   mentioned — omission carries real information ("silent on pets"),
   distinct from explicit negation ("pets_allowed: false").
2. **Booleans are scored three-way.** `present-true / present-false /
   absent`. A model that emits `false` for unmentioned flags is
   penalised as a false-positive hallucination, not rewarded for
   "matching absence".
3. **Evidence grounding is mandatory.** Every extraction must include a
   verbatim quote from the description in an `evidence` map, keyed by
   dotted-path. The scorer checks whether each quote (after whitespace
   + unicode normalisation) is a substring of the description. This is
   the cheapest hallucination detector you can build.

## What the scorer measures

| Metric | What it catches |
|---|---|
| `parse_rate` | Model outputs invalid JSON. Trailing prose and `` ```json `` fences are tolerated before parsing, so a low rate means the model genuinely broke. |
| `schema_rate` | JSON parses but violates the Pydantic schema — usually extra keys (e.g. Haiku inventing `sale.rental_income_monthly`) or wrong types. Fine-tunes regress here first. |
| `macro_accuracy` | Unweighted mean of per-field accuracy. A micro-average is dominated by trivially-easy fields like `beds`; we want rare fields to count equally. |
| **Per-field** `precision / recall / F1` | Field-presence metrics — "did the model correctly decide to emit this key at all". |
| **Per-field** `set_f1` (list fields) | Member-level F1 for list-valued fields like `amenities_interior`. Complements field-presence F1. |
| `vocab!` | Predictions outside the controlled vocabulary (e.g. `"Electric heating"` when only `"Gas heating"` exists). |
| `case!` | Predictions that differ from gold **only** in case. Separates typographic regression from semantic error. |
| `evidence_grounding_rate` | Fraction of evidence quotes that are actually substrings of the source description. Lies surface here. |

Latency, cost, and per-example spot-check notes are the **runner's**
job — the scorer reads only JSONL files.

## Design decisions worth knowing about

- **`window_orientation`** is stored as `"N,E,S"` in the schema and
  compared as a comma-split set by the scorer. Order doesn't matter.
- **`floors_occupied`** is scored order-insensitively even though the
  schema prints it in order. Descriptions often hint at floors
  non-linearly.
- **`nearby_mentions`** list-of-dicts are matched by `name` only, not
  by `walk_minutes`. A wrong walk-time doesn't count as a miss — most
  descriptions under-specify that number anyway.
- **Float tolerance** is relative 5% (e.g. `total_floor_area_sqm` 137.0
  vs 137.3 passes). Override per-field in `scorer.py` if needed.
- **`deposit_amount` midpoint convention.** When a description gives a
  range like "£70–£80", labellers should pick the midpoint. Codify this
  in the prompt if you want the fine-tuned model to do the same.
- **Studios have `beds: 0`.** Document this convention in `prompt.md`
  if you want to standardise it across labellers.

## When to regenerate the test set

**Almost never.** The whole point of the frozen set is that it anchors
every benchmark number in the history to a consistent target.

Valid reasons to regenerate (with `--force`):

- The source labelled dataset grew by more than ~4x and the old set no
  longer represents the distribution.
- The schema changed in a way that changes what "correct" means for
  old rows (e.g. a new required field was added).

Do **not** regenerate to "make the numbers look better". That destroys
the comparability of every historical `report.md` in `benchmarks/runs/`.

## Testing

```bash
pytest -v
```

All 24 tests run in under a second and cover the scorer's edge cases
explicitly:

- omit-vs-explicit-false for booleans
- evidence grounding with whitespace/unicode normalisation
- vocab violations vs case mismatches
- set-F1 for list fields
- list-dict matching by identifying field only
- gold-vs-gold scoring perfectly
- JSON fence stripping and listing-type inference

If you touch `scorer.py` or `schema.py`, add the corresponding test
before merging.

## What this repo is **not**

- Not a scraper. Listing descriptions come from an external pipeline;
  this repo only sees the text.
- Not a training harness. Training happens in an Unsloth notebook /
  RunPod job; artefacts land back here only as `predictions.jsonl`.
- Not a deployment. Bedrock Custom Model Import is an external
  one-off step and lives in the consuming application's infra code.
- Runners for local student models (Ollama) and the merged fine-tune
  are not yet implemented — `BedrockHaikuRunner` (teacher) and
  `BedrockLlamaRunner` (student base model) are the providers wired
  in today. Adding a new runner is a matter of implementing
  `listing_parser.runners.base.Runner` (two attributes, one async
  method) and registering it in the CLI's runner lookup.
