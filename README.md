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

| Stage | Artefact |
|---|---|
| Prompt + schema design | `prompt.md`, `examples.json` (mirrored into `src/listing_parser/_assets/`) |
| Teacher labelling pipeline | `src/listing_parser/labelling/` |
| Frozen test set + scorer | `benchmarks/test_set.jsonl`, `src/listing_parser/benchmarks/` |
| Teacher baseline | `benchmarks/runs/haiku-4.5-teacher/report.md` |
| Student base-model baseline | `benchmarks/runs/llama-3.1-8b-base/report.md` |
| Fine-tuning (Unsloth QLoRA on Colab) | `Listing_Parser_Fine_Tune_Unsloth.ipynb` |
| Student fine-tune served via Bedrock CMI | `benchmarks/runs/llama-3.1-8b-ft-v1-*-bedrock/report.md` |

Current scorecard across all committed runs:

| Run | `parse_rate` | `schema_rate` | `macro_accuracy` | `evidence_grounding_rate` |
|---|---:|---:|---:|---:|
| `haiku-4.5-teacher` | 100.0% | 98.3% | 99.3% | 97.0% |
| `llama-3.1-8b-base` | 100.0% | 65.0% | 81.7% | 82.4% |
| `llama-3.1-8b-ft-v1-quick-colab` (in-notebook eval) | 100.0% | 100.0% | 96.5% | 99.6% |
| `llama-3.1-8b-ft-v1-quick-bedrock` (CMI-served) | 100.0% | 90.0% | 92.4% | 99.4% |
| `llama-3.1-8b-ft-v1-full-bedrock` (CMI-served) | 100.0% | 95.0% | 92.2% | 96.7% |

Regenerate any of these with `lp-benchmark run` + `lp-benchmark score` —
see "Typical workflows" below.

## Repo layout

```
listing-parser/
├── Listing_Parser_Fine_Tune_Unsloth.ipynb   Colab notebook that produces
│                                            the merged fine-tune (Unsloth
│                                            QLoRA on A100). See "Fine-tuning"
│                                            section below.
├── prompt.md                          System prompt + JSON schema (source of truth;
│                                      mirrored into src/listing_parser/_assets/ so
│                                      the pip-installed package works without the
│                                      repo checkout, e.g. from a Colab notebook)
├── examples.json                      Worked examples used as few-shot + docs
├── data/                              Listings in (JSON) and labelled rows out
│   ├── listing_<stamp>.csv               Listing descriptions (single column)
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
│           ├── predictions.jsonl            Produced by a runner (committed)
│           ├── report.md                    Human-readable scorecard (committed)
│           ├── report.json                  Raw numbers for diffing runs (committed)
│           ├── _row_ids.txt                 Resume sidecar (gitignored)
│           └── _log.jsonl                   Per-row diagnostics (gitignored)
├── src/listing_parser/
│   ├── _assets/                          prompt.md + examples.json, packaged
│   │                                     with the wheel so pip-installed
│   │                                     consumers don't need the repo.
│   ├── schema.py                         Enums + Pydantic model, mirrors prompt.md
│   ├── cleaning.py                       Pure repair rules + parse helpers; single
│   │                                     source of truth for "what valid labels look
│   │                                     like" — used by the labeller AND the migration
│   ├── prompting.py                      Assemble system + user messages; stitches
│   │                                     prompt.md, schema.py, and examples.json
│   ├── labelling/                        Bedrock wire protocols + teacher pipeline
│   │   ├── sources.py                       Read data/listing_*.json -> ListingInput
│   │   ├── teacher.py                       Converse API client (cachePoint support)
│   │   ├── bedrock_invoke.py                InvokeModel client for CMI-imported models
│   │   ├── pipeline.py                      Per-row loop: prompt -> teacher -> clean
│   │   │                                    -> validate -> retry -> emit
│   │   └── sinks.py                         Append-only writer + index regeneration
│   ├── runners/                          Gold -> predictions (for benchmarking)
│   │   ├── base.py                          Runner Protocol + RunnerResult
│   │   ├── _output.py                       Shared parse-and-retry (one retry on
│   │   │                                    bad JSON; no schema retry)
│   │   ├── bedrock.py                       BedrockHaikuRunner (teacher, Converse)
│   │   │                                    BedrockLlamaRunner (base, Converse)
│   │   │                                    BedrockFineTuneRunner (CMI, InvokeModel)
│   │   └── pipeline.py                      Gold JSONL -> predictions.jsonl driver
│   │                                         with resume + per-run sidecar/log
│   └── benchmarks/
│       ├── test_set.py                     Builds benchmarks/test_set.jsonl from HF
│       ├── scorer.py                       Per-field metrics (pure, no I/O)
│       └── cli.py                          `lp-benchmark` entry point (run/score/smoke)
├── tests/                             pytest suite covering cleaning, prompting,
│                                      scorer edge cases, runner dispatch, and the
│                                      Llama-3.1 chat template bytes.
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
pytest -q                                              # full test suite
lp-benchmark smoke --gold benchmarks/test_set.jsonl    # scorer gold-vs-gold
```

`lp-benchmark smoke` scores the gold test set against itself, so
`parse_rate`, `schema_rate`, and `macro_accuracy` should all be 1.0.
Anything less is a bug in the scorer, not the data.

### Credentials you'll need

Actual workflows (labelling, benchmarking, fine-tuning) need external
services. The install alone needs nothing; credentials only come into
play when you run one of the workflow commands below.

| Service | What for | Where to set |
|---|---|---|
| AWS (Bedrock + S3) | Teacher Haiku runs, base-Llama runs, CMI imports, fine-tune deployment | `AWS_PROFILE` env var or `~/.aws/credentials` |
| HF Hub | Push/pull labelled dataset, push fine-tuned weights | `HF_TOKEN` in `.env` (repo root) |
| W&B | Training-metric logging during fine-tune | `WANDB_API_KEY` (Colab secret or env var) |

A `.env` file is loaded by the CLI tools that need it (`label_listings.py`,
`push_labels_to_hf.py`, `lp-benchmark run`); the file is gitignored.

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

Three runners ship today:

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
- `bedrock-ft` — a Custom Model Import (CMI) of a fine-tuned Llama
  3.1 8B. Requires `--model-id <arn>` because CMI ARNs are account-
  specific. Defaults: region `eu-central-1` (CMI isn't available in
  `eu-west-2` at time of writing), temperature 0.0. Uses the
  InvokeModel API rather than Converse — CMI imports don't support
  Converse, so this runner dispatches to `labelling/bedrock_invoke.py`
  internally. See the "Fine-tuning" + "Deploying to Bedrock" sections
  below for how to produce the ARN.

Example invocations:

```bash
# Teacher baseline — Haiku 4.5, eu-west-2, uses prompt caching.
AWS_PROFILE=XXXXXXX lp-benchmark run \
    --runner bedrock-haiku \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/haiku-4.5-teacher \
    --concurrency 4

# Student base-model baseline — Llama 3.1 8B Instruct on Bedrock,
# us-west-2, greedy decoding, no prompt caching.
AWS_PROFILE=XXXXXXX lp-benchmark run \
    --runner bedrock-llama \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/llama-3.1-8b-base \
    --concurrency 4

# Fine-tuned student via Bedrock CMI, eu-central-1. The ARN comes
# from the model import step — see "Deploying to Bedrock" below.
AWS_PROFILE=XXXXXXX lp-benchmark run \
    --runner bedrock-ft \
    --model-id "arn:aws:bedrock:eu-central-1:ACCOUNT:imported-model/ABC123" \
    --name-slug "llama-3.1-8b-ft-v1-quick-bedrock" \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/llama-3.1-8b-ft-v1-quick-bedrock \
    --concurrency 1   # CMI default quota is ~1 req/s
```

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

- `--runner {bedrock-haiku,bedrock-llama,bedrock-ft}` — picks the provider.
- `--model-id` — required for `bedrock-ft` (the CMI ARN); override the
  default for `bedrock-haiku`/`bedrock-llama` e.g. to pin an inference
  profile like `us.meta.llama3-1-8b-instruct-v1:0` when an ON_DEMAND
  quota is saturated.
- `--region` overrides the runner's default. CMI imports live in
  whichever region you ran the import job in (we default to
  `eu-central-1`).
- `--max-parse-retries` controls how many times a JSON-parse failure
  gets a correction prompt retry. Default 1; set 0 to measure the
  model's first-pass parse rate honestly.
- `--concurrency` — cap on in-flight requests. Start low for CMI
  (default 1) and for Llama base-model (ON_DEMAND quotas throttle
  above 4); Haiku tolerates 4–8 comfortably thanks to prompt caching.
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

### 7. Fine-tune the student model (Colab + Unsloth)

The fine-tune itself runs in `Listing_Parser_Fine_Tune_Unsloth.ipynb`,
a Colab notebook you open via the "Open in Colab" badge at the top.
Training happens on rented A100 GPU time; the notebook produces a
merged bf16 safetensors model that uploads to Hugging Face under
`standrey/listing-parser-llama31-8b-ft-v1[-full]`.

Why a notebook rather than a script in this repo:

- Unsloth needs a CUDA-enabled GPU and a specific torch/CUDA/triton
  stack; running it from a Mac dev environment isn't viable.
- Colab is the cheapest A100 you can rent by the hour (~$10/mo for
  Pro; this fine-tune takes ~25 min on A100, so one fine-tune run
  costs a few dollars).
- The weights output (~16GB) doesn't belong in git. HF is the natural
  store.

What the notebook does, cell-by-cell:

1. **Install** Unsloth + TRL + transformers pinned against each
   other, plus `listing-parser` from this GitHub repo so training
   uses the exact same `build_system_prompt` / `build_user_message`
   the teacher did. Getting the prompt to diff between labelling and
   training would silently invalidate everything.
2. **Load** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` at 4-bit
   with `max_seq_length=8192`. Token-length diagnostics showed p95
   ~6,300 for our data; 8192 gives comfortable headroom.
3. **Patch the Llama-3.1 chat template** with `{% generation %}`
   markers. This is load-bearing: neither Unsloth's bundled template
   nor Meta's own template has these markers, and without them
   `tokenizer.apply_chat_template(..., return_assistant_tokens_mask
   =True)` returns an all-zeros mask. The loss-masking collator
   silently computes loss on nothing, training becomes a no-op, and
   the only signal is an eerily-low stuck loss. Cell 14 includes a
   byte-equality check against `NousResearch/Meta-Llama-3.1-8B-Instruct`
   so this patch can't silently drift from the reference tokenizer.
4. **Attach LoRA adapters** at rank 32, α 32, all 7 projection
   matrices. Narrow structured-output tasks benefit from more
   adapter capacity than chat-style fine-tunes.
5. **Load + tokenize the HF dataset**, apply the patched chat
   template, and assert that `assistant_masks` has non-zero entries.
   The assertion fails fast if the template patch didn't land — it's
   the single cheapest guard against the "loss stuck at 0" failure
   mode we hit during bringup.
6. **`AssistantMaskCollator`** (cell 15) — a hand-rolled collator
   that pads sequences, builds `labels` from the assistant mask (set
   to -100 outside the assistant span), and hands the batch to TRL's
   `SFTTrainer`. We do this ourselves rather than relying on TRL's
   `assistant_only_loss=True` because Unsloth's patched trainer
   rejects it on pre-tokenized datasets.
7. **Train** in two phases: `PHASE="quick"` (300 steps, ~10 min on
   A100) to confirm the pipeline works, then `PHASE="full"`
   (3 epochs, ~25 min on A100). Loss starts around 1.8–2.2 and
   descends to 0.2–0.4. W&B logging is wired in (cell 18).
8. **Save merged bf16 weights** locally + push to HF. The merged
   model is what Bedrock CMI ingests — Bedrock doesn't accept LoRA
   adapters, it wants a standalone HF-format directory.
9. **In-notebook eval** — download the committed gold set, run 60
   predictions through the merged model, write a `predictions.jsonl`,
   download it. This gives you an immediate scorecard without waiting
   for Bedrock CMI (which takes 30–90 min to import).

The notebook's `predictions.jsonl` lands at
`benchmarks/runs/llama-3.1-8b-ft-v1-*-colab/predictions.jsonl` when
you score it locally with `lp-benchmark score`.

Cost and time expectations for one full fine-tune run:

| Component | Time | Cost |
|---|---|---|
| Colab Pro A100 (fine-tune + eval) | ~1 hour | ~$2–4 |
| HF bandwidth for merged weights push | ~5 min | free |

### 8. Deploy a fine-tune to Bedrock Custom Model Import

The fine-tune needs to be served to be useful. Bedrock CMI is the
cheapest option for bursty / occasional-inference workloads (scale-
to-zero, billed per 5-minute window of activity). For sustained
traffic, consider SageMaker; the break-even is roughly 5 hours/day
of active invocation. Economics live in
`src/listing_parser/runners/CLAUDE.md`.

Steps, assuming the fine-tune notebook has already pushed weights
to `standrey/listing-parser-llama31-8b-ft-v1-*`:

```bash
# 1. Configure region / bucket / IDs once per fine-tune version.
export REGION=eu-central-1      # Frankfurt — CMI isn't in London (eu-west-2) yet
export BUCKET=listing-parser-cmi
export ACCOUNT_ID=$(AWS_PROFILE=XXXXXXX aws sts get-caller-identity \
    --query Account --output text)

# 2. Download weights from HF, upload to S3. Easiest path is from
#    the Colab notebook itself — see the "S3 upload" cells in
#    Listing_Parser_Fine_Tune_Unsloth.ipynb. From a local machine,
#    you can also use `huggingface-cli download` + `aws s3 sync`.

# 3. Create the IAM role Bedrock CMI assumes (one-time).
#
# Trust policy (cmi-trust.json) — allow bedrock service to assume the
# role, scoped to your account + the model-import-job ARN pattern:
#   {"Version":"2012-10-17","Statement":[{
#     "Effect":"Allow",
#     "Principal":{"Service":"bedrock.amazonaws.com"},
#     "Action":"sts:AssumeRole",
#     "Condition":{
#       "StringEquals":{"aws:SourceAccount":"ACCOUNT_ID"},
#       "ArnLike":{"aws:SourceArn":"arn:aws:bedrock:REGION:ACCOUNT_ID:model-import-job/*"}
#     }
#   }]}
#
# S3 read policy (cmi-s3.json) — GetObject on bucket/*, ListBucket on bucket:
#   {"Version":"2012-10-17","Statement":[
#     {"Effect":"Allow","Action":["s3:GetObject"],"Resource":"arn:aws:s3:::BUCKET/*"},
#     {"Effect":"Allow","Action":["s3:ListBucket"],"Resource":"arn:aws:s3:::BUCKET"}
#   ]}
AWS_PROFILE=XXXXXXX aws iam create-role \
    --role-name BedrockCMIListingParser \
    --assume-role-policy-document file:///tmp/cmi-trust.json
AWS_PROFILE=XXXXXXX aws iam put-role-policy \
    --role-name BedrockCMIListingParser --policy-name CMIReadS3 \
    --policy-document file:///tmp/cmi-s3.json

# 4. Launch the import. Takes 30-90 min.
AWS_PROFILE=XXXXXXX aws bedrock create-model-import-job \
    --job-name "listing-parser-ft-v1-$(date +%s)" \
    --imported-model-name "listing-parser-ft-v1" \
    --role-arn "arn:aws:iam::${ACCOUNT_ID}:role/BedrockCMIListingParser" \
    --model-data-source "s3DataSource={s3Uri=s3://${BUCKET}/llama31-8b-ft-v1/}" \
    --region ${REGION}

# 5. Poll until `Completed`.
AWS_PROFILE=XXXXXXX aws bedrock get-model-import-job \
    --job-identifier "listing-parser-ft-v1-..." \
    --region ${REGION} --query 'status'

# 6. Grab the ARN. This is what you pass to bedrock-ft.
ARN=$(AWS_PROFILE=XXXXXXX aws bedrock list-imported-models \
    --region ${REGION} \
    --query 'modelSummaries[?modelName==`listing-parser-ft-v1`].modelArn' \
    --output text)

# 7. Smoke test via InvokeModel (Converse doesn't support CMI).
cat > /tmp/invoke-smoke.json <<EOF
{"prompt":"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nok<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n","max_gen_len":10,"temperature":0.0}
EOF
AWS_PROFILE=XXXXXXX aws bedrock-runtime invoke-model \
    --model-id "$ARN" --region ${REGION} \
    --body file:///tmp/invoke-smoke.json \
    --cli-binary-format raw-in-base64-out /tmp/smoke-out.json
cat /tmp/smoke-out.json

# 8. Score the CMI-served model.
AWS_PROFILE=XXXXXXX lp-benchmark run \
    --runner bedrock-ft \
    --model-id "$ARN" \
    --name-slug "llama-3.1-8b-ft-v1-bedrock" \
    --gold benchmarks/test_set.jsonl \
    --out-dir benchmarks/runs/llama-3.1-8b-ft-v1-bedrock \
    --concurrency 1 \
    --region ${REGION}

lp-benchmark score \
    --gold benchmarks/test_set.jsonl \
    --predictions benchmarks/runs/llama-3.1-8b-ft-v1-bedrock/predictions.jsonl \
    --out-dir benchmarks/runs/llama-3.1-8b-ft-v1-bedrock \
    --name "llama-3.1-8b-ft-v1 (Bedrock)"
```

Gotchas worth knowing:

- **Cold start.** The first `InvokeModel` call after >5 min idle
  takes 60–120s. Our `BedrockFineTuneRunner` treats
  `ModelNotReadyException` as a retryable throttle, so you'll see
  backoff retries on the first row of a run — not a failure.
- **Quota.** CMI's default on-demand quota is ~1 req/s. Starting
  `--concurrency 2` on a new imported model will immediately trigger
  sustained throttling. Stay at 1 unless you have a quota increase.
- **Region.** CMI is NOT available in `eu-west-2` (London) at time
  of writing. `eu-central-1` (Frankfurt) is the nearest EU region
  with CMI support.
- **Colab vs Bedrock inference.** Our scoreboard shows the same
  weights scoring 96.5% macro in Colab and 92.4% via Bedrock CMI.
  That 4-point gap isn't explained yet — candidates are bf16
  rounding differences, prompt tokenization drift between the
  patched training template and Bedrock's serving tokenizer, or
  sampling differences at temperature 0. Open investigation.

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
pytest -q
```

All tests run in under a second. Coverage includes:

- Scorer edge cases: omit-vs-explicit-false for booleans; evidence
  grounding with whitespace/unicode normalisation; vocab violations
  vs case mismatches; set-F1 for list fields; list-dict matching by
  identifying field only; gold-vs-gold scoring perfectly.
- Parse-and-retry: JSON-fence stripping, correction-prompt path,
  max-retry exhaustion, non-retryable runtime errors.
- Runner registry + dispatch: CLI flags wire correctly to concrete
  runners; `bedrock-ft` uses `_USES_INVOKE_MODEL=True` (CMI rejects
  Converse); Haiku/Llama stay on Converse (so prompt caching
  continues to work on Anthropic models).
- Wire-level Bedrock shape: Llama-3.1 chat template rendering is
  byte-identical between training and CMI serving; `ModelNotReadyException`
  is converted to a retryable throttle.

If you touch `scorer.py` or `schema.py`, add the corresponding test
before merging. Same rule for any `runners/` or `labelling/` change
that touches the Bedrock wire protocol.

## What this repo is **not**

- Not a scraper. Listing descriptions come from an external pipeline;
  this repo only sees the text.
- Not a training harness. Training happens in a Colab notebook
  (`Listing_Parser_Fine_Tune_Unsloth.ipynb`); the repo holds the
  notebook but Colab-Pro compute isn't bundled. Training artefacts
  flow back in via HF + Bedrock CMI.
- Not a production serving stack. `BedrockFineTuneRunner` exercises
  a CMI-served fine-tune well enough to score it, but a real product
  would layer caching, rate limiting, and retry-with-backoff on top
  of the raw InvokeModel call.
- Not an Ollama path. We deliberately avoid laptop-quant baselines
  because Q4_K_M routinely drops 1–3 points on structured-extraction
  tasks vs the bf16 Bedrock serves, and we'd rather not compare the
  fine-tune to a confounded baseline. Adding an Ollama runner is
  straightforward (implement `Runner`, register in CLI) if you want
  that number anyway.
