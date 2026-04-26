# CLAUDE.md — benchmarks package

Context for AI assistants editing `src/listing_parser/benchmarks/`.
This is subordinate to the repo-level `CLAUDE.md`; read that first.

## Module responsibilities

| File | Responsibility | Touches I/O? |
|---|---|---|
| `test_set.py` | Build `benchmarks/test_set.jsonl` from a labelled HF dataset (one-shot, frozen afterwards). | Yes: HF Hub + disk. |
| `scorer.py` | Pure scoring logic. `score_run(gold, pred) -> AggregateReport`. | File read only. |
| `cli.py` | `lp-benchmark {run,score,smoke}` — rendering + glue. `run` dispatches to a runner; `score` and `smoke` are offline. | File read/write; `run` additionally imports runners (which touch the network). |

## Hard rules

- **`scorer.py` does not do network I/O.** No HTTP, no boto, no
  datasets library. If something tempts you to add it, it belongs in
  a runner (a future module) or in `cli.py`, not here.
- **`scorer.py` does not raise on bad predictions.** A missing key is
  not an exception; unparseable JSON surfaces as `parse_ok=False`. A
  scoring run over 500 predictions must complete even if 50 of them
  are garbage.
- **The `FieldResult` dataclass is the atomic unit.** Every comparator
  returns one. Aggregation derives everything (precision, recall, F1,
  accuracy) from the list of `FieldResult` — don't shortcut by
  computing aggregates inside comparators.
- **Field paths use dotted notation.** `rent.bills_included`, not
  `rent/bills_included` or `rent__bills_included`. The evidence map
  uses the same convention.

## Adding a new comparator kind

1. Add the literal to `FieldKind` in `schema.py`.
2. Write a `_cmp_<kind>(path, gold, pred, ...) -> FieldResult` in
   `scorer.py`. Follow the signature of existing comparators;
   return a `FieldResult` whose `correct / gold_present /
   pred_present` are set correctly. Don't set `tp/fp/fn` — they're
   computed properties.
3. Wire it into `_score_field`.
4. Add unit tests covering:
   - perfect match (correct=True)
   - partial match (if applicable; e.g. list P/R)
   - absent-on-both (tp-like outcome expected in aggregation)
   - hallucination (pred present, gold absent)
   - miss (gold present, pred absent)

## How aggregation works

`aggregate(scores) -> AggregateReport` collects per-field stats from
every example's `field_results` list. For each field path:

- `tp / fp / fn / wrong / tn` counts come from `FieldResult` booleans.
- `precision = tp / (tp + fp + wrong)` — wrong-but-present predictions
  count against precision; they are not free because the value is
  non-null but incorrect.
- `recall = tp / (tp + fn + wrong)` — similarly, a wrong value
  doesn't credit recall just because a key was emitted.
- `f1` is the harmonic mean. Zero denominators return 0.0.
- `accuracy = (tp + tn) / n` — includes agreement on absence.
- `set_f1` is only populated for list-set fields (`precision` / `recall`
  are per-example, aggregated via `statistics.fmean`).

The **headline `macro_accuracy`** is the unweighted mean of per-field
accuracies. Intentionally dumb: every field counts equally, so a
regression on `room.current_gender` hurts the headline even though
that field is rarely present.

## Testing

Every comparator change needs tests in `tests/test_scorer.py`. The
"omit-vs-false" test (`test_scalar_bool_three_way_omit_vs_false`) and
the "gold-vs-gold is perfect" test
(`test_gold_vs_gold_is_perfect`) are the load-bearing cases. Don't
remove them when refactoring.

## Things that live elsewhere (on purpose)

- **Runners.** Any code that calls an LLM and produces
  `predictions.jsonl` — Ollama, Anthropic, Bedrock, vLLM, etc. —
  lives in `src/listing_parser/runners/`. The scorer stays agnostic
  to provider. `lp-benchmark run` drives runners; `lp-benchmark
  score` is pure and only reads JSONL.
- **Augmentation / data prep.** Teacher-model calls to generate
  additional labelled rows live in `scripts/`. They produce HF
  dataset uploads, not benchmark predictions.
- **Training.** Unsloth notebooks / RunPod scripts don't live in this
  repo at all. Training artefacts flow back in only as
  `predictions.jsonl`.
