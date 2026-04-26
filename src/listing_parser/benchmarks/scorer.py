"""Per-field scorer for listing-description extractions.

Compares `predictions.jsonl` against a frozen gold test set and emits a
structured report. Every metric is broken out per-field so regressions
are easy to localise.

Design notes that matter:

* "Omit, don't nullify" — the prompt instructs the model to omit keys
  when a fact is not mentioned. That means missing-vs-explicit-false is
  meaningful information. Scalar booleans are scored as a 3-way
  categorical (present-true / present-false / absent), so a model that
  hallucinates `false` for unmentioned fields is penalised.
* Evidence grounding — for every evidence quote, we check whether its
  normalised substring appears in the normalised description. Quotes
  that don't ground are the cheapest hallucination signal we have.
* Aggregate-vs-per-field — the headline score is a macro average over
  fields (each field weighted equally), because a micro average is
  dominated by easy fields like `beds` and hides regressions on
  low-frequency ones like `room.current_gender`.
* Vocabulary violations — an enum field emitting a value outside its
  controlled vocabulary is recorded separately from a wrong-but-valid
  value. "flat" vs "Flat" is a case regression; "Apartment" vs "Flat" is
  a semantic error. Both count as wrong, but for different reasons.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from listing_parser.schema import (AMENITY_GROUPS, COMPASS_POINTS, FIELD_KINDS,
                                   NESTED_FIELD_KINDS, Extraction, FieldKind,
                                   expected_vocab)

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    """Collapse whitespace and lowercase for evidence substring checks.

    Descriptions are full of \\n, \\r, NBSP, non-breaking hyphens etc. A
    strict `in` check would miss evidence that's obviously grounded.
    """
    return _WS_RE.sub(" ", s.replace("\xa0", " ").replace("\u2013", "-").lower()).strip()


def _compass_set(value: str | None) -> set[str]:
    if not value:
        return set()
    parts = {p.strip().upper() for p in value.split(",")}
    return {p for p in parts if p in COMPASS_POINTS}


# ---------------------------------------------------------------------------
# Per-field comparison primitives.
#
# Each comparator returns a FieldResult which records whether the field
# was correct, how it was wrong, and optional side-channel info (e.g.
# "value was outside the controlled vocabulary").
# ---------------------------------------------------------------------------


@dataclass
class FieldResult:
    """Outcome of comparing one (field_path, gold_value, pred_value) triple."""

    path: str
    kind: FieldKind
    gold_present: bool
    pred_present: bool
    correct: bool
    # For list_set fields: precision / recall over set members.
    precision: float | None = None
    recall: float | None = None
    # Flagged problems, for the report.
    vocab_violation: bool = False
    case_mismatch: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def tp(self) -> bool:
        """Both sides agree a value is present AND the values match."""
        return self.gold_present and self.pred_present and self.correct

    @property
    def fp(self) -> bool:
        """Model asserted a value the gold label omits (hallucinated)."""
        return not self.gold_present and self.pred_present

    @property
    def fn(self) -> bool:
        """Gold has a value the model missed."""
        return self.gold_present and not self.pred_present

    @property
    def wrong(self) -> bool:
        """Both sides have values but they disagree."""
        return self.gold_present and self.pred_present and not self.correct


def _cmp_scalar_exact(path: str, gold: Any, pred: Any, kind: FieldKind) -> FieldResult:
    gold_present = gold is not None
    pred_present = pred is not None
    if not gold_present or not pred_present:
        return FieldResult(path, kind, gold_present, pred_present, correct=False)

    vocab = expected_vocab(path)
    case_mismatch = False
    vocab_violation = False
    if isinstance(gold, str) and isinstance(pred, str) and vocab is not None:
        if pred not in vocab:
            vocab_violation = True
        if pred != gold and pred.lower() == gold.lower():
            case_mismatch = True

    return FieldResult(
        path,
        kind,
        gold_present,
        pred_present,
        correct=gold == pred,
        vocab_violation=vocab_violation,
        case_mismatch=case_mismatch,
    )


def _cmp_scalar_float(
    path: str, gold: Any, pred: Any, rel_tol: float = 0.05
) -> FieldResult:
    gp, pp = gold is not None, pred is not None
    if not gp or not pp:
        return FieldResult(path, "scalar_float", gp, pp, correct=False)
    try:
        g, p = float(gold), float(pred)
    except (TypeError, ValueError):
        return FieldResult(
            path, "scalar_float", gp, pp, correct=False, notes=["non-numeric pred"]
        )
    # Accept exact zero match; otherwise relative tolerance.
    correct = g == p or (g != 0 and abs(g - p) / abs(g) <= rel_tol)
    return FieldResult(path, "scalar_float", gp, pp, correct=correct)


def _cmp_list_set(
    path: str, gold: Any, pred: Any, *, use_compass: bool = False
) -> FieldResult:
    def normalise(v: Any) -> set[str]:
        if v is None:
            return set()
        if use_compass and isinstance(v, str):
            return _compass_set(v)
        if isinstance(v, str):
            v = [v]
        return {str(x).strip() for x in v if str(x).strip()}

    g_set = normalise(gold)
    p_set = normalise(pred)
    gp = bool(g_set) or gold is not None
    pp = bool(p_set) or pred is not None

    if not g_set and not p_set:
        return FieldResult(path, "list_set", gp, pp, correct=True, precision=1.0, recall=1.0)

    tp = len(g_set & p_set)
    precision = tp / len(p_set) if p_set else 0.0
    recall = tp / len(g_set) if g_set else 0.0
    correct = g_set == p_set

    vocab = expected_vocab(path)
    vocab_violation = False
    if vocab is not None and p_set - set(vocab):
        vocab_violation = True

    return FieldResult(
        path,
        "list_set",
        gp,
        pp,
        correct=correct,
        precision=precision,
        recall=recall,
        vocab_violation=vocab_violation,
    )


def _cmp_list_ordered(path: str, gold: Any, pred: Any) -> FieldResult:
    gp = gold is not None
    pp = pred is not None
    if not gp and not pp:
        return FieldResult(path, "list_ordered", gp, pp, correct=True)
    if gold is None or pred is None:
        return FieldResult(path, "list_ordered", gp, pp, correct=False)
    # floors_occupied: order-insensitive in practice even though the schema
    # prints them ordered. Compare as sets for correctness, report order as a note.
    g_set = set(gold) if isinstance(gold, list) else set()
    p_set = set(pred) if isinstance(pred, list) else set()
    return FieldResult(path, "list_ordered", gp, pp, correct=g_set == p_set)


def _cmp_list_dict(path: str, gold: Any, pred: Any, key: str) -> FieldResult:
    """Match list-of-dicts by the identifying field (name / destination).

    Two rows with the same `name` are treated as the same mention even if
    their walk_minutes differ. We surface the value mismatch as a note so
    the report can show it, but we don't mark it as a hard miss.
    """
    def keys_of(v: Any) -> set[str]:
        if not isinstance(v, list):
            return set()
        return {
            _norm_text(str(item.get(key, "")))
            for item in v
            if isinstance(item, dict) and item.get(key)
        }

    gk, pk = keys_of(gold), keys_of(pred)
    gp = gold is not None
    pp = pred is not None
    if not gk and not pk:
        return FieldResult(path, "list_dict", gp, pp, correct=True, precision=1.0, recall=1.0)
    tp = len(gk & pk)
    precision = tp / len(pk) if pk else 0.0
    recall = tp / len(gk) if gk else 0.0
    return FieldResult(
        path,
        "list_dict",
        gp,
        pp,
        correct=gk == pk,
        precision=precision,
        recall=recall,
    )


def _cmp_string(path: str, gold: Any, pred: Any) -> FieldResult:
    """Free-text strings — use normalised equality (case + whitespace)."""
    gp, pp = gold is not None, pred is not None
    if not gp or not pp:
        return FieldResult(path, "string", gp, pp, correct=False)
    return FieldResult(
        path, "string", gp, pp, correct=_norm_text(str(gold)) == _norm_text(str(pred))
    )


# ---------------------------------------------------------------------------
# Top-level scorer
# ---------------------------------------------------------------------------


@dataclass
class ExampleScore:
    """Per-example score. One entry per gold row."""

    index: int
    listing_type: str
    parse_ok: bool
    schema_ok: bool
    field_results: list[FieldResult]
    evidence_total: int = 0
    evidence_grounded: int = 0
    error: str | None = None


@dataclass
class AggregateReport:
    n: int
    n_parse_ok: int
    n_schema_ok: int
    by_field: dict[str, dict[str, float]]
    evidence_grounding_rate: float
    macro_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "n_parse_ok": self.n_parse_ok,
            "n_schema_ok": self.n_schema_ok,
            "parse_rate": self.n_parse_ok / self.n if self.n else 0.0,
            "schema_rate": self.n_schema_ok / self.n if self.n else 0.0,
            "macro_accuracy": self.macro_accuracy,
            "evidence_grounding_rate": self.evidence_grounding_rate,
            "by_field": self.by_field,
        }


def _score_evidence(
    evidence: dict[str, str] | None, description: str
) -> tuple[int, int]:
    """Return (total_quotes, grounded_quotes)."""
    if not isinstance(evidence, dict):
        return 0, 0
    desc_norm = _norm_text(description)
    total = grounded = 0
    for quote in evidence.values():
        if not isinstance(quote, str) or not quote.strip():
            continue
        total += 1
        if _norm_text(quote) in desc_norm:
            grounded += 1
    return total, grounded


def _score_field(
    path: str, kind: FieldKind, gold: Any, pred: Any
) -> FieldResult | None:
    if kind == "scalar_enum":
        return _cmp_scalar_exact(path, gold, pred, kind)
    if kind == "scalar_int":
        return _cmp_scalar_exact(path, gold, pred, kind)
    if kind == "scalar_bool":
        return _cmp_scalar_exact(path, gold, pred, kind)
    if kind == "scalar_float":
        return _cmp_scalar_float(path, gold, pred)
    if kind == "list_set":
        use_compass = path == "window_orientation"
        return _cmp_list_set(path, gold, pred, use_compass=use_compass)
    if kind == "list_ordered":
        return _cmp_list_ordered(path, gold, pred)
    if kind == "list_dict":
        # Choose the identifier field based on path. Both current list_dict
        # fields (nearby_mentions, commute_claims) identify on a single key.
        id_key = "destination" if path == "commute_claims" else "name"
        return _cmp_list_dict(path, gold, pred, id_key)
    if kind == "string":
        return _cmp_string(path, gold, pred)
    # Nested blocks and evidence are handled by the caller.
    return None


def score_one(
    gold: dict[str, Any], pred: dict[str, Any] | None, description: str, index: int
) -> ExampleScore:
    """Score a single prediction against its gold label.

    Returns a dataclass the report layer can consume without re-parsing.
    """
    listing_type = gold.get("listing_type") or ""

    if pred is None:
        return ExampleScore(
            index=index,
            listing_type=listing_type,
            parse_ok=False,
            schema_ok=False,
            field_results=[],
            error="prediction unparseable",
        )

    schema_ok = True
    try:
        Extraction.model_validate(pred)
    except Exception as e:
        schema_ok = False
        schema_err = str(e)[:200]
    else:
        schema_err = None

    gold_obj = gold.get("output", gold)

    results: list[FieldResult] = []
    for path, kind in FIELD_KINDS.items():
        if kind == "nested_block":
            g_block = gold_obj.get(path) or {}
            p_block = pred.get(path) or {}
            for sub_path, sub_kind in NESTED_FIELD_KINDS.items():
                if not sub_path.startswith(path + "."):
                    continue
                sub_field = sub_path.split(".", 1)[1]
                g_val = g_block.get(sub_field) if isinstance(g_block, dict) else None
                p_val = p_block.get(sub_field) if isinstance(p_block, dict) else None
                r = _score_field(sub_path, sub_kind, g_val, p_val)
                if r is not None:
                    results.append(r)
            continue

        if kind == "evidence_map":
            continue  # evidence grounding is tallied separately below

        r = _score_field(path, kind, gold_obj.get(path), pred.get(path))
        if r is not None:
            results.append(r)

    ev_total, ev_grounded = _score_evidence(pred.get("evidence"), description)

    return ExampleScore(
        index=index,
        listing_type=listing_type,
        parse_ok=True,
        schema_ok=schema_ok,
        field_results=results,
        evidence_total=ev_total,
        evidence_grounded=ev_grounded,
        error=schema_err,
    )


def aggregate(scores: list[ExampleScore]) -> AggregateReport:
    """Collapse per-example scores into a per-field table + headlines.

    For each field path, we compute:
      - coverage  : fraction of examples where gold has a value
      - precision : tp / (tp + fp)
      - recall    : tp / (tp + fn)
      - f1        : harmonic mean
      - accuracy  : (tp + tn) / n  — where tn means both sides agree on absence
      - set_f1    : only for list_set fields; member-level F1 averaged over
                    examples where at least one side has values
    """
    per_field: dict[str, list[FieldResult]] = {}
    for s in scores:
        for fr in s.field_results:
            per_field.setdefault(fr.path, []).append(fr)

    by_field: dict[str, dict[str, float]] = {}
    for path, frs in per_field.items():
        n = len(frs)
        tp = sum(1 for r in frs if r.tp)
        fp = sum(1 for r in frs if r.fp)
        fn = sum(1 for r in frs if r.fn)
        wrong = sum(1 for r in frs if r.wrong)
        tn = sum(1 for r in frs if not r.gold_present and not r.pred_present)

        precision = tp / (tp + fp + wrong) if (tp + fp + wrong) else 0.0
        recall = tp / (tp + fn + wrong) if (tp + fn + wrong) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / n if n else 0.0

        entry: dict[str, float] = {
            "n": float(n),
            "coverage": (tp + fn + wrong) / n if n else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "vocab_violations": float(sum(1 for r in frs if r.vocab_violation)),
            "case_mismatches": float(sum(1 for r in frs if r.case_mismatch)),
        }

        # For set-valued fields, report member-level F1 averaged across
        # examples where at least one side has values. Complements the
        # field-presence F1 above.
        set_frs = [r for r in frs if r.precision is not None and (r.gold_present or r.pred_present)]
        if set_frs:
            entry["set_f1"] = statistics.fmean(
                2 * r.precision * r.recall / (r.precision + r.recall)
                if (r.precision + r.recall)
                else 0.0
                for r in set_frs
            )

        by_field[path] = entry

    # Headline macro accuracy: unweighted mean of per-field accuracy. Each
    # field contributes equally — `beds` doesn't get to dominate the score
    # just because every listing has one.
    macro_acc = statistics.fmean(v["accuracy"] for v in by_field.values()) if by_field else 0.0

    ev_total = sum(s.evidence_total for s in scores)
    ev_grounded = sum(s.evidence_grounded for s in scores)
    grounding_rate = ev_grounded / ev_total if ev_total else 0.0

    return AggregateReport(
        n=len(scores),
        n_parse_ok=sum(1 for s in scores if s.parse_ok),
        n_schema_ok=sum(1 for s in scores if s.schema_ok),
        by_field=by_field,
        evidence_grounding_rate=grounding_rate,
        macro_accuracy=macro_acc,
    )


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_run(gold_path: Path, predictions_path: Path) -> AggregateReport:
    """Score a predictions file against a gold file.

    Both files are JSONL. The gold file is the output of `test_set.py`.
    The predictions file carries rows of the shape:
        {"row_index": N, "pred": <dict-or-null>, "raw": "<raw model output>"}

    Rows are joined by `row_index`; we DO NOT assume the two files are in
    the same order, because some providers (batch APIs) reorder results.
    """
    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(predictions_path)

    pred_by_idx = {r["row_index"]: r for r in pred_rows}

    example_scores: list[ExampleScore] = []
    for g in gold_rows:
        idx = g["row_index"]
        pr = pred_by_idx.get(idx)
        pred_obj = pr.get("pred") if pr else None
        example_scores.append(score_one(g, pred_obj, g["description"], idx))

    return aggregate(example_scores)


if __name__ == "__main__":
    # Smoke entry point: `python -m listing_parser.benchmarks.scorer GOLD PREDS`
    import sys as _sys

    if len(_sys.argv) != 3:
        print(
            "usage: python -m listing_parser.benchmarks.scorer GOLD.jsonl PREDS.jsonl",
            file=_sys.stderr,
        )
        raise SystemExit(2)
    report = score_run(Path(_sys.argv[1]), Path(_sys.argv[2]))
    print(json.dumps(report.to_dict(), indent=2))
