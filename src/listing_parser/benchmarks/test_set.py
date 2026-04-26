"""Build a held-out test set from the HF-hosted labelled dataset.

Pulls N stratified rows from `standrey/listing-descriptions` and freezes
them to `benchmarks/test_set.jsonl`. Once frozen, this file is committed
to git and MUST NOT be regenerated — it's the honest evaluation set.

The stratification axis is `listing_type` (Rent / Sale / Room), inferred
from which sub-block the gold output populates. This makes sure the test
set mirrors the real distribution instead of being 80% rent listings.

Usage:
    python -m listing_parser.benchmarks.test_set \
        --repo standrey/listing-descriptions \
        --split train \
        --n 60 \
        --out benchmarks/test_set.jsonl

Run ONCE per dataset version. Re-running silently produces a DIFFERENT
test set because of non-determinism in `Dataset.shuffle` across library
versions — the command writes a `.seed` sidecar file and refuses to
overwrite without --force.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from listing_parser.cleaning import parse_output_json, strip_json_fences

ListingType = Literal["Rent", "Sale", "Room"]

# Re-exported for backwards compatibility — callers that still import
# `parse_output_json` / `strip_json_fences` from this module keep working
# while the canonical definitions live in `listing_parser.cleaning`.
__all__ = [
    "ListingType",
    "dedupe_by_description",
    "infer_listing_type",
    "load_hf_rows",
    "main",
    "parse_output_json",
    "stratified_sample",
    "strip_json_fences",
]


def infer_listing_type(output: dict[str, Any]) -> ListingType:
    """Classify a gold row as Rent/Sale/Room from which sub-block it uses.

    Order matters: a listing carrying `room` data is a Room listing even
    if `rent` is also populated (SpareRoom rooms have rent metadata too).
    """
    if isinstance(output.get("room"), dict) and output["room"]:
        return "Room"
    if isinstance(output.get("sale"), dict) and output["sale"]:
        return "Sale"
    return "Rent"


def dedupe_by_description(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The HF dataset has near-duplicate descriptions (e.g. "Mitre Yard" x3).
    Keep the first occurrence so the test set doesn't concentrate on one listing.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row["description"] or "").strip()[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def stratified_sample(
    rows: list[dict[str, Any]],
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Proportional stratified sample across inferred listing_type.

    We match the input distribution rather than enforcing equal buckets —
    the training set is skewed and we want the test set to be honest about
    that. For small N this can produce 0 rows of a rare class; the caller
    should check the class histogram in the logs.
    """
    rng = random.Random(seed)
    buckets: dict[ListingType, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row["listing_type"]].append(row)

    total = sum(len(v) for v in buckets.values())
    picks: list[dict[str, Any]] = []
    for lt, bucket in buckets.items():
        quota = round(n * len(bucket) / total)
        quota = min(quota, len(bucket))
        rng.shuffle(bucket)
        picks.extend(bucket[:quota])
        print(f"  {lt}: took {quota}/{len(bucket)} available", file=sys.stderr)

    # Rounding can leave us +/- a few rows. Trim or top up from the largest bucket.
    if len(picks) > n:
        rng.shuffle(picks)
        picks = picks[:n]
    elif len(picks) < n:
        leftover = [r for r in rows if r not in picks]
        rng.shuffle(leftover)
        picks.extend(leftover[: n - len(picks)])

    return picks


def load_hf_rows(repo: str, split: str, config: str | None) -> list[dict[str, Any]]:
    """Load rows from the HF hub. Defers the import so the module stays importable
    without the `datasets` dependency for users who only want the scorer."""
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)

    rows: list[dict[str, Any]] = []
    for i, item in enumerate(ds):
        description = item.get("description")
        raw_output = item.get("output")
        if not description or not raw_output:
            continue
        parsed = parse_output_json(raw_output)
        if parsed is None:
            print(f"  row {i}: unparseable output, skipping", file=sys.stderr)
            continue
        column_type = item.get("listing_type") or ""
        rows.append(
            {
                "row_index": i,
                "description": description,
                "output": parsed,
                "listing_type": column_type if column_type in ("Rent", "Sale", "Room") else infer_listing_type(parsed),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="standrey/listing-descriptions")
    parser.add_argument("--config", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=60, help="target number of rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/test_set.jsonl"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing test set (DANGER: invalidates all historical runs)",
    )
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        print(
            f"refusing to overwrite {args.out}; pass --force to regenerate "
            f"(this invalidates all previous benchmark runs against the old set)",
            file=sys.stderr,
        )
        return 2

    print(f"loading {args.repo}/{args.config}:{args.split} ...", file=sys.stderr)
    rows = load_hf_rows(args.repo, args.split, args.config)
    print(f"  {len(rows)} valid rows after parse", file=sys.stderr)

    rows = dedupe_by_description(rows)
    print(f"  {len(rows)} rows after description dedup", file=sys.stderr)

    hist_before = Counter(r["listing_type"] for r in rows)
    print(f"  class histogram (full set): {dict(hist_before)}", file=sys.stderr)

    picked = stratified_sample(rows, args.n, seed=args.seed)
    hist_after = Counter(r["listing_type"] for r in picked)
    print(f"  class histogram (test set): {dict(hist_after)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in picked:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta = {
        "repo": args.repo,
        "config": args.config,
        "split": args.split,
        "seed": args.seed,
        "n_requested": args.n,
        "n_written": len(picked),
        "class_histogram": dict(hist_after),
        "source_total_rows": len(rows),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"wrote {len(picked)} rows -> {args.out}", file=sys.stderr)
    print(f"wrote metadata      -> {meta_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
