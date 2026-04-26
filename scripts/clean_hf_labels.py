"""One-off migration: clean already-labelled rows on the HF source dataset.

History
-------
Before `listing_parser.labelling` existed, teacher-labelled rows landed
on `standrey/listing-descriptions` without any post-processing, and
Haiku's systematic mistakes (hallucinated keys, out-of-vocab amenities,
ungrounded evidence, the occasional unparseable row) went with them.
This script exists to repair the already-uploaded rows in place.

It is **not** on the runtime path. Once the labelling pipeline owns all
future writes to the HF hub, every new row is cleaned at label-writing
time via `listing_parser.cleaning.clean_extraction`, and this script is
only useful if you ever need to re-clean the existing hub rows against
a newer version of the cleaning rules (e.g. after adding a vocabulary
value or tightening an enum).

All the repair logic lives in `listing_parser.cleaning` so the labelling
pipeline and this one-off migration can't drift apart. This file is only
the HF-specific glue: fetch rows, apply cleaning, check the frozen
test-set invariant, and push back.

Priorities (mirroring CLAUDE.md)
--------------------------------
  * Hard-fail if any row in `benchmarks/test_set.jsonl` would be
    DROPPED by cleanup — that changes what `test_set.py --seed 42`
    reproduces, which is the silent-invalidation scenario the CLAUDE
    rules call out.
  * Warn (not fail) if any test-set row would be content-REPAIRED. The
    committed gold file embeds its own parsed copy, so past reports
    stay valid; the HF source and gold just drift until the user
    decides to refresh the gold file.

Usage
-----
    # Dry-run (default). Writes cleaned rows to a local JSONL for
    # inspection without touching HF.
    python scripts/clean_hf_labels.py --out /tmp/cleaned.jsonl

    # After eyeballing the diff, push back:
    python scripts/clean_hf_labels.py --push

Requires `HF_TOKEN` in `.env` with write access to the target repo for
the push path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from listing_parser.benchmarks.test_set import infer_listing_type
from listing_parser.cleaning import (
    clean_extraction,
    format_fenced_json,
    parse_output_json,
    validate_extraction,
)


def _load_env_file(path: Path) -> None:
    """Minimal .env loader — avoids pulling python-dotenv for one variable."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def run(
    repo: str,
    config: str,
    split: str,
    out_path: Path | None,
    test_set_path: Path,
    push: bool,
    commit_message: str,
) -> int:
    from datasets import load_dataset  # noqa: PLC0415

    print(f"loading {repo}/{config}:{split} ...", file=sys.stderr)
    ds = load_dataset(repo, config, split=split)
    print(f"  {ds.num_rows} rows loaded", file=sys.stderr)

    test_set_indices: set[int] = set()
    test_set_gold: dict[int, dict[str, Any]] = {}
    if test_set_path.exists():
        with test_set_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                test_set_indices.add(row["row_index"])
                test_set_gold[row["row_index"]] = row["output"]
        print(
            f"  loaded {len(test_set_indices)} test-set row indices for invariant check",
            file=sys.stderr,
        )
    else:
        print(
            f"  WARNING: {test_set_path} not found; skipping test-set invariant check",
            file=sys.stderr,
        )

    changes: Counter[str] = Counter()
    row_outcome: Counter[str] = Counter()
    dropped_indices: list[tuple[int, str]] = []
    modified_test_set_rows: list[int] = []

    cleaned_rows: list[dict[str, str]] = []

    for i, item in enumerate(ds):
        description = item.get("description")
        raw_output = item.get("output")
        if not description or not raw_output:
            row_outcome["drop_missing_fields"] += 1
            dropped_indices.append((i, "missing_fields"))
            continue

        parsed = parse_output_json(raw_output)
        if parsed is None:
            row_outcome["drop_unparseable"] += 1
            dropped_indices.append((i, "unparseable"))
            continue

        row_changes: Counter[str] = Counter()
        cleaned, _ = clean_extraction(parsed, description, row_changes)

        err = validate_extraction(cleaned)
        if err is not None:
            row_outcome["drop_schema_invalid"] += 1
            dropped_indices.append((i, f"schema_invalid: {err[:120]}"))
            continue

        changes.update(row_changes)
        if row_changes:
            row_outcome["modified"] += 1
            if i in test_set_indices and cleaned != test_set_gold.get(i):
                modified_test_set_rows.append(i)
        else:
            row_outcome["unchanged"] += 1

        # `listing_type` becomes a first-class column on HF. Prefer the
        # existing value if the parquet already carries one (idempotent
        # on repeated migrations); otherwise infer from which sub-block
        # the cleaned output populates — same rule test_set.py uses.
        listing_type = item.get("listing_type") or infer_listing_type(cleaned)

        cleaned_rows.append(
            {
                "description": description,
                "listing_type": listing_type,
                "output": format_fenced_json(cleaned),
            }
        )

    # ---------------------------------------------------------------
    # Report.
    # ---------------------------------------------------------------
    print("\n=== row outcomes ===", file=sys.stderr)
    for k, v in sorted(row_outcome.items()):
        print(f"  {k:>24}: {v}", file=sys.stderr)

    print(f"\n  kept     : {len(cleaned_rows)} / {ds.num_rows}", file=sys.stderr)
    print(f"  dropped  : {len(dropped_indices)}", file=sys.stderr)

    if changes:
        print("\n=== cleanup operations (counts) ===", file=sys.stderr)
        for k, v in sorted(changes.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {v:>4}  {k}", file=sys.stderr)

    if dropped_indices:
        print(
            f"\n=== dropped row indices (first 20 of {len(dropped_indices)}) ===",
            file=sys.stderr,
        )
        for idx, reason in dropped_indices[:20]:
            print(f"  row {idx}: {reason}", file=sys.stderr)

    # ---------------------------------------------------------------
    # Frozen test-set invariant checks.
    # ---------------------------------------------------------------
    dropped_test_set = [idx for idx, _ in dropped_indices if idx in test_set_indices]
    if dropped_test_set:
        print(
            f"\nERROR: {len(dropped_test_set)} rows in the frozen test set would be "
            f"DROPPED by cleanup: {dropped_test_set}",
            file=sys.stderr,
        )
        print(
            "Refusing to proceed. Dropping a test-set row silently invalidates "
            "every historical benchmark report. Fix the offending rows by hand, "
            "or regenerate the test set with --force in test_set.py (which has "
            "its own consequences — see CLAUDE.md).",
            file=sys.stderr,
        )
        return 2

    if modified_test_set_rows:
        print(
            f"\nWARNING: {len(modified_test_set_rows)} rows in the frozen test set "
            f"would be REPAIRED by cleanup: {modified_test_set_rows}",
            file=sys.stderr,
        )
        print(
            "The committed benchmarks/test_set.jsonl already contains the "
            "PRE-cleanup labels, so historical reports stay valid. After push, "
            "the HF source will differ from the gold file for these indices. "
            "Either accept the drift or refresh the gold file on purpose.",
            file=sys.stderr,
        )

    # ---------------------------------------------------------------
    # Write / push.
    # ---------------------------------------------------------------
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in cleaned_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nwrote {len(cleaned_rows)} cleaned rows -> {out_path}", file=sys.stderr)

    if push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print(
                "ERROR: HF_TOKEN not set (add it to .env or export it).",
                file=sys.stderr,
            )
            return 2
        _push_to_hub(repo, cleaned_rows, token, commit_message)
    else:
        print("\n(dry-run — pass --push to upload to HF)", file=sys.stderr)

    return 0


def _push_to_hub(
    repo: str, rows: list[dict[str, str]], token: str, commit_message: str
) -> None:
    """Replace the parquet files under `data/` with a single cleaned batch.

    The dataset was originally produced by Unsloth's data-designer tool,
    which writes 25 small `data/batch_NNNNN.parquet` files plus a
    `metadata.json` + `builder_config.json` at the repo root.
    `load_dataset(repo, 'data')` globs `data/*.parquet`, so replacing
    the 25 batches with a single consolidated file loads identically.

    The commit is atomic: we add the new parquet and the refreshed
    `metadata.json`, and delete the old batches in the same operation
    list so the repo never exists in a half-migrated state.
    """
    import json as _json  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import (  # noqa: PLC0415
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
        hf_hub_download,
    )

    print(f"\npushing to {repo} ...", file=sys.stderr)
    api = HfApi(token=token)

    existing_files = api.list_repo_files(repo, repo_type="dataset")
    old_batches = [
        f for f in existing_files if f.startswith("data/") and f.endswith(".parquet")
    ]
    print(f"  will delete {len(old_batches)} old parquet files", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        new_parquet = tmp_path / "batch_00000.parquet"
        # Rows arrive here as {description, listing_type, output}; that
        # shape is the single source of truth for the HF mirror. See
        # README and push_labels_to_hf.py for why listing_type is a
        # top-level column rather than embedded in `output`.
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, new_parquet)
        print(
            f"  wrote replacement parquet: {new_parquet.stat().st_size:,} bytes",
            file=sys.stderr,
        )

        operations: list[CommitOperationAdd | CommitOperationDelete] = []
        try:
            meta_local = hf_hub_download(
                repo, "metadata.json", repo_type="dataset", token=token
            )
            meta = _json.loads(Path(meta_local).read_text())
            meta["actual_num_records"] = len(rows)
            meta["file_paths"] = {"data": ["data/batch_00000.parquet"]}
            meta["num_completed_batches"] = 1
            meta["total_num_batches"] = 1
            new_meta = tmp_path / "metadata.json"
            new_meta.write_text(_json.dumps(meta, indent=2))
            operations.append(
                CommitOperationAdd(
                    path_in_repo="metadata.json", path_or_fileobj=str(new_meta)
                )
            )
        except Exception as e:  # pragma: no cover — best effort only
            print(f"  WARNING: could not refresh metadata.json ({e})", file=sys.stderr)

        operations.append(
            CommitOperationAdd(
                path_in_repo="data/batch_00000.parquet",
                path_or_fileobj=str(new_parquet),
            )
        )
        for old in old_batches:
            if old == "data/batch_00000.parquet":
                continue
            operations.append(CommitOperationDelete(path_in_repo=old))

        api.create_commit(
            repo_id=repo,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
        )
    print(f"  push complete: {len(rows)} rows in 1 parquet batch", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="standrey/listing-descriptions")
    parser.add_argument("--config", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write cleaned rows as JSONL to this path (for dry-run inspection)",
    )
    parser.add_argument(
        "--test-set",
        type=Path,
        default=Path("benchmarks/test_set.jsonl"),
        help="path to the frozen test set; used for the invariant check",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="upload the cleaned rows back to the HF repo (requires HF_TOKEN)",
    )
    parser.add_argument(
        "--commit-message",
        default="clean: repair Haiku label artefacts (typos, out-of-vocab, ungrounded evidence)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="path to .env file to source HF_TOKEN from (optional)",
    )
    args = parser.parse_args()

    _load_env_file(args.env_file)

    return run(
        repo=args.repo,
        config=args.config,
        split=args.split,
        out_path=args.out,
        test_set_path=args.test_set,
        push=args.push,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    raise SystemExit(main())
