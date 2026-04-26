"""Mirror the local labelled index to the HF dataset.

Reads `data/labelled/_index.json` (rebuilt by `label_listings.py` on
every run) and replaces the parquet files on
`standrey/listing-descriptions` with a single consolidated batch.

The repo is the authoritative store — the HF hub is a mirror built on
demand for training jobs that prefer `load_dataset(...)` to reading
JSON from a local checkout. Nothing about labelling actually needs HF;
this script only exists so external consumers (Unsloth notebooks, etc.)
can keep using the familiar HF load path.

Commit layout is kept byte-compatible with `clean_hf_labels.py`: one
`data/batch_00000.parquet`, metadata.json refreshed, old batches
deleted in the same atomic commit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def push(
    index_path: Path,
    repo: str,
    token: str,
    commit_message: str,
) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415
    from huggingface_hub import (  # noqa: PLC0415
        CommitOperationAdd,
        CommitOperationDelete,
        HfApi,
        hf_hub_download,
    )

    index = json.loads(index_path.read_text(encoding="utf-8"))
    rows = index["rows"]
    if not rows:
        print("index has no rows; refusing to push an empty dataset.", file=sys.stderr)
        sys.exit(2)

    print(f"pushing {len(rows)} rows to {repo} ...", file=sys.stderr)

    api = HfApi(token=token)
    existing = api.list_repo_files(repo, repo_type="dataset")
    old_batches = [
        f for f in existing if f.startswith("data/") and f.endswith(".parquet")
    ]
    print(f"  will delete {len(old_batches)} old parquet files", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        parquet_path = tmp_path / "batch_00000.parquet"
        # {description, listing_type, output} shape. `listing_type` is
        # a first-class column so downstream trainers / analysis can
        # filter on it without parsing the JSON in `output`. Rows that
        # are missing the field (migrated from a pre-listing_type era
        # of this repo) fall back to "" — HF parquet tolerates that.
        table = pa.Table.from_pylist(
            [
                {
                    "description": r["description"],
                    "listing_type": r.get("listing_type", ""),
                    "output": r["output"],
                }
                for r in rows
            ]
        )
        pq.write_table(table, parquet_path)
        print(
            f"  wrote replacement parquet: {parquet_path.stat().st_size:,} bytes",
            file=sys.stderr,
        )

        operations: list[CommitOperationAdd | CommitOperationDelete] = []

        # Best-effort refresh of metadata.json so downstream tooling
        # (including the Unsloth data-designer catalogue) sees a
        # consistent record count. We tolerate failures here — the core
        # artefact is the parquet.
        try:
            meta_local = hf_hub_download(
                repo, "metadata.json", repo_type="dataset", token=token
            )
            meta = json.loads(Path(meta_local).read_text())
            meta["actual_num_records"] = len(rows)
            meta["file_paths"] = {"data": ["data/batch_00000.parquet"]}
            meta["num_completed_batches"] = 1
            meta["total_num_batches"] = 1
            new_meta = tmp_path / "metadata.json"
            new_meta.write_text(json.dumps(meta, indent=2))
            operations.append(
                CommitOperationAdd(
                    path_in_repo="metadata.json", path_or_fileobj=str(new_meta)
                )
            )
        except Exception as e:  # pragma: no cover
            print(f"  WARNING: could not refresh metadata.json ({e})", file=sys.stderr)

        operations.append(
            CommitOperationAdd(
                path_in_repo="data/batch_00000.parquet",
                path_or_fileobj=str(parquet_path),
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

    print(f"  push complete: {len(rows)} rows", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--index",
        type=Path,
        default=Path("data/labelled/_index.json"),
        help="path to the consolidated labelled index (default: data/labelled/_index.json)",
    )
    p.add_argument("--repo", default="standrey/listing-descriptions")
    p.add_argument(
        "--commit-message",
        default="sync: mirror labelled/_index.json from repo",
    )
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    args = p.parse_args()

    _load_env_file(args.env_file)
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set (add it to .env or export it).", file=sys.stderr)
        return 2

    if not args.index.exists():
        print(
            f"ERROR: {args.index} not found. Run scripts/label_listings.py first.",
            file=sys.stderr,
        )
        return 2

    push(args.index, args.repo, token, args.commit_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
