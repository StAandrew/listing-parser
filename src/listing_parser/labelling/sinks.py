"""Append-only sinks for labelled rows + the regenerated index.

The design priority is that re-running the labeller on the same input
is cheap and safe:

  * `LabelledFileWriter` is append-only over a stable `row_id`, so a
    resumed run only re-appends rows that weren't emitted last time.
  * `regenerate_index()` is idempotent — it walks every
    `data/labelled/*.json` batch and rebuilds the consolidated
    `_index.json` from scratch.

We keep the on-disk shape **lean** (matching HF's `{description,
output: fenced_string}` exactly) so the same file can be pushed to the
HF hub without any transformation. Metadata (retries, token counts,
model id) goes to a sibling JSONL under `_log/` that's gitignored —
that way noisy debug information never contaminates a PR diff.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_LABELLED_DIR = Path("data/labelled")


@dataclass(frozen=True)
class LabelledRow:
    """One labelled extraction ready for disk.

    `row_id` and `listing_type` are NOT written to the HF-shaped output
    file. They're only used for:
      * the sink's resume logic (`row_id` as the unique key);
      * the log file (so you can join logs back to source inputs).
    """

    row_id: str
    description: str
    listing_type: str
    fenced_output: str


class LabelledFileWriter:
    """Append-only, resume-aware writer for a single labelled batch.

    Layout on disk:

        data/labelled/<basename>.json        — human-readable JSON
                                                array, same shape as
                                                HF: {description, output}.
        data/labelled/_row_ids/<basename>.txt — one row_id per line,
                                                used to resume. Not
                                                committed (gitignored).
        data/labelled/_log/<basename>.jsonl   — per-attempt diagnostics.
                                                Not committed.

    Why a plain JSON array rather than JSONL for the visible file:
    PR reviewers open these in GitHub, which pretty-prints JSON but
    renders JSONL as a wall of text. 200-3200 rows fit comfortably.
    """

    def __init__(
        self,
        basename: str,
        *,
        labelled_dir: Path = DEFAULT_LABELLED_DIR,
    ) -> None:
        self.basename = basename
        self.out_path = labelled_dir / f"{basename}.json"
        self.row_ids_path = labelled_dir / "_row_ids" / f"{basename}.txt"
        self.log_path = labelled_dir / "_log" / f"{basename}.jsonl"

        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.row_ids_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._emitted: list[LabelledRow] = []
        # Row ids we've already written to disk in a previous run, loaded
        # once on construction. The pipeline asks us via
        # `already_emitted()` which row_ids to skip.
        self._existing_ids: set[str] = set()
        if self.row_ids_path.exists():
            self._existing_ids = {
                line.strip() for line in self.row_ids_path.read_text().splitlines() if line.strip()
            }
        if self.out_path.exists():
            for row in json.loads(self.out_path.read_text()):
                # Preserve existing rows so the final write merges old+new.
                # Old 2-column files (pre-listing_type era) won't have the
                # field; default to empty string so the merge doesn't
                # crash. The next run writes the new 3-column shape.
                self._emitted.append(
                    LabelledRow(
                        row_id="",  # unknown for previously-written rows
                        description=row["description"],
                        listing_type=row.get("listing_type", ""),
                        fenced_output=row["output"],
                    )
                )

    def already_emitted(self, row_id: str) -> bool:
        return row_id in self._existing_ids

    def emit(self, row: LabelledRow) -> None:
        """Queue a freshly-labelled row for writing.

        We don't `fsync` on every call; instead the pipeline calls
        `flush()` at a fixed cadence and at the end of the run.
        """
        self._emitted.append(row)
        self._existing_ids.add(row.row_id)

    def log_event(self, event: dict[str, Any]) -> None:
        """Append one line to the log file. Used for both success and failure
        events — include enough info to diagnose a bad row post-hoc.
        """
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        """Rewrite both the labelled file and the row_ids sidecar.

        Idempotent: calling twice with no new emits produces the same
        file bytes both times. This matters because the pipeline may
        flush mid-run (e.g. on a keyboard interrupt handler) and again
        at shutdown — we don't want half-written state.
        """
        # Stable ordering so diffs don't churn. Sort by row_id where
        # known, else by description. Row_id is deterministic
        # (sha1 of description) so it IS a stable ordering.
        ordered = sorted(self._emitted, key=lambda r: r.row_id or r.description)
        payload = [
            {
                "description": r.description,
                "listing_type": r.listing_type,
                "output": r.fenced_output,
            }
            for r in ordered
        ]
        self.out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.row_ids_path.write_text(
            "\n".join(rid for rid in sorted(self._existing_ids) if rid) + "\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Consolidated index across all labelled batches.
#
# The HF push script reads the index (not individual batches) so we have
# exactly one place that makes decisions about dedup and ordering. The
# benchmark test-set builder can also point at this file once it's
# ready, decoupling the benchmark layer from HF entirely.
# ---------------------------------------------------------------------------


def regenerate_index(
    labelled_dir: Path = DEFAULT_LABELLED_DIR,
) -> dict[str, Any]:
    """Walk all `data/labelled/*.json` batch files and rebuild `_index.json`.

    Dedup strategy: first 200 characters of description, matching the
    rule `test_set.py` uses. If two batches label the same description,
    the first one wins — that's the conservative choice because it
    preserves the "labelled once" state the test set was built against.

    Each row is preserved verbatim (`{description, listing_type,
    output}`) so the index can be used directly as the fine-tune
    dataset source without further transformation.

    Returns the parsed index dict so the CLI can print summary stats.
    """
    index_path = labelled_dir / "_index.json"
    batches = sorted(p for p in labelled_dir.glob("*.json") if p.name != "_index.json")

    seen_keys: set[str] = set()
    merged: list[dict[str, Any]] = []
    per_batch_counts: dict[str, int] = {}
    listing_type_counts: dict[str, int] = {}
    dedup_drops = 0

    for batch in batches:
        rows = json.loads(batch.read_text(encoding="utf-8"))
        kept = 0
        for row in rows:
            desc = row.get("description", "")
            key = desc.strip()[:200]
            if key in seen_keys:
                dedup_drops += 1
                continue
            seen_keys.add(key)
            # Defensively copy — some old batches may be missing
            # listing_type; carry "" through so downstream consumers
            # don't crash on a KeyError.
            merged.append({
                "description": row["description"],
                "listing_type": row.get("listing_type", ""),
                "output": row["output"],
            })
            lt = row.get("listing_type", "")
            listing_type_counts[lt] = listing_type_counts.get(lt, 0) + 1
            kept += 1
        per_batch_counts[batch.name] = kept

    index = {
        "n_rows": len(merged),
        "per_batch_counts": per_batch_counts,
        "listing_type_counts": listing_type_counts,
        "dedup_drops": dedup_drops,
        "rows": merged,
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    return index
