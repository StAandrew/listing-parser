"""The gold-to-predictions driver.

Reads a gold JSONL (row_index, description, listing_type, output),
invokes a `Runner` per row under bounded concurrency, and writes a
`predictions.jsonl` the scorer can consume. Resume is on by default:
the driver writes a sidecar of completed `row_index` values and skips
them on restart, same pattern as `labelling/sinks.py` but keyed on
integer row indices instead of sha1 row_ids.

Layout on disk (under `benchmarks/runs/<slug>/`):

    predictions.jsonl   public output — {row_index, pred, raw, [error]}
    _row_ids.txt        resume sidecar, one row_index per line
    _log.jsonl          per-attempt diagnostics (usage, timing, errors)

The sidecar + log files sit next to `predictions.jsonl` rather than in
a sibling `_row_ids/` directory — a benchmark run produces exactly one
output file, so there's no need for the multi-batch layout the labeller
uses. `.gitignore` patterns already exclude `_row_ids.txt` / `_log.jsonl`
in `benchmarks/runs/*/` via the directory-scoped rule in the repo root
gitignore (see ".gitignore" for details).

The scorer must be invariant to row order in predictions.jsonl — some
future batch providers reorder results — so we never try to preserve
gold order here. The driver writes as rows complete.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from listing_parser.labelling.progress import ProgressReporter
from listing_parser.runners.base import ListingType, Runner


@dataclass
class RunStats:
    """Summary of a prediction run — for the CLI footer."""

    n_gold: int
    n_skipped_resumed: int
    n_predicted: int
    n_parse_failed: int
    outcome_counts: Counter[str] = field(default_factory=Counter)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    elapsed_s: float = 0.0


def _load_gold(gold_path: Path) -> list[dict[str, Any]]:
    """Parse a gold JSONL into a list of dicts.

    Kept separate from the async driver so tests can use a list literal.
    """
    rows: list[dict[str, Any]] = []
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_resume_ids(sidecar_path: Path) -> set[int]:
    """Return the set of row_index values already completed in a prior run."""
    if not sidecar_path.exists():
        return set()
    ids: set[int] = set()
    for line in sidecar_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ids.add(int(line))
        except ValueError:
            # Tolerate a stray malformed line rather than wiping progress.
            continue
    return ids


class _PredictionsWriter:
    """Append-only writer for `predictions.jsonl` + sidecars.

    Single-threaded from the outside: the async driver awaits each
    `write()` in sequence under its own lock-free model (one writer,
    many awaiters).  We fsync at flush boundaries so a Ctrl-C loses
    at most the current in-flight chunk's worth of rows.
    """

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = out_dir / "predictions.jsonl"
        self.sidecar_path = out_dir / "_row_ids.txt"
        self.log_path = out_dir / "_log.jsonl"
        # Open in append mode so resume is naturally additive. The
        # scorer reads the whole file, order-insensitively; dupes are
        # not possible because resume skips already-written row_index.
        self._preds_fh = self.predictions_path.open("a", encoding="utf-8")
        self._sidecar_fh = self.sidecar_path.open("a", encoding="utf-8")
        self._log_fh = self.log_path.open("a", encoding="utf-8")

    def write_prediction(
        self,
        row_index: int,
        pred: dict[str, Any] | None,
        raw: str,
        error: str | None,
    ) -> None:
        payload: dict[str, Any] = {
            "row_index": row_index,
            "pred": pred,
            "raw": raw,
        }
        if error is not None:
            payload["error"] = error
        self._preds_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._sidecar_fh.write(f"{row_index}\n")

    def write_log(self, event: dict[str, Any]) -> None:
        self._log_fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._preds_fh.flush()
        self._sidecar_fh.flush()
        self._log_fh.flush()

    def close(self) -> None:
        self.flush()
        self._preds_fh.close()
        self._sidecar_fh.close()
        self._log_fh.close()


async def _predict_one(
    runner: Runner,
    row: dict[str, Any],
    writer: _PredictionsWriter,
    stats: RunStats,
    reporter: ProgressReporter,
    lock: asyncio.Lock,
) -> None:
    """Run `runner.predict` on one row and persist the result.

    `lock` serialises writer access — the file is append-only but two
    tasks racing on `.write()` would interleave bytes. A lock is cheap
    since the write itself is microseconds next to the Bedrock call.
    """
    row_index = int(row["row_index"])
    listing_type: ListingType = row.get("listing_type") or "Rent"
    description: str = row["description"]

    await reporter.on_row_start()
    started = time.monotonic()
    result = await runner.predict(description, listing_type)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    parse_ok = result.pred is not None
    success = parse_ok and result.error is None

    async with lock:
        writer.write_prediction(row_index, result.pred, result.raw, result.error)
        writer.write_log(
            {
                "row_index": row_index,
                "listing_type": listing_type,
                "parse_ok": parse_ok,
                "error": result.error,
                "elapsed_ms": elapsed_ms,
                "usage": result.usage or {},
            }
        )

    if parse_ok:
        stats.n_predicted += 1
        stats.outcome_counts["success_parsed"] += 1
    else:
        stats.n_parse_failed += 1
        stats.outcome_counts[
            "fail_runner_error" if result.error and "parse" not in result.error else "fail_parse"
        ] += 1

    usage = result.usage or {}
    stats.total_input_tokens += int(usage.get("input_tokens", 0) or 0)
    stats.total_output_tokens += int(usage.get("output_tokens", 0) or 0)
    stats.total_cache_read_tokens += int(usage.get("cache_read_tokens", 0) or 0)
    stats.total_cache_write_tokens += int(usage.get("cache_write_tokens", 0) or 0)

    await reporter.on_row_done(success=success)


async def _run_async(
    runner: Runner,
    gold_rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    concurrency: int,
    resume: bool,
    flush_every: int,
    show_progress: bool,
) -> RunStats:
    writer = _PredictionsWriter(out_dir)

    existing = _load_resume_ids(writer.sidecar_path) if resume else set()
    todo = [r for r in gold_rows if int(r["row_index"]) not in existing]
    stats = RunStats(
        n_gold=len(gold_rows),
        n_skipped_resumed=len(gold_rows) - len(todo),
        n_predicted=0,
        n_parse_failed=0,
    )

    reporter = ProgressReporter(total=len(todo))
    reporter.start()
    ticker_task: asyncio.Task[None] | None = None
    if show_progress and todo:
        ticker_task = asyncio.create_task(reporter.run_ticker())

    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(concurrency)

    async def _gated(row: dict[str, Any]) -> None:
        # The runner may have its own semaphore (Bedrock does) but we
        # also cap here so non-Bedrock runners without internal rate
        # limiting still benefit from --concurrency. Taking both gates
        # is fine — the outer one becomes a no-op once the runner's is
        # the tighter bound.
        async with semaphore:
            try:
                await _predict_one(runner, row, writer, stats, reporter, lock)
            except BaseException:
                await reporter.on_row_done(success=False)
                raise

    started = time.monotonic()
    try:
        for chunk_start in range(0, len(todo), flush_every):
            chunk = todo[chunk_start : chunk_start + flush_every]
            await asyncio.gather(*(_gated(row) for row in chunk))
            writer.flush()
            if show_progress:
                await reporter.on_flush(
                    chunk_index=chunk_start // flush_every + 1,
                    cumulative_success=stats.n_predicted,
                )
    finally:
        if ticker_task is not None:
            ticker_task.cancel()
            try:
                await ticker_task
            except asyncio.CancelledError:
                pass
        if show_progress and todo:
            reporter.stop()
        writer.close()

    stats.elapsed_s = time.monotonic() - started
    return stats


def run_predictions(
    runner: Runner,
    gold_path: Path,
    out_dir: Path,
    *,
    concurrency: int = 4,
    resume: bool = True,
    flush_every: int = 8,
    show_progress: bool = True,
) -> RunStats:
    """Synchronous entry point used by the CLI.

    `flush_every=8` (vs 32 in the labeller) because benchmark runs are
    ~60 rows — flushing every 8 gives ~7 checkpoints per run, so a
    Ctrl-C loses at most 8 rows but still amortises fsync cost.
    """
    gold_rows = _load_gold(gold_path)
    return asyncio.run(
        _run_async(
            runner,
            gold_rows,
            out_dir,
            concurrency=concurrency,
            resume=resume,
            flush_every=flush_every,
            show_progress=show_progress,
        )
    )


def format_stats(stats: RunStats, runner_name: str) -> str:
    """Human-readable summary written to stderr at the end of a run."""
    lines = [
        "",
        f"  runner           : {runner_name}",
        f"  gold rows        : {stats.n_gold}",
        f"  resumed (skipped): {stats.n_skipped_resumed}",
        f"  predicted (ok)   : {stats.n_predicted}",
        f"  failed           : {stats.n_parse_failed}",
        f"  elapsed          : {stats.elapsed_s:.1f}s",
    ]
    if stats.outcome_counts:
        lines.append("")
        lines.append("  outcome counts:")
        for k, v in sorted(stats.outcome_counts.items()):
            lines.append(f"    {k:>28}: {v}")
    if stats.total_input_tokens or stats.total_output_tokens:
        lines.append("")
        lines.append("  token usage:")
        lines.append(f"    input       : {stats.total_input_tokens:,}")
        lines.append(f"    output      : {stats.total_output_tokens:,}")
        lines.append(f"    cache read  : {stats.total_cache_read_tokens:,}")
        lines.append(f"    cache write : {stats.total_cache_write_tokens:,}")
    return "\n".join(lines)
