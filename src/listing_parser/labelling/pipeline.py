"""The per-row labelling state machine.

Each row goes through:

    1.  Build system + user messages (prompting.py).
    2.  Call Bedrock (teacher.py).
    3.  Parse the fenced JSON (cleaning.parse_output_json).
    4.  Clean + validate (cleaning.clean_extraction + validate_extraction).
    5.  On parse/validation failure, send a correction prompt back to
        the teacher; retry up to `max_retries` times total.
    6.  Emit to the sink on success, log + drop on failure.

Concurrency is driven by asyncio.gather over a bounded semaphore inside
`teacher.call_teacher`. The pipeline itself has no shared mutable state
beyond the writer, which owns its own serialisation.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from listing_parser.cleaning import (
    clean_extraction,
    format_fenced_json,
    parse_output_json,
    validate_extraction,
)
from listing_parser.labelling.progress import ProgressReporter
from listing_parser.labelling.sinks import (
    DEFAULT_LABELLED_DIR,
    LabelledFileWriter,
    LabelledRow,
    regenerate_index,
)
from listing_parser.labelling.sources import ListingInput
from listing_parser.labelling.teacher import (
    DEFAULT_MODEL_ID,
    DEFAULT_REGION,
    TeacherResponse,
    call_teacher,
)
from listing_parser.prompting import (
    build_correction_message,
    build_system_prompt,
    build_user_message,
)


@dataclass
class RunStats:
    """Summary of a labelling run, printed by the CLI on exit."""

    n_inputs: int
    n_skipped_resumed: int
    n_success: int
    n_failed: int
    outcome_counts: Counter[str]
    cleaning_changes: Counter[str]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_write_tokens: int = 0
    elapsed_s: float = 0.0


async def _label_one(
    row: ListingInput,
    *,
    system_prompt: str,
    writer: LabelledFileWriter,
    semaphore: asyncio.Semaphore,
    stats: RunStats,
    reporter: ProgressReporter,
    max_retries: int,
    profile: str | None,
    region: str,
    model_id: str,
) -> None:
    """Process a single input row end-to-end.

    All failures are captured into the stats counter + written to the
    log file. We never let an exception propagate out of here; that
    would tear down the whole `asyncio.gather` and kill all in-flight
    rows.
    """
    user_msg = build_user_message(row.description, row.listing_type)
    turns: tuple[str, ...] = (user_msg,)
    last_raw: str = ""
    last_error: str = ""

    await reporter.on_row_start()

    try:
        for attempt in range(max_retries + 1):
            try:
                resp: TeacherResponse = await call_teacher(
                    system_prompt,
                    turns,
                    semaphore=semaphore,
                    profile=profile,
                    region=region,
                    model_id=model_id,
                    on_throttle=reporter.on_throttle,
                )
            except Exception as e:  # pragma: no cover — surfaces runtime errors
                stats.outcome_counts["fail_teacher_error"] += 1
                stats.n_failed += 1
                reporter.on_teacher_error(row.row_id, f"{type(e).__name__}: {e}")
                writer.log_event(
                    {
                        "row_id": row.row_id,
                        "listing_type": row.listing_type,
                        "outcome": "fail_teacher_error",
                        "error": f"{type(e).__name__}: {e}",
                        "attempt": attempt,
                    }
                )
                await reporter.on_row_done(success=False)
                return

            stats.total_input_tokens += resp.input_tokens
            stats.total_output_tokens += resp.output_tokens
            stats.total_cache_read_tokens += resp.cache_read_tokens
            stats.total_cache_write_tokens += resp.cache_write_tokens
            if resp.cache_write_tokens:
                await reporter.on_cache_write(resp.cache_write_tokens)
            if resp.cache_read_tokens:
                await reporter.on_cache_read(resp.cache_read_tokens)
            last_raw = resp.text

            parsed = parse_output_json(resp.text)
            if parsed is None:
                last_error = "model output was not a parseable JSON object"
            else:
                row_changes: Counter[str] = Counter()
                cleaned, _ = clean_extraction(parsed, row.description, row_changes)
                err = validate_extraction(cleaned)
                if err is None:
                    # Success path. Merge the per-row cleaning changes
                    # into the run-wide stats so the CLI can report
                    # "N rows had this particular repair" at the end.
                    stats.cleaning_changes.update(row_changes)
                    stats.outcome_counts[
                        "success_clean" if not row_changes else "success_with_repairs"
                    ] += 1
                    stats.n_success += 1
                    writer.emit(
                        LabelledRow(
                            row_id=row.row_id,
                            description=row.description,
                            listing_type=row.listing_type,
                            fenced_output=format_fenced_json(cleaned),
                        )
                    )
                    writer.log_event(
                        {
                            "row_id": row.row_id,
                            "listing_type": row.listing_type,
                            "outcome": "success",
                            "attempt": attempt,
                            "repairs": dict(row_changes),
                            "usage": {
                                "input_tokens": resp.input_tokens,
                                "output_tokens": resp.output_tokens,
                                "cache_read": resp.cache_read_tokens,
                                "cache_write": resp.cache_write_tokens,
                                "latency_ms": resp.latency_ms,
                            },
                        }
                    )
                    await reporter.on_row_done(success=True)
                    return
                last_error = err

            # Failure path — either unparseable or schema-invalid. If
            # we have retries left, build a correction turn and loop.
            if attempt < max_retries:
                await reporter.on_retry(row.row_id, attempt, last_error)
                turns = (user_msg, build_correction_message(last_raw, last_error))
                continue

            # Out of retries; log and drop.
            stats.outcome_counts["fail_after_retries"] += 1
            stats.n_failed += 1
            reporter.on_drop(row.row_id, last_error)
            writer.log_event(
                {
                    "row_id": row.row_id,
                    "listing_type": row.listing_type,
                    "outcome": "fail_after_retries",
                    "attempt": attempt,
                    "final_error": last_error[:500],
                    "final_raw_preview": last_raw[:500],
                }
            )
            await reporter.on_row_done(success=False)
            return
    except BaseException:
        # Make sure a task cancellation (Ctrl-C) doesn't leave the
        # in_flight counter permanently skewed. We re-raise so the
        # gather still sees the cancellation.
        await reporter.on_row_done(success=False)
        raise


async def _run_async(
    inputs: list[ListingInput],
    *,
    labelled_dir: Path,
    basename: str,
    concurrency: int,
    max_retries: int,
    profile: str | None,
    region: str,
    model_id: str,
    flush_every: int,
    show_progress: bool,
) -> RunStats:
    """Orchestrator: spin up a gather, flush the writer periodically.

    `show_progress=True` starts a ticker that repaints a status line;
    turn it off when running under a subprocess that captures stderr
    for parsing (the rolling numbers aren't useful there and they make
    the logs noisy).
    """
    system_prompt = build_system_prompt()
    writer = LabelledFileWriter(basename, labelled_dir=labelled_dir)

    # Resume support: skip inputs whose row_id we've already written.
    todo: list[ListingInput] = [r for r in inputs if not writer.already_emitted(r.row_id)]
    n_skipped = len(inputs) - len(todo)

    stats = RunStats(
        n_inputs=len(inputs),
        n_skipped_resumed=n_skipped,
        n_success=0,
        n_failed=0,
        outcome_counts=Counter(),
        cleaning_changes=Counter(),
    )

    semaphore = asyncio.Semaphore(concurrency)
    reporter = ProgressReporter(total=len(todo))
    reporter.start()

    # Start the status-line ticker only if caller wants progress AND
    # there's work to do. Zero-row runs happen on full resumes — no
    # point confusing the user with a 0% bar that never moves.
    ticker_task: asyncio.Task[None] | None = None
    if show_progress and todo:
        ticker_task = asyncio.create_task(reporter.run_ticker())

    started = time.monotonic()

    try:
        # We don't create all tasks up front because keyboard interrupts
        # during gather are clumsy; instead we process in flush-sized
        # chunks so we can flush between them and a Ctrl-C loses at
        # most `flush_every` rows' worth of work.
        for chunk_idx, chunk_start in enumerate(range(0, len(todo), flush_every)):
            chunk = todo[chunk_start : chunk_start + flush_every]
            await asyncio.gather(
                *(
                    _label_one(
                        row,
                        system_prompt=system_prompt,
                        writer=writer,
                        semaphore=semaphore,
                        stats=stats,
                        reporter=reporter,
                        max_retries=max_retries,
                        profile=profile,
                        region=region,
                        model_id=model_id,
                    )
                    for row in chunk
                )
            )
            writer.flush()
            if show_progress:
                await reporter.on_flush(
                    chunk_index=chunk_idx + 1,
                    cumulative_success=stats.n_success,
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

    stats.elapsed_s = time.monotonic() - started
    # Regenerate the consolidated index at the end of every run so
    # downstream consumers always see the latest state.
    regenerate_index(labelled_dir)
    return stats


def run(
    inputs: Iterable[ListingInput],
    *,
    basename: str,
    labelled_dir: Path = DEFAULT_LABELLED_DIR,
    concurrency: int = 8,
    max_retries: int = 2,
    profile: str | None = None,
    region: str = DEFAULT_REGION,
    model_id: str = DEFAULT_MODEL_ID,
    flush_every: int = 32,
    show_progress: bool = True,
) -> RunStats:
    """Synchronous entry point used by the CLI.

    Accepting an `Iterable[ListingInput]` rather than a file path lets
    tests swap in a small hand-built list without touching disk.
    """
    inputs_list = list(inputs)
    return asyncio.run(
        _run_async(
            inputs_list,
            labelled_dir=labelled_dir,
            basename=basename,
            concurrency=concurrency,
            max_retries=max_retries,
            profile=profile,
            region=region,
            model_id=model_id,
            flush_every=flush_every,
            show_progress=show_progress,
        )
    )


def format_stats(stats: RunStats) -> str:
    """Render a human-readable summary — used by the CLI's stderr output."""
    lines = [
        "",
        f"  inputs           : {stats.n_inputs}",
        f"  resumed (skipped): {stats.n_skipped_resumed}",
        f"  success          : {stats.n_success}",
        f"  failed           : {stats.n_failed}",
        f"  elapsed          : {stats.elapsed_s:.1f}s",
        "",
        "  outcome counts:",
    ]
    for k, v in sorted(stats.outcome_counts.items()):
        lines.append(f"    {k:>28}: {v}")
    if stats.cleaning_changes:
        lines.append("")
        lines.append("  cleaning repairs applied (top 15):")
        for k, v in sorted(
            stats.cleaning_changes.items(), key=lambda kv: (-kv[1], kv[0])
        )[:15]:
            lines.append(f"    {v:>4}  {k}")
    if stats.total_input_tokens or stats.total_output_tokens:
        lines.append("")
        lines.append("  token usage:")
        lines.append(f"    input       : {stats.total_input_tokens:,}")
        lines.append(f"    output      : {stats.total_output_tokens:,}")
        lines.append(f"    cache read  : {stats.total_cache_read_tokens:,}")
        lines.append(f"    cache write : {stats.total_cache_write_tokens:,}")
    return "\n".join(lines)
