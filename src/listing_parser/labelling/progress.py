"""Progress reporter for the labelling CLI.

The labelling pipeline is I/O-bound and, at N=8 concurrency on ~2k rows,
takes ~6 minutes. Without incremental output the CLI looks frozen, and
slow-path issues (a particular listing taking 20s, Bedrock throttling,
a prompt cache miss storm) stay invisible until the final stats.

This module is the thin UI layer on top of the pipeline:

  * A `ProgressReporter` that holds counters and renders a single-line
    status. `\\r`-overwrite on a TTY, newline-append when piped to a
    file (so `2>run.log` still gives you a useful running trace).
  * Direct `print()` for "notable" events (cache transitions, retries,
    drops, throttles) that are cheap and deserve immediate attention.
  * A small rolling-window latency tracker so the status line can show
    something more useful than "current row = 42/1800" — specifically,
    an ETA based on the last 20 rows' completion rate.

The pipeline holds a single `ProgressReporter` instance and hands it
to every `_label_one` task. Updates happen under a lock because tasks
can race on the same counters; the per-call overhead is microseconds.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TextIO


def _is_tty(stream: TextIO) -> bool:
    """Whether it's safe to use `\\r` overwriting on this stream."""
    try:
        return stream.isatty()
    except (AttributeError, ValueError):
        return False


@dataclass
class ProgressReporter:
    """Thread-safe-ish counters + periodic redraw.

    "Thread-safe-ish" because asyncio tasks all run on one thread — we
    use an `asyncio.Lock` to guard the event emitter, but the numeric
    counters are plain ints. Increments are atomic on CPython, which is
    good enough for a display-only counter; nothing else reads these
    values until the run finishes.
    """

    total: int                      # number of rows queued for labelling
    stream: TextIO = field(default_factory=lambda: sys.stderr)
    redraw_interval_s: float = 0.5

    done: int = 0
    in_flight: int = 0
    retries: int = 0
    drops: int = 0
    throttles: int = 0

    # Rolling window of (monotonic_timestamp, cumulative_done) so we can
    # compute rows-per-second over the last ~N completions, independent
    # of where we are in the run.
    _window: deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=32))

    # Flags so we only print the cache transition messages once.
    _cache_write_announced: bool = False
    _cache_read_announced: bool = False

    started_at: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _stopped: bool = False

    def start(self) -> None:
        self.started_at = time.monotonic()
        self._window.append((self.started_at, 0))

    # ----------------------------------------------------------------
    # Event hooks called from the pipeline.
    # These are `async def` so callers can `await` them — we don't
    # actually need to block, but uniform signatures avoid bugs where
    # one hook is awaited and another isn't.
    # ----------------------------------------------------------------

    async def on_row_start(self) -> None:
        async with self._lock:
            self.in_flight += 1

    async def on_row_done(self, success: bool) -> None:
        async with self._lock:
            self.in_flight = max(0, self.in_flight - 1)
            self.done += 1
            self._window.append((time.monotonic(), self.done))
            if not success:
                self.drops += 1

    async def on_retry(self, row_id: str, attempt: int, reason: str) -> None:
        """A parse/validation failure triggered a correction prompt."""
        async with self._lock:
            self.retries += 1
        # Retries are rare (<5% in practice); surface them inline so the
        # log doesn't have to be read to know they're happening.
        # Flatten the reason — Pydantic errors contain newlines that
        # would otherwise break the single-line log format.
        flat_reason = " ".join(reason.split())
        self._emit_inline(
            f"retry attempt={attempt} row_id={row_id} reason={flat_reason[:120]}"
        )

    async def on_throttle(self) -> None:
        async with self._lock:
            self.throttles += 1
            n = self.throttles
        # Throttles are common under concurrency + quota limits. Log
        # every 10 so the user sees sustained pressure without being
        # drowned in per-event lines.
        if n == 1 or n % 10 == 0:
            self._emit_inline(
                f"bedrock throttle (total {n} so far — consider lowering --concurrency)"
            )

    async def on_cache_write(self, tokens: int) -> None:
        """First successful cache write of the run. After this, subsequent
        rows should see `on_cache_read` firing instead.
        """
        if self._cache_write_announced:
            return
        self._cache_write_announced = True
        self._emit_inline(
            f"prompt cache: warming ({tokens:,} tokens written, 1h TTL)"
        )

    async def on_cache_read(self, tokens: int) -> None:
        if self._cache_read_announced:
            return
        self._cache_read_announced = True
        self._emit_inline(
            f"prompt cache: HIT ({tokens:,} tokens read from cache)"
        )

    def on_drop(self, row_id: str, reason: str) -> None:
        """A row ran out of retries and was dropped. Called synchronously
        from the pipeline so the log line appears immediately.
        """
        self._emit_inline(
            f"DROP row_id={row_id} after_retries — {reason[:120]}"
        )

    def on_teacher_error(self, row_id: str, error: str) -> None:
        """Non-retryable teacher error (auth failure, bad profile, …)."""
        self._emit_inline(f"TEACHER ERROR row_id={row_id}: {error[:160]}")

    async def on_flush(self, chunk_index: int, cumulative_success: int) -> None:
        """Called after each `flush_every` chunk writes to disk."""
        rate = self._rate_per_s()
        eta = self._eta_s()
        self._emit_inline(
            f"flushed chunk {chunk_index} (cumulative {cumulative_success} ok) "
            f"rate={rate:.1f} rows/s eta={_fmt_duration(eta)}"
        )

    # ----------------------------------------------------------------
    # Rendering
    # ----------------------------------------------------------------

    def render_status(self) -> str:
        rate = self._rate_per_s()
        eta = self._eta_s()
        pct = (100.0 * self.done / self.total) if self.total else 0.0
        bar = _progress_bar(pct, width=24)
        return (
            f"  [{bar}] {self.done}/{self.total} ({pct:5.1f}%) "
            f"in_flight={self.in_flight} retries={self.retries} drops={self.drops} "
            f"throttles={self.throttles} rate={rate:.1f}/s eta={_fmt_duration(eta)}"
        )

    def _emit_inline(self, line: str) -> None:
        """Print a standalone log line without disturbing the status bar.

        On a TTY we write a carriage return + pad-to-clear first so the
        status line is overwritten, then print the event + newline, then
        redraw the status on the next tick. When piped, we just print.
        """
        if _is_tty(self.stream):
            self.stream.write("\r" + " " * 120 + "\r")
            self.stream.write(line + "\n")
            # Redraw status immediately so the user never loses context.
            self.stream.write(self.render_status())
            self.stream.flush()
        else:
            self.stream.write(line + "\n")
            self.stream.flush()

    async def run_ticker(self) -> None:
        """Background task: repaint the status line periodically.

        Safe to cancel; just breaks out of the loop.
        """
        tty = _is_tty(self.stream)
        try:
            while not self._stopped:
                if tty:
                    self.stream.write("\r" + self.render_status())
                    self.stream.flush()
                await asyncio.sleep(self.redraw_interval_s)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Final render + newline so the prompt lands on a fresh line."""
        self._stopped = True
        if _is_tty(self.stream):
            self.stream.write("\r" + self.render_status() + "\n")
        else:
            self.stream.write(self.render_status() + "\n")
        self.stream.flush()

    # ----------------------------------------------------------------
    # Rate / ETA math
    # ----------------------------------------------------------------

    def _rate_per_s(self) -> float:
        if len(self._window) < 2:
            return 0.0
        t0, n0 = self._window[0]
        t1, n1 = self._window[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return (n1 - n0) / dt

    def _eta_s(self) -> float:
        rate = self._rate_per_s()
        if rate <= 0:
            return 0.0
        remaining = max(0, self.total - self.done)
        return remaining / rate


def _progress_bar(pct: float, width: int) -> str:
    filled = int(width * pct / 100.0)
    return "#" * filled + "-" * (width - filled)


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "--:--"
    seconds = int(seconds)
    if seconds >= 3600:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h{m:02d}m{s:02d}s"
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
