"""Tests for the labelling sinks + progress reporter.

We don't test the pipeline end-to-end here (that would need a mocked
Bedrock client); we cover the pieces that are pure logic: the
append-only writer, the index regeneration, and the progress reporter's
event bookkeeping.
"""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from listing_parser.labelling.progress import (
    ProgressReporter,
    _fmt_duration,
    _progress_bar,
)
from listing_parser.labelling.sinks import (
    LabelledFileWriter,
    LabelledRow,
    regenerate_index,
)

# ---------------------------------------------------------------------------
# LabelledFileWriter — append, resume, flush idempotency.
# ---------------------------------------------------------------------------


def _row(rid: str, desc: str, lt: str = "Rent", output: str = "{}") -> LabelledRow:
    return LabelledRow(
        row_id=rid,
        description=desc,
        listing_type=lt,
        fenced_output=output,
    )


def test_writer_persists_listing_type_as_third_column(tmp_path: Path) -> None:
    # Regression test for the 3-column on-disk shape. Earlier revisions
    # wrote only {description, output} and lost listing_type between
    # the input and the label, which made the fine-tune dataset useless
    # for per-type filtering.
    w = LabelledFileWriter("batch_a", labelled_dir=tmp_path)
    w.emit(_row("abc1", "desc one", "Room"))
    w.emit(_row("abc2", "desc two", "Sale"))
    w.flush()

    rows = json.loads((tmp_path / "batch_a.json").read_text())
    assert all(set(r.keys()) == {"description", "listing_type", "output"} for r in rows)
    assert {r["listing_type"] for r in rows} == {"Room", "Sale"}


def test_writer_flush_is_idempotent(tmp_path: Path) -> None:
    # The pipeline flushes mid-run and again at shutdown; both writes
    # should produce the same bytes so git diffs don't churn.
    w = LabelledFileWriter("batch_b", labelled_dir=tmp_path)
    w.emit(_row("a", "first", "Rent"))
    w.flush()
    first = (tmp_path / "batch_b.json").read_bytes()
    w.flush()
    second = (tmp_path / "batch_b.json").read_bytes()
    assert first == second


def test_writer_resumes_from_existing_row_ids(tmp_path: Path) -> None:
    # First run writes one row, then the process "dies". A new writer
    # on the same basename must treat that row_id as already emitted.
    w1 = LabelledFileWriter("batch_c", labelled_dir=tmp_path)
    w1.emit(_row("rid-1", "desc"))
    w1.flush()

    w2 = LabelledFileWriter("batch_c", labelled_dir=tmp_path)
    assert w2.already_emitted("rid-1")
    assert not w2.already_emitted("rid-999")


def test_writer_tolerates_legacy_2_column_batch_on_resume(tmp_path: Path) -> None:
    # Pre-listing_type batch files exist in the wild (early labelling
    # runs before we added the column). On resume, the writer should
    # load them without crashing and preserve them in the next flush
    # — the missing field becomes "" until the row is re-labelled.
    out = tmp_path / "legacy.json"
    out.write_text(
        json.dumps(
            [{"description": "old desc", "output": "```json\n{}\n```"}],
            indent=2,
        )
    )
    w = LabelledFileWriter("legacy", labelled_dir=tmp_path)
    w.emit(_row("new-rid", "fresh desc", "Room"))
    w.flush()

    rows = json.loads(out.read_text())
    assert len(rows) == 2
    # Legacy row kept but with an empty listing_type.
    legacy = next(r for r in rows if r["description"] == "old desc")
    assert legacy["listing_type"] == ""


# ---------------------------------------------------------------------------
# regenerate_index — dedup, type counting.
# ---------------------------------------------------------------------------


def test_regenerate_index_dedups_by_description_prefix(tmp_path: Path) -> None:
    # Two batches, same description -> kept once. Matches the dedup
    # rule in test_set.py so train + eval see the same row exactly once.
    b1 = [{"description": "a shared desc" + " x" * 50, "listing_type": "Rent", "output": "{}"}]
    b2 = [
        {"description": "a shared desc" + " x" * 50, "listing_type": "Rent", "output": "{}"},
        {"description": "different desc entirely", "listing_type": "Room", "output": "{}"},
    ]
    (tmp_path / "b1.json").write_text(json.dumps(b1))
    (tmp_path / "b2.json").write_text(json.dumps(b2))

    idx = regenerate_index(tmp_path)
    assert idx["n_rows"] == 2
    assert idx["dedup_drops"] == 1
    assert idx["listing_type_counts"] == {"Rent": 1, "Room": 1}


def test_regenerate_index_preserves_three_column_shape(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text(
        json.dumps([
            {"description": "d", "listing_type": "Sale", "output": "{}"},
        ])
    )
    idx = regenerate_index(tmp_path)
    row = idx["rows"][0]
    assert set(row.keys()) == {"description", "listing_type", "output"}


# ---------------------------------------------------------------------------
# ProgressReporter — event bookkeeping.
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_progress_reporter_counts_start_and_done() -> None:
    async def scenario():
        r = ProgressReporter(total=3, stream=io.StringIO())
        r.start()
        await r.on_row_start()
        await r.on_row_start()
        assert r.in_flight == 2
        await r.on_row_done(success=True)
        assert r.in_flight == 1 and r.done == 1 and r.drops == 0
        await r.on_row_done(success=False)
        assert r.drops == 1
    asyncio.run(scenario())


def test_progress_reporter_announces_cache_transitions_once() -> None:
    async def scenario():
        buf = io.StringIO()
        r = ProgressReporter(total=5, stream=buf)
        r.start()
        await r.on_cache_write(4402)
        await r.on_cache_write(4402)  # second call should be a no-op
        await r.on_cache_read(5938)
        await r.on_cache_read(5938)  # no-op
        out = buf.getvalue()
        assert out.count("cache: warming") == 1
        assert out.count("cache: HIT") == 1
    asyncio.run(scenario())


def test_progress_reporter_retry_flattens_multiline_reason() -> None:
    async def scenario():
        buf = io.StringIO()
        r = ProgressReporter(total=1, stream=buf)
        r.start()
        # Pydantic errors carry newlines; a naive f-string would break
        # the one-line-per-event log format.
        await r.on_retry("rid", 0, "1 validation error for Extraction\nrent.x\n  bad")
        line = buf.getvalue().strip()
        assert "\n" not in line
        assert "rent.x" in line
    asyncio.run(scenario())


def test_progress_reporter_throttle_logs_on_1_and_every_10() -> None:
    async def scenario():
        buf = io.StringIO()
        r = ProgressReporter(total=1, stream=buf)
        r.start()
        for _ in range(11):
            await r.on_throttle()
        # First throttle + the 10th should log; in total 2 lines.
        assert buf.getvalue().count("bedrock throttle") == 2
    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Rendering helpers — pure, trivial to test.
# ---------------------------------------------------------------------------


def test_fmt_duration_formats_minutes_and_hours() -> None:
    assert _fmt_duration(0) == "--:--"
    assert _fmt_duration(59) == "00:59"
    assert _fmt_duration(65) == "01:05"
    assert _fmt_duration(3725) == "1h02m05s"


def test_progress_bar_is_fixed_width() -> None:
    # At any percent, the bar is exactly `width` characters. Matters for
    # the `\r`-overwrite path — a wider bar would leave trailing chars
    # from the previous render.
    for pct in (0, 13, 50, 99, 100):
        assert len(_progress_bar(pct, width=20)) == 20


# ---------------------------------------------------------------------------
# teacher.use_cache toggle — the Haiku/Llama wire difference.
#
# Bedrock Meta Llama models reject `cachePoint` with a
# ValidationException. Anthropic + Nova accept it. We pin the wire-
# level shape so a future refactor can't silently turn caching back
# on for Llama (which would make `lp-benchmark run --runner
# bedrock-llama` fail on every row).
# ---------------------------------------------------------------------------


def test_sync_converse_includes_cache_point_when_enabled() -> None:
    from listing_parser.labelling.teacher import _sync_converse

    captured: dict = {}

    class _FakeClient:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
            }

    _sync_converse(
        _FakeClient(),
        "model-id",
        "sys prompt",
        ("user turn",),
        max_tokens=10,
        temperature=0.0,
        use_cache=True,
    )
    system_blocks = captured["system"]
    assert {"cachePoint": {"type": "default", "ttl": "1h"}} in system_blocks
    assert any(b.get("text") == "sys prompt" for b in system_blocks)


def test_sync_converse_omits_cache_point_when_disabled() -> None:
    from listing_parser.labelling.teacher import _sync_converse

    captured: dict = {}

    class _FakeClient:
        def converse(self, **kwargs):
            captured.update(kwargs)
            return {
                "output": {"message": {"content": [{"text": "ok"}]}},
                "usage": {"inputTokens": 1, "outputTokens": 1},
            }

    _sync_converse(
        _FakeClient(),
        "meta.llama3-1-8b-instruct-v1:0",
        "sys prompt",
        ("user turn",),
        max_tokens=10,
        temperature=0.0,
        use_cache=False,
    )
    system_blocks = captured["system"]
    # Plain block only; no cachePoint directive. Bedrock's Llama
    # integration raises ValidationException otherwise.
    assert system_blocks == [{"text": "sys prompt"}]
