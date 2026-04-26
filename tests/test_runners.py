"""Tests for the runner layer.

We avoid hitting Bedrock here — the Haiku runner is covered by hand-
rolled fakes that implement the `Runner` protocol. The pipeline
(concurrency, resume, writer) is the interesting thing to exercise;
the thin `BedrockHaikuRunner` wrapper around `teacher.call_teacher`
would only repeat what `tests/test_labelling.py` already covers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from listing_parser.runners._output import ParseAttempt, parse_with_retry
from listing_parser.runners.base import RunnerResult
from listing_parser.runners.pipeline import (
    _load_gold,
    _load_resume_ids,
    _PredictionsWriter,
    run_predictions,
)

# ---------------------------------------------------------------------------
# Test fakes
# ---------------------------------------------------------------------------


class _FakeRunner:
    """Deterministic stand-in for a real provider.

    Yields results from a pre-seeded dict keyed by row_index. Unset
    indices return a generic success; tests that want to exercise
    failure paths seed explicit entries.
    """

    name = "fake"

    def __init__(self, by_row: dict[int, RunnerResult] | None = None) -> None:
        self._by_row = by_row or {}
        self.calls: list[tuple[str, str]] = []

    async def predict(self, description: str, listing_type: str) -> RunnerResult:
        self.calls.append((description, listing_type))
        # row_index isn't passed into predict() — it's a runner-facing
        # contract that the pipeline owns. For test purposes, we encode
        # it into the description ("row-42:…") and parse it back out.
        key = None
        if description.startswith("row-"):
            try:
                key = int(description.split(":", 1)[0].removeprefix("row-"))
            except ValueError:
                key = None
        if key is not None and key in self._by_row:
            return self._by_row[key]
        return RunnerResult(
            pred={"property_type": "Flat"},
            raw='```json\n{"property_type": "Flat"}\n```',
            error=None,
            usage={"input_tokens": 5, "output_tokens": 3},
        )


def _write_gold(path: Path, row_indexes: list[int]) -> None:
    """Write a minimal gold JSONL that the pipeline can consume."""
    with path.open("w", encoding="utf-8") as f:
        for i in row_indexes:
            f.write(
                json.dumps(
                    {
                        "row_index": i,
                        "description": f"row-{i}: a property for rent",
                        "output": {"property_type": "Flat"},
                        "listing_type": "Rent",
                    }
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# parse_with_retry — correction prompt wiring
# ---------------------------------------------------------------------------


def test_parse_with_retry_succeeds_on_first_attempt() -> None:
    """Happy path — one call, parseable result. No retry attempt made."""
    async def call(turns):
        return ParseAttempt(raw="{}", pred={}, usage=None)

    async def scenario():
        return await parse_with_retry(call, "user msg")

    pred, raw, err, _ = asyncio.run(scenario())
    assert pred == {}
    assert raw == "{}"
    assert err is None


def test_parse_with_retry_sends_correction_on_parse_failure() -> None:
    """Unparseable first attempt -> correction user-turn appended on second call.

    The correction prompt carries the raw previous response plus the
    synthetic "model output was not a parseable JSON object" error,
    which is what the labelling pipeline uses too.
    """
    attempts: list[tuple[str, ...]] = []

    async def call(turns):
        attempts.append(turns)
        if len(attempts) == 1:
            return ParseAttempt(raw="not json", pred=None, usage=None)
        return ParseAttempt(raw='{"ok": true}', pred={"ok": True}, usage=None)

    async def scenario():
        return await parse_with_retry(call, "user msg")

    pred, _, err, _ = asyncio.run(scenario())
    assert pred == {"ok": True}
    assert err is None
    assert len(attempts) == 2
    # Second attempt has 2 user turns; first has 1. Correction message
    # is in the second turn and mentions the original raw output.
    assert len(attempts[0]) == 1
    assert len(attempts[1]) == 2
    assert "not json" in attempts[1][1]


def test_parse_with_retry_gives_up_after_max() -> None:
    async def call(turns):
        return ParseAttempt(raw="still not json", pred=None, usage=None)

    async def scenario():
        return await parse_with_retry(call, "u", max_parse_retries=1)

    pred, raw, err, _ = asyncio.run(scenario())
    assert pred is None
    assert raw == "still not json"
    assert err is not None and "parse failed" in err


def test_parse_with_retry_does_not_retry_on_runtime_error() -> None:
    """A runtime exception (auth failure, network) is NOT retried — it's
    the caller's job to decide whether to re-run the whole command.
    """
    calls = 0

    async def call(turns):
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    async def scenario():
        return await parse_with_retry(call, "u", max_parse_retries=3)

    pred, _, err, _ = asyncio.run(scenario())
    assert pred is None
    assert err is not None and "RuntimeError" in err
    assert calls == 1


# ---------------------------------------------------------------------------
# _PredictionsWriter — append-only, sidecar invariants
# ---------------------------------------------------------------------------


def test_writer_appends_row_indexes_to_sidecar(tmp_path: Path) -> None:
    w = _PredictionsWriter(tmp_path)
    w.write_prediction(1, {"x": 1}, "raw1", None)
    w.write_prediction(2, None, "raw2", "parse failed")
    w.flush()
    w.close()

    sidecar = (tmp_path / "_row_ids.txt").read_text().splitlines()
    assert sidecar == ["1", "2"]

    preds = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text().splitlines()
    ]
    assert preds[0] == {"row_index": 1, "pred": {"x": 1}, "raw": "raw1"}
    # Errors are carried so `lp-benchmark score` can still join by row_index
    # and the log has enough info to debug later.
    assert preds[1]["error"] == "parse failed"
    assert preds[1]["pred"] is None


def test_load_resume_ids_tolerates_blank_and_garbage_lines(tmp_path: Path) -> None:
    p = tmp_path / "_row_ids.txt"
    p.write_text("1\n\n2\nnotanumber\n3\n")
    assert _load_resume_ids(p) == {1, 2, 3}


# ---------------------------------------------------------------------------
# run_predictions — end-to-end with a fake runner
# ---------------------------------------------------------------------------


def test_run_predictions_writes_predictions_jsonl_and_sidecar(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    _write_gold(gold, [10, 11, 12])
    out_dir = tmp_path / "run"

    runner = _FakeRunner()
    stats = run_predictions(
        runner,
        gold,
        out_dir,
        concurrency=2,
        resume=True,
        flush_every=2,
        show_progress=False,
    )

    assert stats.n_gold == 3
    assert stats.n_skipped_resumed == 0
    assert stats.n_predicted == 3
    assert stats.n_parse_failed == 0

    preds = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert {p["row_index"] for p in preds} == {10, 11, 12}
    assert all("pred" in p and "raw" in p for p in preds)

    sidecar = (out_dir / "_row_ids.txt").read_text().splitlines()
    assert set(sidecar) == {"10", "11", "12"}


def test_run_predictions_resume_skips_completed_rows(tmp_path: Path) -> None:
    # First run processes rows [1,2]; second run with gold [1,2,3,4]
    # should skip [1,2] and only call the runner for [3,4].
    gold1 = tmp_path / "gold1.jsonl"
    _write_gold(gold1, [1, 2])
    out_dir = tmp_path / "run"

    r1 = _FakeRunner()
    run_predictions(r1, gold1, out_dir, concurrency=1, flush_every=4, show_progress=False)
    assert len(r1.calls) == 2

    gold2 = tmp_path / "gold2.jsonl"
    _write_gold(gold2, [1, 2, 3, 4])
    r2 = _FakeRunner()
    stats = run_predictions(
        r2, gold2, out_dir, concurrency=1, flush_every=4, show_progress=False
    )

    assert stats.n_skipped_resumed == 2
    assert stats.n_predicted == 2
    assert len(r2.calls) == 2
    called_rows = sorted(int(desc.split(":", 1)[0].removeprefix("row-")) for desc, _ in r2.calls)
    assert called_rows == [3, 4]

    # Predictions file is append-only: 4 total lines, one per row_index.
    preds = [
        json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
    ]
    assert {p["row_index"] for p in preds} == {1, 2, 3, 4}


def test_run_predictions_records_failure_rows(tmp_path: Path) -> None:
    # Seed a parse-failed result for row 5; the pipeline must still
    # write it (with pred=null) so the scorer's parse_rate metric has
    # all gold rows to measure against.
    gold = tmp_path / "gold.jsonl"
    _write_gold(gold, [5, 6])
    out_dir = tmp_path / "run"

    seeded = {
        5: RunnerResult(pred=None, raw="not json", error="parse failed after retries"),
    }
    runner = _FakeRunner(by_row=seeded)
    stats = run_predictions(
        runner, gold, out_dir, concurrency=1, flush_every=4, show_progress=False
    )

    assert stats.n_parse_failed == 1
    assert stats.n_predicted == 1

    preds = {
        json.loads(line)["row_index"]: json.loads(line)
        for line in (out_dir / "predictions.jsonl").read_text().splitlines()
    }
    assert preds[5]["pred"] is None
    assert preds[5]["error"] == "parse failed after retries"
    assert preds[6]["pred"] == {"property_type": "Flat"}


def test_run_predictions_handles_empty_gold(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"
    gold.write_text("")
    out_dir = tmp_path / "run"
    runner = _FakeRunner()
    stats = run_predictions(runner, gold, out_dir, show_progress=False)
    assert stats.n_gold == 0
    assert stats.n_predicted == 0
    assert (out_dir / "predictions.jsonl").exists()


# ---------------------------------------------------------------------------
# _load_gold — line tolerance
# ---------------------------------------------------------------------------


def test_load_gold_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "g.jsonl"
    p.write_text(
        json.dumps({"row_index": 1, "description": "x", "output": {}}) + "\n"
        "\n"
        + json.dumps({"row_index": 2, "description": "y", "output": {}}) + "\n"
    )
    rows = _load_gold(p)
    assert [r["row_index"] for r in rows] == [1, 2]


# ---------------------------------------------------------------------------
# CLI wiring — confirm `lp-benchmark run` is registered and dispatches
# correctly. We don't actually invoke the Bedrock runner (that would
# need AWS creds); we just assert the subparser is wired up and a
# missing gold file returns the usage exit code.
# ---------------------------------------------------------------------------


def test_cli_run_subcommand_exists() -> None:
    from listing_parser.benchmarks.cli import build_parser

    parser = build_parser()
    # parse_args will SystemExit(2) on an unknown subcommand; finding
    # `run` means it's registered.
    args = parser.parse_args([
        "run",
        "--runner", "bedrock-haiku",
        "--gold", "nonexistent-gold.jsonl",
        "--out-dir", "nonexistent-out",
    ])
    assert args.cmd == "run"
    assert args.runner == "bedrock-haiku"
    assert args.concurrency == 4  # default
    # Region / model-id default to None so runner-class defaults win.
    # This is what lets `--runner bedrock-llama` use us-west-2 without
    # a `--region us-west-2` flag.
    assert args.region is None
    assert args.model_id is None


def test_cli_run_knows_about_bedrock_llama() -> None:
    """Guard against silent registry regressions: if someone deletes
    `bedrock-llama` from `_RUNNER_BUILDERS` the baseline report in
    `benchmarks/runs/llama-3.1-8b-base/` becomes unreproducible.
    """
    from listing_parser.benchmarks.cli import _RUNNER_BUILDERS, build_parser

    assert "bedrock-llama" in _RUNNER_BUILDERS
    parser = build_parser()
    args = parser.parse_args([
        "run",
        "--runner", "bedrock-llama",
        "--gold", "g.jsonl",
        "--out-dir", "o",
    ])
    assert args.runner == "bedrock-llama"


def test_cli_run_returns_2_on_missing_gold(tmp_path: Path, capsys) -> None:
    from listing_parser.benchmarks.cli import main

    rc = main([
        "run",
        "--runner", "bedrock-haiku",
        "--gold", str(tmp_path / "does-not-exist.jsonl"),
        "--out-dir", str(tmp_path / "run"),
        "--env-file", str(tmp_path / "no.env"),
    ])
    assert rc == 2
    err = capsys.readouterr().err
    assert "gold file not found" in err


# ---------------------------------------------------------------------------
# Bedrock runner defaults — guard against silent drift between the
# two provider configs. These don't hit the network; they only check
# the class-level defaults, which are what makes `lp-benchmark run
# --runner bedrock-llama` do the right thing without extra flags.
# ---------------------------------------------------------------------------


def test_bedrock_haiku_defaults_enable_caching_and_use_eu_west_2() -> None:
    from listing_parser.runners.bedrock import BedrockHaikuRunner

    r = BedrockHaikuRunner()
    assert r.name == "haiku-4.5-teacher"
    assert r._region == "eu-west-2"
    # Prompt caching is only supported on the `global.*` Anthropic
    # inference profiles; if someone switches to `eu.*` the cachePoint
    # directive silently becomes a no-op (confirmed empirically during
    # labelling-pipeline bring-up).
    assert r._model_id.startswith("global.anthropic.claude-haiku-4-5-")
    assert r._use_cache is True
    # Teacher runs at 0.2 to match labelling — so teacher-vs-gold is
    # apples-to-apples with the labels the teacher originally produced.
    assert r._temperature == 0.2


def test_bedrock_llama_defaults_disable_caching_and_use_us_west_2() -> None:
    from listing_parser.runners.bedrock import BedrockLlamaRunner

    r = BedrockLlamaRunner()
    assert r.name == "llama-3.1-8b-base"
    # Llama 3.1 8B ON_DEMAND is us-west-2 only — not eu-west-2 where
    # the teacher lives.
    assert r._region == "us-west-2"
    assert r._model_id == "meta.llama3-1-8b-instruct-v1:0"
    # Bedrock's Llama integration rejects `cachePoint` with
    # ValidationException, so caching MUST be off. Turning it back on
    # would break the whole run, not just silently fail.
    assert r._use_cache is False
    # Greedy decoding on the baseline — we want reproducible numbers,
    # not sampling-noise variance across re-runs.
    assert r._temperature == 0.0


def test_bedrock_runner_constructor_overrides_respected() -> None:
    """Every default is a kwarg; regression test so a subclass adding
    new defaults doesn't lose the per-instance override path.
    """
    from listing_parser.runners.bedrock import BedrockLlamaRunner

    r = BedrockLlamaRunner(
        name="custom",
        region="ap-southeast-1",
        model_id="meta.llama3-3-70b-instruct-v1:0",
        temperature=0.5,
        use_cache=True,
        max_parse_retries=3,
    )
    assert r.name == "custom"
    assert r._region == "ap-southeast-1"
    assert r._model_id == "meta.llama3-3-70b-instruct-v1:0"
    assert r._temperature == 0.5
    assert r._use_cache is True
    assert r._max_parse_retries == 3
