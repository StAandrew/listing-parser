"""`lp-benchmark` CLI — run a runner, score predictions, render a report.

Three subcommands:

  * `lp-benchmark run`    — invoke a runner on the gold set and write
                            `predictions.jsonl`. Calls an LLM.
  * `lp-benchmark score`  — read predictions + gold, write `report.md`
                            and `report.json`. Pure / offline.
  * `lp-benchmark smoke`  — score gold-vs-gold; sanity-check the scorer.

The split is deliberate: `score` has zero runtime dependencies beyond
`pydantic` and the stdlib, so it can run on any machine (including CI)
without AWS credentials. `run` is the only place that imports boto3 /
Ollama / etc., and it's lazy-loaded inside the subcommand so `score`
and `smoke` don't pay the boto3 import cost.

Typical flow:

    # 1. Generate predictions from a provider.
    lp-benchmark run \\
        --runner bedrock-haiku \\
        --gold benchmarks/test_set.jsonl \\
        --out-dir benchmarks/runs/haiku-4.5-teacher \\
        --concurrency 4

    # 2. Score them against the gold.
    lp-benchmark score \\
        --gold benchmarks/test_set.jsonl \\
        --predictions benchmarks/runs/haiku-4.5-teacher/predictions.jsonl \\
        --out-dir benchmarks/runs/haiku-4.5-teacher \\
        --name "haiku-4.5 (teacher)"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from listing_parser.benchmarks.scorer import AggregateReport, score_run


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _fmt(x: float, width: int = 5) -> str:
    return f"{x:>{width}.3f}"


def render_markdown(run_name: str, report: AggregateReport) -> str:
    d = report.to_dict()
    lines: list[str] = []
    lines.append(f"# Benchmark: {run_name}")
    lines.append("")
    lines.append("## Headlines")
    lines.append("")
    lines.append(f"- **Examples:** {d['n']}")
    lines.append(f"- **JSON parse rate:** {_fmt_pct(d['parse_rate'])}")
    lines.append(f"- **Schema validity rate:** {_fmt_pct(d['schema_rate'])}")
    lines.append(f"- **Macro field accuracy:** {_fmt_pct(d['macro_accuracy'])}")
    lines.append(
        f"- **Evidence grounding rate:** {_fmt_pct(d['evidence_grounding_rate'])}"
    )
    lines.append("")
    lines.append("## Per-field")
    lines.append("")
    lines.append(
        "| field | n | cov | acc | prec | rec | F1 | set_F1 | vocab! | case! |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )

    # Sort by lowest accuracy first — failures float to the top.
    rows = sorted(d["by_field"].items(), key=lambda kv: kv[1]["accuracy"])
    for path, stats in rows:
        set_f1 = stats.get("set_f1")
        set_f1_str = _fmt(set_f1) if set_f1 is not None else "   — "
        lines.append(
            "| `{path}` | {n:>3} | {cov} | {acc} | {prec} | {rec} | {f1} | {sf} | {vv:>3} | {cm:>3} |".format(
                path=path,
                n=int(stats["n"]),
                cov=_fmt(stats["coverage"]),
                acc=_fmt(stats["accuracy"]),
                prec=_fmt(stats["precision"]),
                rec=_fmt(stats["recall"]),
                f1=_fmt(stats["f1"]),
                sf=set_f1_str,
                vv=int(stats["vocab_violations"]),
                cm=int(stats["case_mismatches"]),
            )
        )

    lines.append("")
    lines.append(
        "Notes: `cov` = fraction of gold rows with a value present; `acc` = "
        "exact-match over all rows (tn included); `F1` = field-presence F1 "
        "(treats partial-value mismatches as errors); `set_F1` = member-level "
        "F1 for list fields; `vocab!` = predictions outside the controlled "
        "vocabulary; `case!` = predictions that only differ in case."
    )
    return "\n".join(lines) + "\n"


def cmd_score(args: argparse.Namespace) -> int:
    gold = Path(args.gold)
    preds = Path(args.predictions)
    if not gold.exists():
        print(f"gold file not found: {gold}", file=sys.stderr)
        return 2
    if not preds.exists():
        print(f"predictions file not found: {preds}", file=sys.stderr)
        return 2

    report = score_run(gold, preds)
    run_name = args.name or preds.parent.name
    md = render_markdown(run_name, report)
    print(md)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(md, encoding="utf-8")
        (out / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n(wrote {out / 'report.md'} and {out / 'report.json'})", file=sys.stderr)
    return 0


def _load_env_file(path: Path) -> None:
    """Minimal .env loader — mirrors `scripts/label_listings.py` so the
    same credentials file works for both labelling and benchmarking."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Runner registry.
#
# A tiny lookup so the CLI can pick a runner by short slug without
# hard-coding imports at module scope. This keeps `score` / `smoke`
# from paying the boto3 import cost when they don't need it.
# ---------------------------------------------------------------------------


def _build_bedrock_haiku(args: argparse.Namespace):
    # Deferred import so `score` / `smoke` stay boto3-free.
    from listing_parser.runners.bedrock import BedrockHaikuRunner  # noqa: PLC0415

    return BedrockHaikuRunner(
        name=args.name_slug,
        profile=args.profile,
        region=args.region,
        model_id=args.model_id,
        concurrency=args.concurrency,
        max_parse_retries=args.max_parse_retries,
    )


def _build_bedrock_llama(args: argparse.Namespace):
    from listing_parser.runners.bedrock import BedrockLlamaRunner  # noqa: PLC0415

    return BedrockLlamaRunner(
        name=args.name_slug,
        profile=args.profile,
        region=args.region,
        model_id=args.model_id,
        concurrency=args.concurrency,
        max_parse_retries=args.max_parse_retries,
    )


def _build_bedrock_ft(args: argparse.Namespace):
    from listing_parser.runners.bedrock import BedrockFineTuneRunner  # noqa: PLC0415

    return BedrockFineTuneRunner(
        name=args.name_slug,
        profile=args.profile,
        region=args.region,
        model_id=args.model_id,
        concurrency=args.concurrency,
        max_parse_retries=args.max_parse_retries,
    )


_RUNNER_BUILDERS: dict[str, Any] = {
    "bedrock-haiku": _build_bedrock_haiku,
    "bedrock-llama": _build_bedrock_llama,
    "bedrock-ft": _build_bedrock_ft,
}


def cmd_run(args: argparse.Namespace) -> int:
    """Invoke a runner on the gold test set and write predictions.jsonl.

    Load order: AWS creds / HF token come from `.env`, mirroring the
    labelling CLI. Both CLIs touch the same credentials and it saves
    users from threading env vars through the shell.
    """
    from listing_parser.runners.pipeline import (  # noqa: PLC0415
        format_stats,
        run_predictions,
    )

    _load_env_file(args.env_file)
    # If --profile wasn't explicitly set, fall back to AWS_PROFILE so
    # `AWS_PROFILE=foo lp-benchmark run ...` works the same as with
    # `scripts/label_listings.py`.
    if args.profile is None:
        args.profile = os.environ.get("AWS_PROFILE")

    gold = Path(args.gold)
    if not gold.exists():
        print(f"gold file not found: {gold}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    builder = _RUNNER_BUILDERS.get(args.runner)
    if builder is None:
        print(
            f"unknown runner: {args.runner!r}; available: "
            f"{sorted(_RUNNER_BUILDERS)}",
            file=sys.stderr,
        )
        return 2
    runner = builder(args)

    print(
        f"runner={runner.name} gold={gold} out={out_dir} "
        f"concurrency={args.concurrency} resume={not args.no_resume}",
        file=sys.stderr,
    )

    stats = run_predictions(
        runner,
        gold,
        out_dir,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        show_progress=not args.no_progress,
    )
    print(format_stats(stats, runner.name), file=sys.stderr)
    print(
        f"\nwrote {out_dir / 'predictions.jsonl'}\n"
        f"next: lp-benchmark score --gold {gold} "
        f"--predictions {out_dir / 'predictions.jsonl'} --out-dir {out_dir}",
        file=sys.stderr,
    )
    return 0 if stats.n_parse_failed == 0 else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    """Score the gold against itself. Expect parse_rate = schema_rate = macro_accuracy = 1.0.

    Sanity check that the scorer isn't broken in an obvious way. If this
    doesn't return 1.0 across the board, there's a bug in the comparators.
    """
    gold = Path(args.gold)
    if not gold.exists():
        print(f"gold file not found: {gold}", file=sys.stderr)
        return 2

    with gold.open(encoding="utf-8") as f:
        gold_rows = [json.loads(line) for line in f if line.strip()]

    preds_path = gold.with_suffix(gold.suffix + ".self_preds.jsonl")
    with preds_path.open("w", encoding="utf-8") as f:
        for row in gold_rows:
            f.write(
                json.dumps(
                    {"row_index": row["row_index"], "pred": row["output"]},
                    ensure_ascii=False,
                )
                + "\n"
            )

    try:
        report = score_run(gold, preds_path)
    finally:
        preds_path.unlink(missing_ok=True)

    d: dict[str, Any] = report.to_dict()
    ok = (
        d["parse_rate"] == 1.0
        and d["schema_rate"] == 1.0
        and d["macro_accuracy"] > 0.999
    )
    print(f"parse_rate     = {d['parse_rate']:.4f}")
    print(f"schema_rate    = {d['schema_rate']:.4f}")
    print(f"macro_accuracy = {d['macro_accuracy']:.4f}")
    print(f"grounding      = {d['evidence_grounding_rate']:.4f}")
    print("\nOK" if ok else "\nFAIL: scorer should produce 1.0 on gold-vs-gold")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lp-benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run ---------------------------------------------------------
    pr = sub.add_parser(
        "run",
        help="invoke a runner on the gold set and write predictions.jsonl",
    )
    pr.add_argument(
        "--runner",
        default="bedrock-haiku",
        choices=sorted(_RUNNER_BUILDERS),
        help="which runner to invoke (default: bedrock-haiku)",
    )
    pr.add_argument("--gold", required=True)
    pr.add_argument(
        "--out-dir",
        required=True,
        help="output directory (one per run); predictions.jsonl + sidecars land here",
    )
    pr.add_argument(
        "--name-slug",
        default=None,
        help="override the runner's `name` attribute (appears in the report header)",
    )
    pr.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="max in-flight requests (default: 4 — avoids Bedrock throttles on default quotas)",
    )
    pr.add_argument(
        "--max-parse-retries",
        type=int,
        default=1,
        help=(
            "retries on JSON parse failure (default: 1). "
            "Set to 0 to measure first-pass rate honestly."
        ),
    )
    pr.add_argument(
        "--profile",
        default=None,
        help="AWS profile (default: $AWS_PROFILE)",
    )
    pr.add_argument(
        "--region",
        default=None,
        help=(
            "AWS region override. Default is runner-specific "
            "(haiku=eu-west-2, llama=us-west-2)."
        ),
    )
    pr.add_argument(
        "--model-id",
        default=None,
        help=(
            "Bedrock model id / inference profile. Default is runner-specific "
            "(haiku=global.anthropic.claude-haiku-4-5-20251001-v1:0, "
            "llama=meta.llama3-1-8b-instruct-v1:0)."
        ),
    )
    pr.add_argument("--env-file", type=Path, default=Path(".env"))
    pr.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore the _row_ids.txt sidecar and re-run every row",
    )
    pr.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the live status line (useful when piping stderr to a log)",
    )
    pr.set_defaults(func=cmd_run)

    # ---- score -------------------------------------------------------
    ps = sub.add_parser("score", help="score predictions against gold")
    ps.add_argument("--gold", required=True)
    ps.add_argument("--predictions", required=True)
    ps.add_argument("--out-dir", default=None, help="write report.md + report.json here")
    ps.add_argument("--name", default=None, help="run name for the report header")
    ps.set_defaults(func=cmd_score)

    # ---- smoke -------------------------------------------------------
    pk = sub.add_parser(
        "smoke", help="score gold-vs-gold; expect 1.0 on every metric"
    )
    pk.add_argument("--gold", required=True)
    pk.set_defaults(func=cmd_smoke)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
