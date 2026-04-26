"""`lp-benchmark` CLI — score a predictions file and render a markdown report.

This is the read-only reporting layer. It does NOT call any LLMs; it
only consumes:
  - a gold JSONL (produced by `test_set.py`)
  - a predictions JSONL (produced by a runner — Ollama/Anthropic/Bedrock,
    written in a later PR)

and produces:
  - stdout: a markdown summary fit for pasting into a PR / notes
  - optional: a `report.json` with the raw aggregate numbers
  - optional: a `report.md` with the same content that went to stdout

Run:
    lp-benchmark score \\
        --gold benchmarks/test_set.jsonl \\
        --predictions benchmarks/runs/ollama-llama31-8b-q6/predictions.jsonl \\
        --out-dir benchmarks/runs/ollama-llama31-8b-q6
"""

from __future__ import annotations

import argparse
import json
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

    ps = sub.add_parser("score", help="score predictions against gold")
    ps.add_argument("--gold", required=True)
    ps.add_argument("--predictions", required=True)
    ps.add_argument("--out-dir", default=None, help="write report.md + report.json here")
    ps.add_argument("--name", default=None, help="run name for the report header")
    ps.set_defaults(func=cmd_score)

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
