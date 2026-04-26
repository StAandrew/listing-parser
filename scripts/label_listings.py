"""CLI entry point for the labelling pipeline.

    python scripts/label_listings.py \\
        --input data/listing_202604261232.json \\
        --concurrency 8

Loads one input JSON, labels every row via Haiku on Bedrock, and writes
`data/labelled/<basename>.json` plus a regenerated `_index.json`.

Resume semantics: if the output file already exists, rows whose
deterministic `row_id` (sha1 of description) is already present in the
sidecar row_id file are skipped. Labelling the same input file twice
is therefore cheap — only new rows get sent to the teacher.

The CLI handles only argument parsing, `.env` loading, and printing
stats. Everything else lives in `listing_parser.labelling`.

Example output after a run of ~1600 listings:
flushed chunk 52 (cumulative 1558 ok) rate=0.2 rows/s eta=--:--                                                         
  [########################] 1635/1635 (100.0%) in_flight=0 retries=2 drops=77 throttles=2620 rate=0.2/s eta=--:--

  inputs           : 1734
  resumed (skipped): 99
  success          : 1558
  failed           : 77
  elapsed          : 9356.6s

  outcome counts:
              fail_teacher_error: 77
                   success_clean: 1022
            success_with_repairs: 536

  cleaning repairs applied (top 15):
     117  drop_ungrounded_evidence:baths
      47  drop_ungrounded_evidence:living_rooms
      43  drop_ungrounded_evidence:beds
      41  drop_out_of_vocab:amenities_outdoor:Garage
      35  drop_ungrounded_evidence:rent.bills_included
      26  drop_ungrounded_evidence:floors_occupied.0
      24  relocate_amenity:amenities_interior->amenities_appliances
      21  drop_ungrounded_evidence:total_floors
      18  drop_ungrounded_evidence:property_type
      17  drop_unknown_item_key:nearby_mentions.cycle_minutes
      15  drop_ungrounded_evidence:rent.furnish_type
      14  drop_out_of_vocab:amenities_outdoor:Private driveway
      14  relocate_amenity:amenities_facilities->amenities_outdoor
      13  drop_ungrounded_evidence:amenities_outdoor.Private Garden
      10  drop_ungrounded_evidence:flags.is_auction

  token usage:
    input       : 562,017
    output      : 695,604
    cache read  : 9,263,280
    cache write : 0

wrote data/labelled/202604261415.json
regenerated data/labelled/_index.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from listing_parser.labelling.pipeline import format_stats, run
from listing_parser.labelling.sources import count_by_type, load_listings
from listing_parser.labelling.teacher import DEFAULT_MODEL_ID, DEFAULT_REGION


def _load_env_file(path: Path) -> None:
    """Minimal .env loader — same logic as `clean_hf_labels.py`."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to a data/listing_*.json file (or a plain JSON list of listings)",
    )
    p.add_argument(
        "--labelled-dir",
        type=Path,
        default=Path("data/labelled"),
        help="output directory for labelled batches + index (default: data/labelled)",
    )
    p.add_argument(
        "--basename",
        default=None,
        help="output file basename (default: derived from --input filename)",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--profile", default=os.environ.get("AWS_PROFILE"))
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N inputs (for smoke tests)",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=32,
        help="flush labelled output to disk every N rows (default: 32)",
    )
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="disable the live status line (useful when piping stderr to a log)",
    )
    args = p.parse_args(argv)

    _load_env_file(args.env_file)

    if args.basename:
        basename = args.basename
    else:
        # Strip the "listing_" prefix if present so outputs read cleanly:
        # listing_202604261232.json -> 202604261232.json.
        stem = args.input.stem
        basename = stem.removeprefix("listing_") or stem

    inputs = list(load_listings(args.input))
    if args.limit is not None:
        inputs = inputs[: args.limit]

    print(f"loaded {len(inputs)} inputs from {args.input}", file=sys.stderr)
    print(f"  listing_type histogram: {count_by_type(inputs)}", file=sys.stderr)
    print(
        f"  labelling with {args.model_id} via region={args.region} "
        f"profile={args.profile or '(default)'}",
        file=sys.stderr,
    )
    print(
        f"  concurrency={args.concurrency} max_retries={args.max_retries} "
        f"flush_every={args.flush_every}",
        file=sys.stderr,
    )

    if not inputs:
        print("no inputs to label — nothing to do.", file=sys.stderr)
        return 0

    stats = run(
        inputs,
        basename=basename,
        labelled_dir=args.labelled_dir,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        profile=args.profile,
        region=args.region,
        model_id=args.model_id,
        flush_every=args.flush_every,
        show_progress=not args.no_progress,
    )

    print(format_stats(stats), file=sys.stderr)
    print(
        f"\nwrote {args.labelled_dir / (basename + '.json')}",
        file=sys.stderr,
    )
    print(
        f"regenerated {args.labelled_dir / '_index.json'}",
        file=sys.stderr,
    )
    return 0 if stats.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
