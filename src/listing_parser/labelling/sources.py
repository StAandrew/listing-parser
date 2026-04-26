"""Read listing inputs from `data/listing_*.json`.

The labeller's view of a listing is minimal — three fields, two of
which come straight from the JSON file and one synthesised here so
resume logic can dedup and skip already-labelled rows:

    row_id         sha1(description)[:16] — stable across runs, doesn't
                   depend on file order; used as the join key between
                   inputs and labelled outputs.
    description    raw text passed to the teacher.
    listing_type   "Rent" | "Sale" | "Room" — from the JSON.

We deliberately avoid using the listing index in the JSON file as the
id: those indices shift when the source file is regenerated, which
would break resumes. Hashing the description is stable as long as the
listing text itself is stable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

ListingType = Literal["Rent", "Sale", "Room"]


@dataclass(frozen=True)
class ListingInput:
    """A single row queued for labelling."""

    row_id: str
    description: str
    listing_type: ListingType


def _row_id(description: str) -> str:
    """Stable, description-only id. Collisions are theoretically possible
    at 16 hex chars but in practice vanishingly unlikely for <100k rows.
    """
    return hashlib.sha1(description.encode("utf-8")).hexdigest()[:16]


def load_listings(path: Path) -> Iterator[ListingInput]:
    """Yield `ListingInput` rows from a `data/listing_*.json` file.

    Accepts both the nested `{"listing": [...]}` shape produced by the
    upstream tool and a bare `[...]` list, since hand-authored batches
    might use either.

    Rows with blank descriptions or an invalid `listing_type` are
    skipped silently — the labeller logs a count of these separately
    so we never silently label garbage.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["listing"] if isinstance(data, dict) else data
    seen: set[str] = set()
    for row in rows:
        desc = (row.get("description") or "").strip()
        lt = row.get("listing_type")
        if not desc or lt not in ("Rent", "Sale", "Room"):
            continue
        rid = _row_id(desc)
        if rid in seen:
            # Duplicate inside the same input file — we skip here so the
            # teacher isn't invoked twice for the same description. The
            # labeller's sink keeps its own dedup based on row_id, but
            # catching this earlier saves API calls.
            continue
        seen.add(rid)
        yield ListingInput(row_id=rid, description=desc, listing_type=lt)


def count_by_type(inputs: list[ListingInput]) -> dict[str, int]:
    """Useful for logging the input distribution before the run starts."""
    counts: dict[str, int] = {"Rent": 0, "Sale": 0, "Room": 0}
    for row in inputs:
        counts[row.listing_type] = counts.get(row.listing_type, 0) + 1
    return counts
