"""Pure, I/O-free cleanup + parsing of teacher-labelled extractions.

This module owns **every** transformation applied to a raw teacher
(Haiku) output before it's treated as a label. It's imported from:

  * the labelling pipeline (so fresh labels are cleaned at write time);
  * `scripts/clean_hf_labels.py` (the one-off HF migration);
  * the benchmark test-set builder (for parsing the fenced JSON that
    older HF rows still wrap).

Keeping all of this in one module is a deliberate design choice: the
training loader and the scorer must see the same label distribution,
and the cheapest way to guarantee that is to give them a single
function to call. If you find yourself about to inline a "small fix"
into a caller, extend `clean_extraction` instead.

The module is strictly pure:

  * no filesystem, no HTTP, no subprocess;
  * given the same `(obj, description)` inputs it always returns the
    same `(cleaned_obj, changes)` pair;
  * never raises — callers decide whether to keep or drop a row based
    on the returned changes and a subsequent `validate()` call.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from listing_parser.schema import (
    AMENITIES_APPLIANCES,
    AMENITIES_FACILITIES,
    AMENITIES_INTERIOR,
    AMENITIES_OUTDOOR,
    CommuteClaim,
    Extraction,
    FlagsBlock,
    NearbyMention,
    RentBlock,
    RoomBlock,
    SaleBlock,
)

__all__ = [
    "clean_extraction",
    "format_fenced_json",
    "parse_output_json",
    "strip_json_fences",
    "validate_extraction",
]


# ---------------------------------------------------------------------------
# Parsing helpers — tolerate Haiku's habit of wrapping JSON in ```json fences
# and occasionally trailing with prose after the closing brace.
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def strip_json_fences(text: str) -> str:
    """Remove ```json … ``` markdown fences if present, else no-op."""
    return _CODE_FENCE_RE.sub("", text).strip()


def parse_output_json(raw: str) -> dict[str, Any] | None:
    """Parse a fenced/partially-fenced JSON object; tolerate trailing prose.

    Returns `None` rather than raising on unparseable input — the caller
    usually wants to drop the row and continue rather than crash halfway
    through a batch.
    """
    cleaned = strip_json_fences(raw)
    if "{" not in cleaned:
        return None
    cleaned = cleaned[cleaned.index("{") :]
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def format_fenced_json(obj: dict[str, Any]) -> str:
    """Serialise an extraction the way the HF dataset stores it.

    Pretty-printed JSON wrapped in a ```json fence. Used by writers that
    need to stay byte-compatible with the existing HF rows and CSV
    exports downstream tooling might read directly.
    """
    return "```json\n" + json.dumps(obj, ensure_ascii=False, indent=2) + "\n```"


# ---------------------------------------------------------------------------
# Repair rules.
#
# `clean_extraction` is the single public entry point. Everything below it
# is implementation detail, but split into named helpers so each repair
# step is independently testable. Every rule records a short string in
# the `changes` counter; the labelling pipeline funnels these into a
# per-run log, and the scorer/CLI use them in their dry-run reports.
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    """Must mirror `scorer._norm_text`. Drifting the two would cause the
    scorer to count as "ungrounded" evidence that the cleaner accepted.

    Rules: lowercase, collapse whitespace, NBSP → space, en-dash → hyphen.
    """
    return _WS_RE.sub(" ", s.replace("\xa0", " ").replace("\u2013", "-").lower()).strip()


_TOP_LEVEL_RENAMES: dict[str, str] = {
    # Haiku occasionally writes `amenities_indoor` instead of
    # `amenities_interior`. Preserves the data, just fixes the key.
    "amenities_indoor": "amenities_interior",
}

_VALID_TOP_LEVEL: set[str] = set(Extraction.model_fields.keys())
_VALID_NEARBY_KEYS: set[str] = set(NearbyMention.model_fields.keys())
_VALID_COMMUTE_KEYS: set[str] = set(CommuteClaim.model_fields.keys())

_VALID_SUB_KEYS: dict[str, set[str]] = {
    "rent": set(RentBlock.model_fields.keys()),
    "sale": set(SaleBlock.model_fields.keys()),
    "room": set(RoomBlock.model_fields.keys()),
    "flags": set(FlagsBlock.model_fields.keys()),
}

_AMENITY_GROUPS: dict[str, tuple[str, ...]] = {
    "amenities_interior": AMENITIES_INTERIOR,
    "amenities_outdoor": AMENITIES_OUTDOOR,
    "amenities_facilities": AMENITIES_FACILITIES,
    "amenities_appliances": AMENITIES_APPLIANCES,
}


def _commute_mode_vocab() -> set[str]:
    """Pull the Literal members out of CommuteClaim.mode at import time.

    Avoids duplicating the list here and drifting it from schema.py.
    """
    ann = CommuteClaim.model_fields["mode"].annotation
    # annotation is `Literal[...] | None`; __args__[0] is the Literal.
    return set(ann.__args__[0].__args__)


_COMMUTE_MODE_VOCAB: set[str] = _commute_mode_vocab()


def _repair_top_level_keys(
    obj: dict[str, Any], changes: Counter[str]
) -> dict[str, Any]:
    """Step 1 — rename typo'd top-level keys, drop unknown ones.

    If a rename collides with an existing key (both `amenities_interior`
    and `amenities_indoor` present), merge list values rather than
    silently dropping one of them.
    """
    out: dict[str, Any] = {}
    for raw_key, value in obj.items():
        key = _TOP_LEVEL_RENAMES.get(raw_key, raw_key)
        if key != raw_key:
            changes[f"rename_top_level:{raw_key}->{key}"] += 1
        if key not in _VALID_TOP_LEVEL:
            changes[f"drop_unknown_top_level_key:{key}"] += 1
            continue
        if key in out and isinstance(out[key], list) and isinstance(value, list):
            value = out[key] + [v for v in value if v not in out[key]]
            changes[f"merge_after_rename:{key}"] += 1
        out[key] = value
    return out


def _repair_amenity_groups(out: dict[str, Any], changes: Counter[str]) -> None:
    """Step 2 — enforce amenity vocab per group, relocate mis-grouped values.

    Mutates `out` in place. An amenity found in the wrong group's list
    (e.g. "Washing machine" under `amenities_interior`) is moved to the
    correct group so we keep the fact and only fix the location. Items
    with no matching vocab anywhere are dropped.

    Empty groups after repair are removed entirely so the downstream
    scorer doesn't count an empty list as "model said there's nothing".
    """
    for field_name, vocab in _AMENITY_GROUPS.items():
        vals = out.get(field_name)
        if not isinstance(vals, list):
            continue
        keep: list[str] = []
        for v in vals:
            if not isinstance(v, str):
                changes[f"drop_non_string_amenity:{field_name}"] += 1
                continue
            if v in vocab:
                keep.append(v)
                continue
            relocated = False
            for other_field, other_vocab in _AMENITY_GROUPS.items():
                if other_field == field_name:
                    continue
                if v in other_vocab:
                    other_list = out.setdefault(other_field, [])
                    if isinstance(other_list, list) and v not in other_list:
                        other_list.append(v)
                    changes[f"relocate_amenity:{field_name}->{other_field}"] += 1
                    relocated = True
                    break
            if not relocated:
                changes[f"drop_out_of_vocab:{field_name}:{v}"] += 1
        # De-duplicate while preserving insertion order.
        seen: set[str] = set()
        deduped: list[str] = []
        for v in keep:
            if v in seen:
                changes[f"dedup_amenity:{field_name}"] += 1
                continue
            seen.add(v)
            deduped.append(v)
        if deduped:
            out[field_name] = deduped
        else:
            out.pop(field_name, None)


def _repair_nested_blocks(out: dict[str, Any], changes: Counter[str]) -> None:
    """Step 3 — drop unknown sub-keys from rent/sale/room/flags.

    If a block ends up empty after cleanup (every sub-key was bogus),
    drop the whole block rather than keep a `{}` which would look like
    "the model asserted a block applies but had nothing to say".
    """
    for block_name, valid_keys in _VALID_SUB_KEYS.items():
        block = out.get(block_name)
        if not isinstance(block, dict):
            continue
        cleaned_block: dict[str, Any] = {}
        for sub_key, sub_val in block.items():
            if sub_key in valid_keys:
                cleaned_block[sub_key] = sub_val
            else:
                changes[f"drop_unknown_sub_key:{block_name}.{sub_key}"] += 1
        if cleaned_block:
            out[block_name] = cleaned_block
        else:
            out.pop(block_name, None)


def _repair_list_dict_fields(out: dict[str, Any], changes: Counter[str]) -> None:
    """Step 4 — clean up nearby_mentions / commute_claims entries.

    * Drops unknown per-item keys (e.g. `mode` on a `NearbyMention`, which
      Haiku sometimes leaks from `commute_claims`).
    * For `commute_claims.mode`, drops the key if its value isn't in the
      Literal — locally-accurate-but-unmodelled values like "tram" or
      "DLR" show up in UK listings and would fail Pydantic outright if
      left in. Keeping the rest of the claim is more useful than losing
      the whole row.
    * Drops entries that lack the identifying field (`name` for
      nearby_mentions, `destination` for commute_claims).
    """
    for list_field, valid_keys in (
        ("nearby_mentions", _VALID_NEARBY_KEYS),
        ("commute_claims", _VALID_COMMUTE_KEYS),
    ):
        items = out.get(list_field)
        if not isinstance(items, list):
            continue
        cleaned_items: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                changes[f"drop_non_dict_item:{list_field}"] += 1
                continue
            cleaned_item: dict[str, Any] = {}
            for k, v in item.items():
                if k in valid_keys:
                    cleaned_item[k] = v
                else:
                    changes[f"drop_unknown_item_key:{list_field}.{k}"] += 1
            if list_field == "commute_claims" and "mode" in cleaned_item:
                mode_val = cleaned_item["mode"]
                if mode_val is not None and mode_val not in _COMMUTE_MODE_VOCAB:
                    changes[f"drop_out_of_vocab:commute_claims.mode:{mode_val}"] += 1
                    cleaned_item.pop("mode")
            id_key = "name" if list_field == "nearby_mentions" else "destination"
            if not cleaned_item.get(id_key):
                changes[f"drop_item_missing_id:{list_field}"] += 1
                continue
            cleaned_items.append(cleaned_item)
        if cleaned_items:
            out[list_field] = cleaned_items
        else:
            out.pop(list_field, None)


def _repair_evidence(
    out: dict[str, Any], description: str, changes: Counter[str]
) -> None:
    """Step 5 — evidence quotes must be substrings of the description.

    Anything else is a hallucination. Letting ungrounded quotes through
    would teach the fine-tune to produce them, and makes the scorer's
    `evidence_grounding_rate` meaningless as an evaluation signal.
    """
    evidence = out.get("evidence")
    if not isinstance(evidence, dict):
        return
    desc_norm = _norm_text(description)
    cleaned_evidence: dict[str, str] = {}
    for path, quote in evidence.items():
        if not isinstance(quote, str) or not quote.strip():
            changes["drop_empty_evidence"] += 1
            continue
        if _norm_text(quote) in desc_norm:
            cleaned_evidence[path] = quote
        else:
            changes[f"drop_ungrounded_evidence:{path}"] += 1
    if cleaned_evidence:
        out["evidence"] = cleaned_evidence
    else:
        out.pop("evidence", None)


def clean_extraction(
    obj: dict[str, Any],
    description: str,
    changes: Counter[str] | None = None,
) -> tuple[dict[str, Any], Counter[str]]:
    """Return a repaired copy of `obj` along with a tally of changes.

    `description` is used (only) to validate evidence quotes against. If
    you don't have a description at hand — e.g. when re-cleaning an
    already-cleaned row that you trust — pass an empty string; the
    evidence step becomes a no-op that drops every quote, which is
    almost certainly not what you want. Prefer re-loading the real
    description from the dataset when in doubt.

    The `changes` counter can be pre-populated by the caller to
    accumulate stats across many rows; if omitted, a fresh counter is
    created and returned.
    """
    if changes is None:
        changes = Counter()

    out = _repair_top_level_keys(obj, changes)
    _repair_amenity_groups(out, changes)
    _repair_nested_blocks(out, changes)
    _repair_list_dict_fields(out, changes)
    _repair_evidence(out, description, changes)

    return out, changes


def validate_extraction(obj: dict[str, Any]) -> str | None:
    """Run Pydantic validation; return a short error string or None on success.

    Thin wrapper so callers don't have to import Pydantic and format the
    error themselves. The returned string is suitable for threading back
    into a correction prompt for the teacher.
    """
    try:
        Extraction.model_validate(obj)
    except Exception as e:  # pydantic.ValidationError, but keep this loose
        return str(e)
    return None
