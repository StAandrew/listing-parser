"""Tests for the prompt assembly layer.

We don't test what the teacher will *do* with these prompts — that's
the labelling pipeline's job. We do check that the prompts are stable,
that they stay in sync with `schema.py`, and that they include the
specific reinforcements (closed-vocab block, listing-type gating) that
should close the gaps the original Haiku run exposed.
"""

from __future__ import annotations

import pytest

from listing_parser.prompting import (
    build_correction_message,
    build_system_prompt,
    build_user_message,
)
from listing_parser.schema import (
    AMENITIES_INTERIOR,
    PROPERTY_TYPES,
    ROOM_TYPES,
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_system_prompt_includes_rules_from_prompt_md():
    prompt = build_system_prompt()
    # Rules block text — a distinctive sentence only found in prompt.md.
    assert "UK property listing description parser" in prompt
    assert "Return JSON only" in prompt


def test_system_prompt_includes_full_output_schema():
    prompt = build_system_prompt()
    # A field that only appears in the schema block.
    assert "total_floor_area_sqm" in prompt
    assert "purchase_scheme" in prompt


def test_system_prompt_includes_amenity_vocab_as_json_array():
    prompt = build_system_prompt()
    # Every interior amenity should appear in the rendered vocab block.
    # Use one that would never match prose accidentally.
    assert '"Walk-in wardrobe"' in prompt
    # The block has a recognisable header so the labelling pipeline can
    # spot-check the prompt manually.
    assert "Closed-vocabulary enforcement" in prompt


def test_system_prompt_lists_every_amenity_interior_value():
    prompt = build_system_prompt()
    for amenity in AMENITIES_INTERIOR:
        assert amenity in prompt, f"missing amenity: {amenity}"


def test_system_prompt_lists_every_property_type():
    prompt = build_system_prompt()
    for prop in PROPERTY_TYPES:
        assert prop in prompt, f"missing property type: {prop}"


def test_system_prompt_lists_room_type_vocab():
    prompt = build_system_prompt()
    # room.room_type is the field path used in the vocab block.
    assert "room.room_type" in prompt
    for rt in ROOM_TYPES:
        assert rt in prompt


def test_system_prompt_has_listing_type_gating_block():
    prompt = build_system_prompt()
    assert "Listing-type gating" in prompt
    # All three rules named explicitly.
    assert "listing_type: Rent" in prompt
    assert "listing_type: Sale" in prompt
    assert "listing_type: Room" in prompt


def test_system_prompt_includes_few_shot_examples():
    # The worked-examples block gives the teacher a concrete pattern to
    # match AND keeps the prompt above Haiku 4.5's cache minimum
    # (measured ~4k tokens on 2026-04 against global.claude-haiku-4.5).
    # Below that threshold, Bedrock silently ignores `cachePoint`.
    prompt = build_system_prompt()
    assert "Worked examples" in prompt
    assert "Example 1" in prompt
    assert "Example 3" in prompt  # all three examples are included


def test_system_prompt_size_is_above_cache_minimum():
    # Rough char/token ratio is ~4:1 for English prose, so we want
    # >=16k chars to comfortably clear a 4k-token threshold with margin.
    prompt = build_system_prompt()
    assert len(prompt) >= 16_000, (
        f"system prompt is only {len(prompt)} chars; Haiku 4.5 needs ~4k "
        "tokens (~16k chars) before `cachePoint` activates"
    )


def test_system_prompt_is_deterministic():
    # Labelling-pipeline resumability assumes identical bytes for the
    # same inputs across runs. If someone introduces non-determinism
    # (e.g. shuffling vocab), this test fails loudly.
    assert build_system_prompt() == build_system_prompt()


# ---------------------------------------------------------------------------
# User message
# ---------------------------------------------------------------------------


def test_user_message_formats_listing_type_and_description():
    msg = build_user_message("A two-bed flat.", "Rent")
    assert msg == "listing_type: Rent\n---\nA two-bed flat."


def test_user_message_rejects_unknown_listing_type():
    with pytest.raises(ValueError):
        build_user_message("...", "Commercial")  # type: ignore[arg-type]


def test_user_message_preserves_description_whitespace_verbatim():
    # Descriptions sometimes carry meaningful whitespace (bulleted lists,
    # tab-indented sections). We should pass them through unchanged so
    # evidence quotes can ground on the same normalisation rules the
    # scorer uses.
    desc = "Line one.\n\n  - bullet\n  - bullet two\n"
    msg = build_user_message(desc, "Room")
    assert msg.endswith(desc)


# ---------------------------------------------------------------------------
# Correction message
# ---------------------------------------------------------------------------


def test_correction_message_includes_error_and_previous_output():
    prev_raw = '```json\n{"amenities_indoor": ["foo"]}\n```'
    err = "1 validation error for Extraction\namenities_indoor\n  Extra inputs are not permitted"
    corrected = build_correction_message(prev_raw, err)
    assert err.strip() in corrected
    assert prev_raw.strip() in corrected
    # The instruction to fix the specific error must be there — a vague
    # "please try again" would waste tokens.
    assert "corrected JSON" in corrected


def test_correction_message_trims_surrounding_whitespace():
    # Bedrock sometimes returns leading/trailing whitespace on raw output;
    # we don't want that pollution in the correction prompt.
    corrected = build_correction_message("   {}   ", "  oops  ")
    assert "{}\n\n" in corrected
    assert "oops\n\n" in corrected
