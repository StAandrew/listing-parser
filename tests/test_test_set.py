"""Smoke tests for the test-set JSON extraction helpers.

We don't test the HF download path (that'd require a network); we do
test the bits that handle the fence-stripping and listing-type inference
because they run over every labelled row.
"""

from __future__ import annotations

from listing_parser.benchmarks.test_set import (dedupe_by_description,
                                                infer_listing_type,
                                                parse_output_json,
                                                strip_json_fences)


def test_strip_fences_basic():
    raw = "```json\n{\"a\": 1}\n```"
    assert strip_json_fences(raw).strip() == '{"a": 1}'


def test_parse_output_tolerates_trailing_prose():
    raw = '```json\n{"a": 1}\n``` and some commentary the model added'
    assert parse_output_json(raw) == {"a": 1}


def test_parse_output_returns_none_on_garbage():
    assert parse_output_json("I cannot answer this") is None


def test_parse_output_handles_missing_fence():
    assert parse_output_json('  {"a": 1}  ') == {"a": 1}


def test_infer_listing_type_routes_correctly():
    assert infer_listing_type({"room": {"room_type": "Double"}}) == "Room"
    assert infer_listing_type({"sale": {"tenure_type": "Freehold"}}) == "Sale"
    assert infer_listing_type({"rent": {"furnish_type": "Furnished"}}) == "Rent"
    # Nothing populated -> default to Rent (most common, conservative).
    assert infer_listing_type({}) == "Rent"
    # Empty sub-blocks shouldn't win over absent ones.
    assert infer_listing_type({"room": {}, "rent": {"furnish_type": "F"}}) == "Rent"


def test_dedupe_keeps_first_occurrence():
    rows = [
        {"description": "Mitre Yard luxury apartments..."},
        {"description": "Mitre Yard luxury apartments..."},  # dupe
        {"description": "Different description here."},
    ]
    out = dedupe_by_description(rows)
    assert len(out) == 2
    assert out[0]["description"].startswith("Mitre Yard")
    assert out[1]["description"].startswith("Different")
