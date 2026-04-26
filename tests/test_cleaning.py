"""Tests for the cleaning rules.

These rules run at label-writing time (on every row Haiku produces) and
also as a one-off migration over the HF source. Because the training
loader and the scorer both trust their input to already be cleaned,
regressions here would silently poison either the training set or the
benchmark, so cover each repair explicitly.
"""

from __future__ import annotations

from collections import Counter

from listing_parser.cleaning import (
    clean_extraction,
    format_fenced_json,
    parse_output_json,
    strip_json_fences,
    validate_extraction,
)

# A minimal description used in most tests. Short and boring by design —
# we only care about the cleaning rules, not the description content.
DESC = "Two-bedroom flat with balcony and gas heating. Close to the station."


# ---------------------------------------------------------------------------
# Parsing / formatting
# ---------------------------------------------------------------------------


def test_strip_json_fences_removes_code_fences():
    assert strip_json_fences("```json\n{\"a\": 1}\n```").strip() == '{"a": 1}'
    assert strip_json_fences("  {\"a\": 1}  ").strip() == '{"a": 1}'


def test_parse_output_json_tolerates_trailing_prose():
    # raw_decode should stop at the first valid JSON object.
    raw = '```json\n{"a": 1}\n``` — plus prose the model added'
    assert parse_output_json(raw) == {"a": 1}


def test_parse_output_json_returns_none_on_garbage():
    assert parse_output_json("I cannot answer this query") is None
    # Missing opening brace entirely.
    assert parse_output_json("") is None


def test_parse_output_json_rejects_non_objects():
    # A valid JSON array isn't a valid extraction — callers expect a dict.
    assert parse_output_json("[1, 2, 3]") is None


def test_format_fenced_json_round_trips():
    obj = {"property_type": "Flat", "beds": 2}
    fenced = format_fenced_json(obj)
    assert fenced.startswith("```json\n") and fenced.endswith("\n```")
    assert parse_output_json(fenced) == obj


# ---------------------------------------------------------------------------
# Top-level key repairs
# ---------------------------------------------------------------------------


def test_typo_amenities_indoor_is_renamed_to_amenities_interior():
    raw = {"amenities_indoor": ["Gas heating"]}
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned == {"amenities_interior": ["Gas heating"]}
    assert changes["rename_top_level:amenities_indoor->amenities_interior"] == 1


def test_typo_rename_merges_with_existing_key():
    # Haiku emits both the typo AND the real key in the same row.
    raw = {
        "amenities_interior": ["Gas heating"],
        "amenities_indoor": ["Wooden floors"],
    }
    cleaned, changes = clean_extraction(raw, DESC)
    assert set(cleaned["amenities_interior"]) == {"Gas heating", "Wooden floors"}
    assert changes["merge_after_rename:amenities_interior"] == 1


def test_unknown_top_level_key_is_dropped():
    raw = {"property_type": "Flat", "secret_sauce": "mystery"}
    cleaned, changes = clean_extraction(raw, DESC)
    assert "secret_sauce" not in cleaned
    assert cleaned["property_type"] == "Flat"
    assert changes["drop_unknown_top_level_key:secret_sauce"] == 1


# ---------------------------------------------------------------------------
# Amenity vocab + relocation
# ---------------------------------------------------------------------------


def test_out_of_vocab_amenity_is_dropped():
    raw = {"amenities_interior": ["Gas heating", "Electric heating"]}
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned["amenities_interior"] == ["Gas heating"]
    assert changes["drop_out_of_vocab:amenities_interior:Electric heating"] == 1


def test_misgrouped_amenity_is_relocated_not_lost():
    # "Washing machine" is an appliance; Haiku sometimes puts it in
    # `amenities_interior`. We want to keep the fact and move it.
    raw = {"amenities_interior": ["Washing machine"]}
    cleaned, changes = clean_extraction(raw, DESC)
    assert "amenities_interior" not in cleaned
    assert cleaned["amenities_appliances"] == ["Washing machine"]
    assert (
        changes["relocate_amenity:amenities_interior->amenities_appliances"] == 1
    )


def test_amenity_group_becomes_empty_is_dropped_not_nulled():
    # An empty amenity list leaks information ("model said nothing applies"
    # which is distinct from "model didn't extract the field"). Prefer to
    # omit the key entirely.
    raw = {"amenities_interior": ["Electric heating"]}  # only an out-of-vocab value
    cleaned, _ = clean_extraction(raw, DESC)
    assert "amenities_interior" not in cleaned


def test_duplicate_amenity_is_deduped():
    raw = {"amenities_interior": ["Gas heating", "Gas heating"]}
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned["amenities_interior"] == ["Gas heating"]
    assert changes["dedup_amenity:amenities_interior"] == 1


# ---------------------------------------------------------------------------
# Nested blocks
# ---------------------------------------------------------------------------


def test_unknown_subkey_on_sale_block_is_dropped():
    # `rental_income_monthly` is a real hallucination from Haiku's output.
    raw = {
        "sale": {
            "tenure_type": "Freehold",
            "rental_income_monthly": 1200,
        }
    }
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned["sale"] == {"tenure_type": "Freehold"}
    assert changes["drop_unknown_sub_key:sale.rental_income_monthly"] == 1


def test_nested_block_with_all_invalid_keys_is_dropped_entirely():
    raw = {"flags": {"bogus_flag_one": True, "bogus_flag_two": False}}
    cleaned, _ = clean_extraction(raw, DESC)
    assert "flags" not in cleaned  # don't leave an empty {} behind


# ---------------------------------------------------------------------------
# nearby_mentions / commute_claims
# ---------------------------------------------------------------------------


def test_nearby_mentions_drops_leaked_mode_key():
    # `mode` belongs on commute_claims, not nearby_mentions. Haiku leaks
    # it roughly 5% of the time.
    raw = {
        "nearby_mentions": [
            {"name": "Clapham North", "walk_minutes": 5, "mode": "tube"}
        ]
    }
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned["nearby_mentions"] == [
        {"name": "Clapham North", "walk_minutes": 5}
    ]
    assert changes["drop_unknown_item_key:nearby_mentions.mode"] == 1


def test_commute_claims_drops_mode_outside_literal_but_keeps_claim():
    # "tram" isn't in CommuteClaim.mode's Literal but is common in UK
    # listings (Nottingham, Manchester). Drop the key, keep the row.
    raw = {
        "commute_claims": [
            {"destination": "Nottingham Trent University", "minutes": 10, "mode": "tram"}
        ]
    }
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned["commute_claims"] == [
        {"destination": "Nottingham Trent University", "minutes": 10}
    ]
    assert changes["drop_out_of_vocab:commute_claims.mode:tram"] == 1


def test_list_dict_entry_missing_id_key_is_dropped():
    # No `name` on this nearby_mentions item — we can't do anything with
    # it downstream, so drop it.
    raw = {
        "nearby_mentions": [
            {"name": "Station", "walk_minutes": 5},
            {"walk_minutes": 10},
        ]
    }
    cleaned, changes = clean_extraction(raw, DESC)
    assert len(cleaned["nearby_mentions"]) == 1
    assert cleaned["nearby_mentions"][0]["name"] == "Station"
    assert changes["drop_item_missing_id:nearby_mentions"] == 1


# ---------------------------------------------------------------------------
# Evidence grounding
# ---------------------------------------------------------------------------


def test_evidence_quote_not_in_description_is_dropped():
    raw = {
        "beds": 2,
        "evidence": {
            "beds": "Two-bedroom flat",  # grounded
            "flags.is_hmo": "registered HMO",  # not in description -> hallucination
        },
    }
    cleaned, changes = clean_extraction(raw, DESC)
    assert cleaned["evidence"] == {"beds": "Two-bedroom flat"}
    assert changes["drop_ungrounded_evidence:flags.is_hmo"] == 1


def test_evidence_grounding_tolerates_whitespace_and_case():
    desc = "Close to the STATION\xa0and\ngas\u2013heating throughout"
    raw = {"evidence": {"amenities_interior.Gas heating": "gas-heating throughout"}}
    cleaned, _ = clean_extraction(raw, desc)
    # NBSP/en-dash/case all normalised — should ground.
    assert "amenities_interior.Gas heating" in cleaned["evidence"]


def test_empty_evidence_quotes_are_dropped():
    raw = {"evidence": {"beds": "", "baths": "   "}}
    cleaned, changes = clean_extraction(raw, DESC)
    assert "evidence" not in cleaned
    assert changes["drop_empty_evidence"] == 2


# ---------------------------------------------------------------------------
# Changes accumulator sharing (the pipeline runs cleanup over many rows
# and uses a single Counter to aggregate stats).
# ---------------------------------------------------------------------------


def test_changes_counter_accumulates_across_calls():
    acc: Counter[str] = Counter()
    clean_extraction({"amenities_indoor": ["Gas heating"]}, DESC, acc)
    clean_extraction({"amenities_indoor": ["Wooden floors"]}, DESC, acc)
    assert acc["rename_top_level:amenities_indoor->amenities_interior"] == 2


# ---------------------------------------------------------------------------
# End-to-end: after cleaning, Pydantic validation should succeed on rows
# that started out with fixable issues only.
# ---------------------------------------------------------------------------


def test_row_with_fixable_issues_passes_validation_after_cleaning():
    # Every mistake here is one that cleanup should repair.
    raw = {
        "property_type": "Flat",
        "beds": 2,
        "amenities_indoor": ["Washing machine"],  # typo AND wrong group
        "sale": {"tenure_type": "Freehold", "rental_income_monthly": 1000},  # hallucinated key
        "nearby_mentions": [{"name": "Station", "walk_minutes": 5, "mode": "tube"}],  # leaked key
        "evidence": {"beds": "Two-bedroom flat"},
    }
    cleaned, _ = clean_extraction(raw, DESC)
    assert validate_extraction(cleaned) is None


def test_row_with_irrecoverable_structure_still_fails_validation():
    # Neither the cleaner nor Pydantic will accept a `beds` that isn't
    # an int. The cleaner leaves the value alone (outside its scope);
    # validation surfaces it as the caller's problem.
    raw = {"beds": "two-ish"}
    cleaned, _ = clean_extraction(raw, DESC)
    assert validate_extraction(cleaned) is not None
