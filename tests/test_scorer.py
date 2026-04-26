"""Unit tests for the scorer. The scorer is the project's load-bearing
component — all training decisions flow from its numbers — so these tests
cover the tricky edge cases explicitly:

- omit vs explicit-false for booleans
- evidence grounding with whitespace/unicode normalisation
- vocab violations (outside enum) vs case mismatches (wrong case)
- set F1 for list fields
- list-dict matching by identifying field only
- gold-vs-gold should score perfectly
"""

from __future__ import annotations

from listing_parser.benchmarks.scorer import (_cmp_list_dict, _cmp_list_set,
                                              _cmp_scalar_exact,
                                              _cmp_scalar_float, _norm_text,
                                              _score_evidence, aggregate,
                                              score_one)

# ---------------------------------------------------------------------------
# Scalar comparators
# ---------------------------------------------------------------------------


def test_scalar_enum_exact_match():
    r = _cmp_scalar_exact("property_type", "Flat", "Flat", "scalar_enum")
    assert r.correct and r.tp and not r.vocab_violation and not r.case_mismatch


def test_scalar_enum_case_mismatch_is_flagged():
    r = _cmp_scalar_exact("property_type", "Flat", "flat", "scalar_enum")
    assert not r.correct
    assert r.case_mismatch is True
    # "flat" isn't in the vocab (case-sensitive), so it's also a vocab violation
    assert r.vocab_violation is True


def test_scalar_enum_vocab_violation_without_case_flag():
    r = _cmp_scalar_exact("property_type", "Flat", "Apartment", "scalar_enum")
    assert not r.correct and r.vocab_violation and not r.case_mismatch


def test_scalar_enum_absent_both_is_not_tp():
    r = _cmp_scalar_exact("property_type", None, None, "scalar_enum")
    # Neither present, not marked correct — absence agreement is handled
    # at the aggregate level (as tn), not here.
    assert not r.tp and not r.fp and not r.fn


def test_scalar_bool_three_way_omit_vs_false():
    # Gold omits (None); model emits false -> fp hallucination.
    r = _cmp_scalar_exact("flags.is_hmo", None, False, "scalar_bool")
    assert r.fp and not r.tp and not r.fn

    # Gold=False (explicit); model omits -> fn.
    r2 = _cmp_scalar_exact("flags.is_hmo", False, None, "scalar_bool")
    assert r2.fn and not r2.tp

    # Both false -> tp.
    r3 = _cmp_scalar_exact("flags.is_hmo", False, False, "scalar_bool")
    assert r3.tp


def test_scalar_float_tolerance():
    # Within 5% default tolerance.
    r = _cmp_scalar_float("total_floor_area_sqm", 100.0, 103.0)
    assert r.correct
    # Outside tolerance.
    r2 = _cmp_scalar_float("total_floor_area_sqm", 100.0, 120.0)
    assert not r2.correct


def test_scalar_float_handles_non_numeric():
    r = _cmp_scalar_float("total_floor_area_sqm", 100.0, "approx 100")
    assert not r.correct and "non-numeric pred" in r.notes


# ---------------------------------------------------------------------------
# List comparators
# ---------------------------------------------------------------------------


def test_list_set_precision_recall():
    gold = ["Gas heating", "Wooden floors"]
    pred = ["Gas heating", "Underfloor heating"]  # missed wooden, added underfloor
    r = _cmp_list_set("amenities_interior", gold, pred)
    assert r.precision == 0.5
    assert r.recall == 0.5
    assert not r.correct


def test_list_set_perfect_empty_agreement():
    r = _cmp_list_set("amenities_interior", None, None)
    # Both empty is perfect agreement at the member level.
    assert r.precision == 1.0 and r.recall == 1.0 and r.correct


def test_list_set_vocab_violation():
    # "Electric heating" isn't in AMENITIES_INTERIOR.
    r = _cmp_list_set(
        "amenities_interior", ["Gas heating"], ["Gas heating", "Electric heating"]
    )
    assert r.vocab_violation


def test_list_set_compass_splits_on_comma():
    # window_orientation is stored as "N,E,S"; compared as a set.
    r = _cmp_list_set("window_orientation", "N,S", "S,N", use_compass=True)
    assert r.correct and r.precision == 1.0 and r.recall == 1.0


def test_list_dict_matches_on_name_only():
    gold = [{"name": "Clapham North", "walk_minutes": 5}]
    pred = [{"name": "clapham north", "walk_minutes": 99}]  # wrong minutes OK
    r = _cmp_list_dict("nearby_mentions", gold, pred, "name")
    # Normalisation lowercases, so these match as the same mention.
    assert r.correct and r.precision == 1.0 and r.recall == 1.0


# ---------------------------------------------------------------------------
# Evidence grounding
# ---------------------------------------------------------------------------


def test_evidence_grounding_with_whitespace_normalisation():
    desc = "The property features a spacious\nopen-plan living\nand dining area"
    ev = {
        "beds": "open-plan living",
        "hallucinated": "has a private helipad",
    }
    total, grounded = _score_evidence(ev, desc)
    assert total == 2 and grounded == 1


def test_evidence_grounding_handles_nbsp_and_endash():
    desc = "approximately\xa05\u20137 minute walk to the station"
    ev = {"commute": "5-7 minute walk"}  # NBSP -> space, en-dash -> hyphen
    total, grounded = _score_evidence(ev, desc)
    assert total == 1 and grounded == 1


def test_evidence_grounding_skips_empty_quotes():
    total, grounded = _score_evidence({"a": "", "b": "   "}, "whatever")
    assert total == 0 and grounded == 0


# ---------------------------------------------------------------------------
# End-to-end: gold-vs-gold must be perfect
# ---------------------------------------------------------------------------


SAMPLE_GOLD = {
    "row_index": 0,
    "description": "Two-bedroom apartment on the 15th floor with lift and open-plan kitchen.",
    "listing_type": "Rent",
    "output": {
        "property_type": "Flat",
        "beds": 2,
        "baths": 1,
        "floors_occupied": [15],
        "amenities_interior": ["Open plan living area"],
        "amenities_facilities": ["Elevator"],
        "rent": {"bills_included": "None", "smokers_allowed": False},
        "evidence": {
            "beds": "Two-bedroom apartment",
            "floors_occupied.15": "15th floor",
            "amenities_facilities.Elevator": "lift",
        },
    },
}


def test_gold_vs_gold_is_perfect():
    score = score_one(SAMPLE_GOLD, SAMPLE_GOLD["output"], SAMPLE_GOLD["description"], 0)
    assert score.parse_ok and score.schema_ok
    # No field should be wrong, though many will be absent on both sides.
    assert not any(r.wrong for r in score.field_results)
    assert not any(r.fp or r.fn for r in score.field_results)
    # Evidence should ground.
    assert score.evidence_grounded == score.evidence_total == 3

    report = aggregate([score])
    d = report.to_dict()
    assert d["parse_rate"] == 1.0
    assert d["schema_rate"] == 1.0
    assert d["macro_accuracy"] > 0.999
    assert d["evidence_grounding_rate"] == 1.0


def test_aggregate_with_one_bad_example_drops_scores():
    # "flat" is a case regression of "Flat" — same string, different case.
    # Both a case_mismatch AND a vocab_violation (vocab is case-sensitive).
    bad_pred = {
        "property_type": "flat",
        "beds": 3,         # wrong value
        "baths": 1,        # correct
    }
    s_good = score_one(SAMPLE_GOLD, SAMPLE_GOLD["output"], SAMPLE_GOLD["description"], 0)
    s_bad = score_one(SAMPLE_GOLD, bad_pred, SAMPLE_GOLD["description"], 1)
    report = aggregate([s_good, s_bad])

    assert report.by_field["property_type"]["accuracy"] == 0.5
    assert report.by_field["beds"]["accuracy"] == 0.5
    assert report.by_field["property_type"]["vocab_violations"] == 1.0
    assert report.by_field["property_type"]["case_mismatches"] == 1.0


def test_norm_text_collapses_whitespace_and_case():
    assert _norm_text("Hello\tworld\n") == "hello world"
    assert _norm_text("FOO\xa0BAR") == "foo bar"
