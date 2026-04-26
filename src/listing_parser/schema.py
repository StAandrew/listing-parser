"""Schema + vocabulary for the listing-description extraction task.

Source of truth is `prompt.md` at the repo root. This module mirrors the
schema as Python enums + a Pydantic model so downstream code (scorer,
test-set builder, training data validator) can rely on a single
programmatic definition.

Keep this in sync with `prompt.md` by hand. A future iteration could
generate this from the prompt via a codegen step, but at <100 enum
values the manual duplication is worth the simplicity.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Controlled vocabularies
#
# These mirror the seeded DB rows in
# `20251111_first.mjs`:
#   - propertyTypeSeed (lines 63-80)
#   - amenitySeed      (lines 91-150)
#   - parkingTypeSeed  (lines 152-159)
#   - purchaseSchemeSeed (lines 161-167)
# ---------------------------------------------------------------------------

PROPERTY_TYPES: tuple[str, ...] = (
    "Flat",
    "House",
    "Bungalow",
    "Maisonette",
    "Detached",
    "Semi-detached",
    "Terraced",
    "End of terrace",
    "Cottage",
    "Park home",
    "Retirement home",
    "Farm",
    "Storage Unit",
    "Land",
    "Garage",
    "Parking Space",
)

AMENITIES_INTERIOR: tuple[str, ...] = (
    "Wooden floors",
    "Underfloor heating",
    "Gas heating",
    "Double glazing",
    "Kitchen island",
    "Open plan living area",
    "Stairs",
    "Floor-to-ceiling windows",
    "High ceilings",
    "Walk-in wardrobe",
    "Loft/Attic",
    "Utility room",
    "En-suite",
    "Bathtub",
    "Walk-in shower",
    "Bidet",
    "Fireplace",
    "Conservatory",
    "A/C",
    "Smart home",
    "Security alarm",
)

AMENITIES_OUTDOOR: tuple[str, ...] = (
    "Balcony",
    "Patio/Terrace",
    "Private Garden",
    "Communal Garden",
    "Private roof terrace",
    "Shed",
    "Penthouse",
    "Basement",
    "Bike storage",
    "EV charger",
    "Tennis court",
    "Gated development",
)

AMENITIES_FACILITIES: tuple[str, ...] = (
    "Sauna",
    "Spa",
    "Swimming pool",
    "Gym/Fitness centre",
    "Concierge/Doorman",
    "Cinema room",
    "Residents' lounge",
    "Co-working space",
    "Package room",
    "Communal laundry",
    "Rooftop garden",
    "Children's play area",
    "Elevator",
)

AMENITIES_APPLIANCES: tuple[str, ...] = (
    "Washing machine",
    "Dryer",
    "Dishwasher",
    "Fridge/Freezer",
    "Microwave",
)

PARKING_TYPES: tuple[str, ...] = (
    "On street",
    "Off street shared",
    "Allocated space",
    "Private driveway",
    "Garage",
    "Underground parking",
)

PURCHASE_SCHEMES: tuple[str, ...] = (
    "Help to Buy",
    "Right to Buy",
    "First Homes",
    "Rent to Buy",
    "Forces Help to Buy",
)

FURNISH_TYPES: tuple[str, ...] = ("Furnished", "Part-furnished", "Unfurnished")
LET_TYPES: tuple[str, ...] = ("Long let", "Short let", "Student let", "Holiday let")
TENURE_TYPES: tuple[str, ...] = ("Freehold", "Leasehold", "Share of Freehold", "Commonhold")
ROOM_TYPES: tuple[str, ...] = ("Double", "Single", "Studio", "Twin")
BILLS_INCLUDED: tuple[str, ...] = ("None", "Some", "All")
COMPASS_POINTS: frozenset[str] = frozenset({"N", "NE", "E", "SE", "S", "SW", "W", "NW"})


# Mapping used by the scorer to route amenity strings to their owning array.
# If a value shows up in the "wrong" list (e.g. "Elevator" under amenities_outdoor),
# the scorer flags it as a schema violation but still credits the fact.
AMENITY_GROUPS: dict[str, tuple[str, ...]] = {
    "amenities_interior": AMENITIES_INTERIOR,
    "amenities_outdoor": AMENITIES_OUTDOOR,
    "amenities_facilities": AMENITIES_FACILITIES,
    "amenities_appliances": AMENITIES_APPLIANCES,
}


# ---------------------------------------------------------------------------
# Pydantic model — used by the scorer as a schema validator.
#
# Design decisions worth calling out:
#   * Every field is optional. The prompt explicitly instructs the model to
#     OMIT unmentioned keys rather than emit nulls, so `None` at parse time
#     means "the model didn't extract this", which is a valid state.
#   * `extra="forbid"` catches hallucinated top-level keys. We keep this
#     strict at validation time; the scorer downgrades unknown keys to a
#     soft warning rather than a hard parse failure.
#   * We accept both proper casing and enum-style strings; case mismatches
#     are flagged by the scorer, not rejected here, so we can distinguish
#     "model emitted 'flat' instead of 'Flat'" from "model emitted garbage".
# ---------------------------------------------------------------------------


class RentBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    furnish_type: str | None = None
    let_type: str | None = None
    tenancy_min_months: int | None = None
    tenancy_max_months: int | None = None
    deposit_weeks: float | None = None
    deposit_amount: float | None = None
    bills_included: str | None = None
    bills_amount: float | None = None
    pets_allowed: bool | None = None
    smokers_allowed: bool | None = None
    wifi_included: bool | None = None
    rental_support_accepted: bool | None = None
    student_friendly: bool | None = None
    dps: bool | None = None


class SaleBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenure_type: str | None = None
    length_of_lease_years: int | None = None
    ground_rent_amount: float | None = None
    service_charge_amount: float | None = None
    is_shared_ownership: bool | None = None
    purchase_scheme: str | None = None


class RoomBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    room_type: str | None = None
    is_ensuite: bool | None = None
    listing_for_number_of_rooms: int | None = None
    current_number_of_flatmates: int | None = None
    current_gender: str | None = None
    current_age_min: int | None = None
    current_age_max: int | None = None
    current_occupation: str | None = None
    current_pets: bool | None = None
    current_smokers: bool | None = None
    desired_age_min: int | None = None
    desired_age_max: int | None = None
    desired_occupation: str | None = None
    desired_gender: str | None = None
    couples_allowed: bool | None = None
    family_friendly: bool | None = None
    vegetarian_household: bool | None = None
    lgbt_household: bool | None = None
    shared_living_room: bool | None = None


class FlagsBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_chain_free: bool | None = None
    is_build_to_rent: bool | None = None
    is_cash_only: bool | None = None
    is_hmo: bool | None = None
    is_auction: bool | None = None
    has_planning_permission: bool | None = None
    is_tenanted: bool | None = None
    has_commercial_tenant: bool | None = None
    is_new_build: bool | None = None
    is_warehouse_conversion: bool | None = None
    is_graded_building: bool | None = None
    wheelchair_access: bool | None = None


class NearbyMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    walk_minutes: int | None = None
    drive_minutes: int | None = None


class CommuteClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    destination: str
    minutes: int | None = None
    mode: Literal["walk", "train", "tube", "bus", "car", "cycle"] | None = None


class Extraction(BaseModel):
    """Top-level schema for a listing-description extraction.

    This is the shape `prompt.md` asks the model to emit.
    """

    model_config = ConfigDict(extra="forbid")

    property_type: str | None = None
    beds: int | None = None
    baths: int | None = None
    living_rooms: int | None = None
    total_floor_area_sqm: float | None = None
    total_floors: int | None = None
    floors_occupied: list[int] | None = None
    build_year_min: int | None = None
    build_year_max: int | None = None
    window_orientation: str | None = None

    amenities_interior: list[str] | None = None
    amenities_outdoor: list[str] | None = None
    amenities_facilities: list[str] | None = None
    amenities_appliances: list[str] | None = None

    parking: list[str] | None = None

    rent: RentBlock | None = None
    sale: SaleBlock | None = None
    room: RoomBlock | None = None
    flags: FlagsBlock | None = None

    nearby_mentions: list[NearbyMention] | None = None
    commute_claims: list[CommuteClaim] | None = None
    school_mentions: list[str] | None = None

    # Evidence map — keyed by dotted-path, values are verbatim quotes.
    evidence: dict[str, str] | None = Field(default=None)


# ---------------------------------------------------------------------------
# Field-kind catalogue driving the scorer.
#
# The scorer doesn't compare whole objects; it compares field-by-field so we
# can see WHICH axis of the extraction the model is weak on. Each top-level
# field below has a "kind" that picks the right comparison function:
#
#   scalar_enum   -> exact string match against a vocabulary
#   scalar_int    -> exact int match
#   scalar_float  -> relative tolerance (default 5%)
#   scalar_bool   -> 3-way: present-true / present-false / absent
#   list_set      -> precision/recall over a set (ignores order + dupes)
#   list_ordered  -> order-preserving list (e.g. floors_occupied)
#   nested_block  -> recurse into a sub-model
#   list_dict     -> list of small dicts (nearby_mentions, commute_claims);
#                    matched by the identifying field (name/destination)
# ---------------------------------------------------------------------------

FieldKind = Literal[
    "scalar_enum",
    "scalar_int",
    "scalar_float",
    "scalar_bool",
    "list_set",
    "list_ordered",
    "nested_block",
    "list_dict",
    "evidence_map",
    "string",
]

FIELD_KINDS: dict[str, FieldKind] = {
    "property_type": "scalar_enum",
    "beds": "scalar_int",
    "baths": "scalar_int",
    "living_rooms": "scalar_int",
    "total_floor_area_sqm": "scalar_float",
    "total_floors": "scalar_int",
    "floors_occupied": "list_ordered",
    "build_year_min": "scalar_int",
    "build_year_max": "scalar_int",
    "window_orientation": "scalar_enum",  # compared as a set on "," split
    "amenities_interior": "list_set",
    "amenities_outdoor": "list_set",
    "amenities_facilities": "list_set",
    "amenities_appliances": "list_set",
    "parking": "list_set",
    "rent": "nested_block",
    "sale": "nested_block",
    "room": "nested_block",
    "flags": "nested_block",
    "nearby_mentions": "list_dict",
    "commute_claims": "list_dict",
    "school_mentions": "list_set",
    "evidence": "evidence_map",
}


# Sub-block field kinds. Keyed by "<block>.<field>".
NESTED_FIELD_KINDS: dict[str, FieldKind] = {
    # rent
    "rent.furnish_type": "scalar_enum",
    "rent.let_type": "scalar_enum",
    "rent.tenancy_min_months": "scalar_int",
    "rent.tenancy_max_months": "scalar_int",
    "rent.deposit_weeks": "scalar_float",
    "rent.deposit_amount": "scalar_float",
    "rent.bills_included": "scalar_enum",
    "rent.bills_amount": "scalar_float",
    "rent.pets_allowed": "scalar_bool",
    "rent.smokers_allowed": "scalar_bool",
    "rent.wifi_included": "scalar_bool",
    "rent.rental_support_accepted": "scalar_bool",
    "rent.student_friendly": "scalar_bool",
    "rent.dps": "scalar_bool",
    # sale
    "sale.tenure_type": "scalar_enum",
    "sale.length_of_lease_years": "scalar_int",
    "sale.ground_rent_amount": "scalar_float",
    "sale.service_charge_amount": "scalar_float",
    "sale.is_shared_ownership": "scalar_bool",
    "sale.purchase_scheme": "scalar_enum",
    # room
    "room.room_type": "scalar_enum",
    "room.is_ensuite": "scalar_bool",
    "room.listing_for_number_of_rooms": "scalar_int",
    "room.current_number_of_flatmates": "scalar_int",
    "room.current_gender": "scalar_enum",
    "room.current_age_min": "scalar_int",
    "room.current_age_max": "scalar_int",
    "room.current_occupation": "string",
    "room.current_pets": "scalar_bool",
    "room.current_smokers": "scalar_bool",
    "room.desired_age_min": "scalar_int",
    "room.desired_age_max": "scalar_int",
    "room.desired_occupation": "string",
    "room.desired_gender": "scalar_enum",
    "room.couples_allowed": "scalar_bool",
    "room.family_friendly": "scalar_bool",
    "room.vegetarian_household": "scalar_bool",
    "room.lgbt_household": "scalar_bool",
    "room.shared_living_room": "scalar_bool",
    # flags
    "flags.is_chain_free": "scalar_bool",
    "flags.is_build_to_rent": "scalar_bool",
    "flags.is_cash_only": "scalar_bool",
    "flags.is_hmo": "scalar_bool",
    "flags.is_auction": "scalar_bool",
    "flags.has_planning_permission": "scalar_bool",
    "flags.is_tenanted": "scalar_bool",
    "flags.has_commercial_tenant": "scalar_bool",
    "flags.is_new_build": "scalar_bool",
    "flags.is_warehouse_conversion": "scalar_bool",
    "flags.is_graded_building": "scalar_bool",
    "flags.wheelchair_access": "scalar_bool",
}


def expected_vocab(field_path: str) -> tuple[str, ...] | None:
    """Return the controlled vocabulary for a field path, if any.

    Used by the scorer to classify enum-violation errors separately from
    semantic mismatches.
    """
    return _ENUM_VOCAB.get(field_path)


_ENUM_VOCAB: dict[str, tuple[str, ...]] = {
    "property_type": PROPERTY_TYPES,
    "parking": PARKING_TYPES,
    "amenities_interior": AMENITIES_INTERIOR,
    "amenities_outdoor": AMENITIES_OUTDOOR,
    "amenities_facilities": AMENITIES_FACILITIES,
    "amenities_appliances": AMENITIES_APPLIANCES,
    "rent.furnish_type": FURNISH_TYPES,
    "rent.let_type": LET_TYPES,
    "rent.bills_included": BILLS_INCLUDED,
    "sale.tenure_type": TENURE_TYPES,
    "sale.purchase_scheme": PURCHASE_SCHEMES,
    "room.room_type": ROOM_TYPES,
}
