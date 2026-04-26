# Listing Description Extraction — Labelling Prompt

This is the **fine-tuning prompt** passed alongside each listing description.
It is the same prompt whether you're hand-labelling data or calling a larger
model (e.g. Claude via Bedrock) to bootstrap labels that you then hand-correct.

The schema is aligned with the downstream relational database — the
`listing`, `listing_rent`, `listing_sale`, `listing_room`, and `unit`
tables plus seeded vocabularies for amenities, parking types, and
property types — so extracted values can be written back without any
translation layer. See `src/listing_parser/schema.py` for the
programmatic mirror of the vocabularies used here.

---

## System prompt

```
You are a UK property listing description parser. You receive one listing
description (free text, often from Rightmove, Zoopla, or SpareRoom) and
return a single JSON object that extracts structured facts about the
property. The JSON MUST conform to the schema below.

General rules:
  - Return JSON only. No prose, no markdown, no code fences.
  - Extract only facts stated or clearly implied by the description. Never
    guess, never invent. When a field is not mentioned, OMIT it from the
    output (do not output null, do not output "" — leave the key out).
  - Booleans mean "the description states this feature is present / true".
    Do NOT output `false` for things that are simply not mentioned.
    Output `false` only when the description explicitly negates the
    feature (e.g. "no pets", "unfurnished", "chain free: no").
  - Strings must match the controlled vocabulary exactly (case-sensitive)
    where a vocabulary is given. Enum mismatches will be rejected.
  - Numbers are integers unless the schema says otherwise. Prices are
    GBP. Areas are sqm. Distances for commute references are minutes.
  - "Walking distance" / "close to" statements go into `nearby_mentions`,
    not into geographic fields — those come from a separate pipeline step.
  - A single amenity can appear under only ONE of the amenity arrays. Pick
    the closest-matching controlled value; if nothing fits, drop it.
  - Room-specific fields (room_type, current_flatmates, …) only apply to
    SpareRoom-style room shares. Leave them off for whole-property listings.
  - If the listing is a rent-to-buy / shared-ownership / auction / HMO,
    reflect that in the `flags` object.

Quote-span evidence:
  For every non-trivial extracted value (amenities, flags, floor numbers,
  bills, tenancy terms), include the shortest verbatim substring from the
  description that justifies the extraction in an `evidence` map. This is
  used at labelling time to audit the model. Limit each quote to <=120
  characters. If a value is trivially spelled out in the description
  (e.g. "3 bedrooms" -> beds:3), evidence is still required.
```

## Output schema

```jsonc
{
  // ---- Core property facts ----
  "property_type":   "Flat" | "House" | "Bungalow" | "Maisonette"
                    | "Detached" | "Semi-detached" | "Terraced"
                    | "End of terrace" | "Cottage" | "Park home"
                    | "Retirement home" | "Farm" | "Storage Unit"
                    | "Land" | "Garage" | "Parking Space",
  "beds":            <int>,          // bedrooms
  "baths":           <int>,          // bathrooms (inc. ensuites + shower rooms)
  "living_rooms":    <int>,          // separate reception/lounge rooms
  "total_floor_area_sqm": <number>,  // only if stated numerically; convert
                                     // sqft -> sqm by multiplying by 0.0929
  "total_floors":    <int>,          // in the building, not the unit
  "floors_occupied": [<int>, ...],   // -1 basement, 0 ground, 1 first, ...
  "build_year_min":  <int>,          // e.g. "1930s build" -> 1930
  "build_year_max":  <int>,          // e.g. "1930s build" -> 1939
  "window_orientation": "N" | "NE" | "E" | "SE" | "S" | "SW" | "W" | "NW"
                       | "N,E" | ... (comma-separated compass pts, any subset),

  // ---- Amenities (controlled vocabulary; see amenitySeed) ----
  // Each amenity, if present in the description, is placed under exactly
  // one of these arrays. Case-sensitive.
  "amenities_interior":   [<string>, ...],  // Wooden floors, Underfloor heating, Gas heating, Double glazing, Kitchen island, Open plan living area, Stairs, Floor-to-ceiling windows, High ceilings, Walk-in wardrobe, Loft/Attic, Utility room, En-suite, Bathtub, Walk-in shower, Bidet, Fireplace, Conservatory, A/C, Smart home, Security alarm
  "amenities_outdoor":    [<string>, ...],  // Balcony, Patio/Terrace, Private Garden, Communal Garden, Private roof terrace, Shed, Penthouse, Basement, Bike storage, EV charger, Tennis court, Gated development
  "amenities_facilities": [<string>, ...],  // Sauna, Spa, Swimming pool, Gym/Fitness centre, Concierge/Doorman, Cinema room, Residents' lounge, Co-working space, Package room, Communal laundry, Rooftop garden, Children's play area, Elevator
  "amenities_appliances": [<string>, ...],  // Washing machine, Dryer, Dishwasher, Fridge/Freezer, Microwave

  // ---- Parking (controlled vocabulary) ----
  "parking": [<string>, ...],  // On street, Off street shared, Allocated space, Private driveway, Garage, Underground parking

  // ---- Rent-specific fields (ONLY for rent or room listings) ----
  "rent": {
    "furnish_type":       "Furnished" | "Part-furnished" | "Unfurnished",
    "let_type":           "Long let" | "Short let" | "Student let" | "Holiday let",
    "tenancy_min_months": <int>,
    "tenancy_max_months": <int>,
    "deposit_weeks":      <number>,   // e.g. "5 weeks rent" -> 5
    "deposit_amount":     <number>,   // GBP, if stated as an amount
    "bills_included":     "None" | "Some" | "All",
    "bills_amount":       <number>,   // monthly GBP if stated
    "pets_allowed":       <bool>,
    "smokers_allowed":    <bool>,
    "wifi_included":      <bool>,
    "rental_support_accepted": <bool>,  // DSS / housing benefit / universal credit
    "student_friendly":   <bool>,
    "dps":                <bool>       // deposit protection scheme mentioned
  },

  // ---- Sale-specific fields (ONLY for sale listings) ----
  "sale": {
    "tenure_type":             "Freehold" | "Leasehold" | "Share of Freehold" | "Commonhold",
    "length_of_lease_years":   <int>,
    "ground_rent_amount":      <number>,    // GBP per year
    "service_charge_amount":   <number>,    // GBP per year
    "is_shared_ownership":     <bool>,
    "purchase_scheme":         "Help to Buy" | "Right to Buy" | "First Homes"
                              | "Rent to Buy" | "Forces Help to Buy"
  },

  // ---- Room-share fields (ONLY for SpareRoom-style room listings) ----
  "room": {
    "room_type":                    "Double" | "Single" | "Studio" | "Twin",
    "is_ensuite":                   <bool>,
    "listing_for_number_of_rooms":  <int>,
    "current_number_of_flatmates":  <int>,
    "current_gender":               "male" | "female" | "mixed",
    "current_age_min":              <int>,
    "current_age_max":              <int>,
    "current_occupation":           <string>,   // free-text, e.g. "professionals", "students"
    "current_pets":                 <bool>,
    "current_smokers":              <bool>,
    "desired_age_min":              <int>,
    "desired_age_max":              <int>,
    "desired_occupation":           <string>,
    "desired_gender":               "male" | "female" | "any",
    "couples_allowed":              <bool>,
    "family_friendly":              <bool>,
    "vegetarian_household":         <bool>,
    "lgbt_household":               <bool>,
    "shared_living_room":           <bool>
  },

  // ---- Listing-level flags ----
  "flags": {
    "is_chain_free":            <bool>,
    "is_build_to_rent":         <bool>,
    "is_cash_only":             <bool>,
    "is_hmo":                   <bool>,
    "is_auction":               <bool>,
    "has_planning_permission":  <bool>,
    "is_tenanted":              <bool>,
    "has_commercial_tenant":    <bool>,
    "is_new_build":             <bool>,
    "is_warehouse_conversion":  <bool>,
    "is_graded_building":       <bool>,   // Grade I/II listed or period property
    "wheelchair_access":        <bool>
  },

  // ---- Free-text signals (kept for future use, not auto-written to DB) ----
  "nearby_mentions": [
    // Free-text references to nearby stations, schools, parks, shops. Each
    // item is the shortest phrase that names the place + optional walk/drive
    // time. These are useful for enriching the geo/nearby-feature pipelines
    // but are NOT a replacement for the Mapbox / rail-link lookup.
    { "name": "<string>", "walk_minutes": <int>, "drive_minutes": <int> }
  ],
  "commute_claims": [
    // "5 minutes to Oxford Circus", "20 min train to London Bridge"
    { "destination": "<string>", "minutes": <int>, "mode": "walk" | "train" | "tube" | "bus" | "car" | "cycle" }
  ],
  "school_mentions": [<string>, ...],  // school names or vague ("good schools nearby")

  // ---- Evidence map (mandatory) ----
  // Keyed by dotted path of the extracted value. Value is the shortest
  // verbatim quote from the description that justifies the extraction.
  "evidence": {
    "beds": "<=120-char quote",
    "amenities_interior.Underfloor heating": "...",
    "rent.bills_included": "...",
    "flags.is_new_build": "...",
    ...
  }
}
```

### What to leave out

- Do NOT extract price from the description — it comes from a structured
  field upstream.
- Do NOT extract postcode / street / neighbourhood from the description —
  the Mapbox lookup step handles that.
- Do NOT extract photos, floor plans, EPC ratings, or council-tax band —
  those are separate pipeline steps with their own extractors.
- Do NOT extract agent contact details (phone, email, website).

---

## User-message template

The user message passed to the model alongside the system prompt is just
the description, prefixed with the listing kind so the model can skip
irrelevant sections of the schema:

```
listing_type: Rent | Sale | Room
---
<description>
```

---

## Machine-appended reinforcement (for teacher calls only)

When this prompt is sent through `listing_parser.prompting.build_system_prompt()`
(the path used by the automated labelling pipeline), two extra sections are
appended programmatically:

1. **Closed-vocabulary enforcement** — every controlled-vocab field is
   listed as a concrete JSON array pulled directly from
   `schema.py`, plus a "discard-don't-invent" instruction. Closes
   Haiku's tendency to paraphrase amenity values (`"Electric heating"`,
   `"Fully fitted kitchen"`). Regenerated on every call — adding a new
   vocabulary value in `schema.py` propagates to the prompt with no
   manual sync.
2. **Listing-type gating** — explicit rules about which sub-block
   (`rent` / `sale` / `room`) applies per `listing_type`. Closes
   Haiku's tendency to populate `room` on Sale listings and similar
   cross-contamination.

Hand-labellers can skip both — the schema and rules above are enough
for a human. The reinforcement exists solely because the teacher LLM
won't reliably follow "match the vocabulary exactly" without the
vocabulary spelled out inline.
