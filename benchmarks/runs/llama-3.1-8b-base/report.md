# Benchmark: llama-3.1-8b-base

## Headlines

- **Examples:** 60
- **JSON parse rate:** 100.0%
- **Schema validity rate:**  65.0%
- **Macro field accuracy:**  81.7%
- **Evidence grounding rate:**  82.4%

## Per-field

| field | n | cov | acc | prec | rec | F1 | set_F1 | vocab! | case! |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `amenities_interior` |  60 | 0.750 | 0.233 | 0.200 | 0.244 | 0.220 | 0.386 |  16 |   0 |
| `nearby_mentions` |  60 | 0.700 | 0.383 | 0.333 | 0.238 | 0.278 | 0.429 |   0 |   0 |
| `amenities_outdoor` |  60 | 0.667 | 0.500 | 0.444 | 0.500 | 0.471 | 0.637 |   5 |   0 |
| `amenities_appliances` |  60 | 0.267 | 0.633 | 0.158 | 0.188 | 0.171 | 0.413 |   3 |   0 |
| `rent.bills_included` |  60 | 0.383 | 0.633 | 0.487 | 0.826 | 0.613 |    —  |   0 |   0 |
| `living_rooms` |  60 | 0.550 | 0.650 | 0.737 | 0.424 | 0.538 |    —  |   0 |   0 |
| `rent.deposit_weeks` |  60 | 0.083 | 0.700 | 0.182 | 0.800 | 0.296 |    —  |   0 |   0 |
| `amenities_facilities` |  60 | 0.100 | 0.717 | 0.000 | 0.000 | 0.000 | 0.279 |   9 |   0 |
| `rent.rental_support_accepted` |  60 | 0.050 | 0.733 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.current_occupation` |  60 | 0.017 | 0.733 | 0.059 | 1.000 | 0.111 |    —  |   0 |   0 |
| `rent.deposit_amount` |  60 | 0.167 | 0.750 | 0.391 | 0.900 | 0.545 |    —  |   0 |   0 |
| `room.current_number_of_flatmates` |  60 | 0.083 | 0.750 | 0.167 | 0.600 | 0.261 |    —  |   0 |   0 |
| `room.family_friendly` |  60 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.vegetarian_household` |  60 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.lgbt_household` |  60 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `parking` |  60 | 0.417 | 0.767 | 0.538 | 0.560 | 0.549 | 0.607 |   5 |   0 |
| `rent.bills_amount` |  60 | 0.017 | 0.767 | 0.067 | 1.000 | 0.125 |    —  |   0 |   0 |
| `rent.dps` |  60 | 0.033 | 0.767 | 0.071 | 0.500 | 0.125 |    —  |   0 |   0 |
| `room.current_gender` |  60 | 0.067 | 0.767 | 0.176 | 0.750 | 0.286 |    —  |   0 |   0 |
| `room.current_pets` |  60 | 0.017 | 0.767 | 0.067 | 1.000 | 0.125 |    —  |   0 |   0 |
| `room.current_smokers` |  60 | 0.000 | 0.767 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `rent.student_friendly` |  60 | 0.133 | 0.783 | 0.308 | 0.500 | 0.381 |    —  |   0 |   0 |
| `room.desired_age_min` |  60 | 0.017 | 0.783 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.desired_occupation` |  60 | 0.067 | 0.783 | 0.083 | 0.250 | 0.125 |    —  |   0 |   0 |
| `room.shared_living_room` |  60 | 0.167 | 0.783 | 0.316 | 0.600 | 0.414 |    —  |   0 |   0 |
| `rent.let_type` |  60 | 0.133 | 0.800 | 0.333 | 0.625 | 0.435 |    —  |   0 |   0 |
| `rent.smokers_allowed` |  60 | 0.083 | 0.800 | 0.294 | 1.000 | 0.455 |    —  |   0 |   0 |
| `room.desired_age_max` |  60 | 0.000 | 0.800 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.desired_gender` |  60 | 0.000 | 0.800 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `property_type` |  60 | 1.000 | 0.817 | 0.817 | 0.817 | 0.817 |    —  |   2 |   0 |
| `beds` |  60 | 0.700 | 0.817 | 0.776 | 0.905 | 0.835 |    —  |   0 |   0 |
| `floors_occupied` |  60 | 0.333 | 0.817 | 0.800 | 0.600 | 0.686 |    —  |   0 |   0 |
| `rent.furnish_type` |  60 | 0.500 | 0.817 | 0.727 | 0.800 | 0.762 |    —  |   0 |   0 |
| `rent.pets_allowed` |  60 | 0.100 | 0.817 | 0.333 | 0.833 | 0.476 |    —  |   0 |   0 |
| `room.couples_allowed` |  60 | 0.067 | 0.817 | 0.267 | 1.000 | 0.421 |    —  |   0 |   0 |
| `school_mentions` |  60 | 0.150 | 0.817 | 0.000 | 0.000 | 0.000 | 0.182 |   0 |   0 |
| `baths` |  60 | 0.583 | 0.833 | 0.775 | 0.886 | 0.827 |    —  |   0 |   0 |
| `total_floors` |  60 | 0.083 | 0.833 | 0.222 | 0.400 | 0.286 |    —  |   0 |   0 |
| `rent.tenancy_min_months` |  60 | 0.067 | 0.850 | 0.200 | 0.500 | 0.286 |    —  |   0 |   0 |
| `flags.is_chain_free` |  60 | 0.050 | 0.850 | 0.200 | 0.667 | 0.308 |    —  |   0 |   0 |
| `rent.wifi_included` |  60 | 0.150 | 0.867 | 0.533 | 0.889 | 0.667 |    —  |   0 |   0 |
| `room.listing_for_number_of_rooms` |  60 | 0.067 | 0.867 | 0.273 | 0.750 | 0.400 |    —  |   0 |   0 |
| `room.current_age_min` |  60 | 0.000 | 0.867 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.current_age_max` |  60 | 0.000 | 0.867 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_cash_only` |  60 | 0.017 | 0.883 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `commute_claims` |  60 | 0.117 | 0.883 | 0.500 | 0.143 | 0.222 | 0.250 |   0 |   0 |
| `rent.tenancy_max_months` |  60 | 0.017 | 0.900 | 0.143 | 1.000 | 0.250 |    —  |   0 |   0 |
| `flags.is_build_to_rent` |  60 | 0.033 | 0.900 | 0.167 | 0.500 | 0.250 |    —  |   0 |   0 |
| `flags.is_hmo` |  60 | 0.033 | 0.900 | 0.250 | 1.000 | 0.400 |    —  |   0 |   0 |
| `flags.is_auction` |  60 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.has_planning_permission` |  60 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_tenanted` |  60 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.has_commercial_tenant` |  60 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_new_build` |  60 | 0.033 | 0.900 | 0.167 | 0.500 | 0.250 |    —  |   0 |   0 |
| `flags.is_warehouse_conversion` |  60 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_graded_building` |  60 | 0.000 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.wheelchair_access` |  60 | 0.017 | 0.900 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.tenure_type` |  60 | 0.050 | 0.917 | 0.286 | 0.667 | 0.400 |    —  |   1 |   0 |
| `room.room_type` |  60 | 0.250 | 0.917 | 0.737 | 0.933 | 0.824 |    —  |   1 |   0 |
| `room.is_ensuite` |  60 | 0.083 | 0.933 | 0.556 | 1.000 | 0.714 |    —  |   0 |   0 |
| `window_orientation` |  60 | 0.033 | 0.950 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.ground_rent_amount` |  60 | 0.000 | 0.967 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.service_charge_amount` |  60 | 0.033 | 0.967 | 0.500 | 0.500 | 0.500 |    —  |   0 |   0 |
| `sale.is_shared_ownership` |  60 | 0.000 | 0.967 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `total_floor_area_sqm` |  60 | 0.067 | 0.983 | 0.800 | 1.000 | 0.889 |    —  |   0 |   0 |
| `build_year_min` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `build_year_max` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `sale.length_of_lease_years` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `sale.purchase_scheme` |  60 | 0.000 | 0.983 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |

Notes: `cov` = fraction of gold rows with a value present; `acc` = exact-match over all rows (tn included); `F1` = field-presence F1 (treats partial-value mismatches as errors); `set_F1` = member-level F1 for list fields; `vocab!` = predictions outside the controlled vocabulary; `case!` = predictions that only differ in case.
