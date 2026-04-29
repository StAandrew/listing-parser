# Benchmark: llama-3.1-8b-ft-v1 (quick, Bedrock)

## Headlines

- **Examples:** 60
- **JSON parse rate:** 100.0%
- **Schema validity rate:**  90.0%
- **Macro field accuracy:**  92.4%
- **Evidence grounding rate:**  99.4%

## Per-field

| field | n | cov | acc | prec | rec | F1 | set_F1 | vocab! | case! |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `amenities_interior` |  60 | 0.750 | 0.417 | 0.324 | 0.267 | 0.293 | 0.573 |   0 |   0 |
| `nearby_mentions` |  60 | 0.700 | 0.483 | 0.500 | 0.262 | 0.344 | 0.381 |   0 |   0 |
| `rent.let_type` |  60 | 0.133 | 0.650 | 0.250 | 0.875 | 0.389 |    —  |   0 |   0 |
| `rent.bills_included` |  60 | 0.383 | 0.650 | 0.529 | 0.783 | 0.632 |    —  |   0 |   0 |
| `rent.dps` |  60 | 0.033 | 0.750 | 0.118 | 1.000 | 0.211 |    —  |   0 |   0 |
| `rent.furnish_type` |  60 | 0.500 | 0.767 | 0.714 | 0.833 | 0.769 |    —  |   0 |   0 |
| `property_type` |  60 | 1.000 | 0.800 | 0.923 | 0.800 | 0.857 |    —  |   0 |   0 |
| `room.listing_for_number_of_rooms` |  60 | 0.067 | 0.817 | 0.182 | 0.500 | 0.267 |    —  |   0 |   0 |
| `amenities_appliances` |  60 | 0.267 | 0.833 | 0.727 | 0.500 | 0.593 | 0.492 |   0 |   0 |
| `amenities_outdoor` |  60 | 0.667 | 0.850 | 0.838 | 0.775 | 0.805 | 0.868 |   0 |   0 |
| `rent.pets_allowed` |  60 | 0.100 | 0.850 | 0.400 | 1.000 | 0.571 |    —  |   0 |   0 |
| `beds` |  60 | 0.700 | 0.867 | 0.833 | 0.952 | 0.889 |    —  |   0 |   0 |
| `room.current_number_of_flatmates` |  60 | 0.083 | 0.867 | 0.143 | 0.200 | 0.167 |    —  |   0 |   0 |
| `baths` |  60 | 0.583 | 0.883 | 0.846 | 0.943 | 0.892 |    —  |   0 |   0 |
| `room.shared_living_room` |  60 | 0.167 | 0.883 | 0.714 | 0.500 | 0.588 |    —  |   0 |   0 |
| `school_mentions` |  60 | 0.150 | 0.883 | 1.000 | 0.222 | 0.364 | 0.222 |   0 |   0 |
| `parking` |  60 | 0.417 | 0.900 | 0.870 | 0.800 | 0.833 | 0.769 |   0 |   0 |
| `rent.wifi_included` |  60 | 0.150 | 0.900 | 0.667 | 0.667 | 0.667 |    —  |   0 |   0 |
| `total_floors` |  60 | 0.083 | 0.917 | 0.500 | 0.200 | 0.286 |    —  |   0 |   0 |
| `amenities_facilities` |  60 | 0.100 | 0.917 | 0.333 | 0.333 | 0.333 | 0.627 |   0 |   0 |
| `rent.deposit_amount` |  60 | 0.167 | 0.917 | 0.727 | 0.800 | 0.762 |    —  |   0 |   0 |
| `room.room_type` |  60 | 0.250 | 0.917 | 0.765 | 0.867 | 0.812 |    —  |   0 |   0 |
| `living_rooms` |  60 | 0.550 | 0.933 | 0.889 | 0.970 | 0.928 |    —  |   0 |   0 |
| `floors_occupied` |  60 | 0.333 | 0.933 | 0.857 | 0.900 | 0.878 |    —  |   0 |   0 |
| `rent.deposit_weeks` |  60 | 0.083 | 0.933 | 0.667 | 0.400 | 0.500 |    —  |   0 |   0 |
| `room.desired_occupation` |  60 | 0.067 | 0.933 | 0.500 | 0.500 | 0.500 |    —  |   0 |   0 |
| `rent.rental_support_accepted` |  60 | 0.050 | 0.950 | 0.500 | 0.333 | 0.400 |    —  |   0 |   0 |
| `rent.student_friendly` |  60 | 0.133 | 0.950 | 0.778 | 0.875 | 0.824 |    —  |   0 |   0 |
| `room.couples_allowed` |  60 | 0.067 | 0.950 | 0.571 | 1.000 | 0.727 |    —  |   0 |   0 |
| `flags.is_build_to_rent` |  60 | 0.033 | 0.950 | 0.333 | 0.500 | 0.400 |    —  |   0 |   0 |
| `commute_claims` |  60 | 0.117 | 0.950 | 0.667 | 0.857 | 0.750 | 0.711 |   0 |   0 |
| `total_floor_area_sqm` |  60 | 0.067 | 0.967 | 0.667 | 1.000 | 0.800 |    —  |   0 |   0 |
| `rent.tenancy_max_months` |  60 | 0.017 | 0.967 | 0.333 | 1.000 | 0.500 |    —  |   0 |   0 |
| `sale.tenure_type` |  60 | 0.050 | 0.967 | 0.600 | 1.000 | 0.750 |    —  |   0 |   0 |
| `room.desired_gender` |  60 | 0.000 | 0.967 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.family_friendly` |  60 | 0.000 | 0.967 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `build_year_min` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `build_year_max` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `window_orientation` |  60 | 0.033 | 0.983 | 0.667 | 1.000 | 0.800 |    —  |   0 |   0 |
| `rent.tenancy_min_months` |  60 | 0.067 | 0.983 | 0.800 | 1.000 | 0.889 |    —  |   0 |   0 |
| `rent.smokers_allowed` |  60 | 0.083 | 0.983 | 0.833 | 1.000 | 0.909 |    —  |   0 |   0 |
| `sale.ground_rent_amount` |  60 | 0.000 | 0.983 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.service_charge_amount` |  60 | 0.033 | 0.983 | 0.500 | 0.500 | 0.500 |    —  |   0 |   0 |
| `room.current_gender` |  60 | 0.067 | 0.983 | 1.000 | 0.750 | 0.857 |    —  |   0 |   0 |
| `room.current_occupation` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `room.current_pets` |  60 | 0.017 | 0.983 | 0.500 | 1.000 | 0.667 |    —  |   0 |   0 |
| `room.desired_age_min` |  60 | 0.017 | 0.983 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_chain_free` |  60 | 0.050 | 0.983 | 1.000 | 0.667 | 0.800 |    —  |   0 |   0 |
| `flags.is_cash_only` |  60 | 0.017 | 0.983 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_hmo` |  60 | 0.033 | 0.983 | 1.000 | 0.500 | 0.667 |    —  |   0 |   0 |
| `flags.is_new_build` |  60 | 0.033 | 0.983 | 1.000 | 0.500 | 0.667 |    —  |   0 |   0 |
| `flags.wheelchair_access` |  60 | 0.017 | 0.983 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `rent.bills_amount` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `sale.length_of_lease_years` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `sale.is_shared_ownership` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.purchase_scheme` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.is_ensuite` |  60 | 0.083 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.current_age_min` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.current_age_max` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.current_smokers` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.desired_age_max` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.vegetarian_household` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.lgbt_household` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_auction` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.has_planning_permission` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_tenanted` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.has_commercial_tenant` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_warehouse_conversion` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_graded_building` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |

Notes: `cov` = fraction of gold rows with a value present; `acc` = exact-match over all rows (tn included); `F1` = field-presence F1 (treats partial-value mismatches as errors); `set_F1` = member-level F1 for list fields; `vocab!` = predictions outside the controlled vocabulary; `case!` = predictions that only differ in case.
