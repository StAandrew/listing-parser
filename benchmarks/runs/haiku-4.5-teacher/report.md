# Benchmark: haiku-4.5-teacher

## Headlines

- **Examples:** 60
- **JSON parse rate:** 100.0%
- **Schema validity rate:**  98.3%
- **Macro field accuracy:**  99.3%
- **Evidence grounding rate:**  97.0%

## Per-field

| field | n | cov | acc | prec | rec | F1 | set_F1 | vocab! | case! |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `amenities_interior` |  60 | 0.750 | 0.900 | 0.875 | 0.933 | 0.903 | 0.928 |   3 |   0 |
| `nearby_mentions` |  60 | 0.700 | 0.900 | 0.867 | 0.929 | 0.897 | 0.935 |   0 |   0 |
| `amenities_appliances` |  60 | 0.267 | 0.933 | 0.765 | 0.812 | 0.788 | 0.901 |   1 |   0 |
| `amenities_outdoor` |  60 | 0.667 | 0.950 | 0.974 | 0.925 | 0.949 | 0.945 |   1 |   0 |
| `amenities_facilities` |  60 | 0.100 | 0.967 | 0.667 | 0.667 | 0.667 | 0.969 |   1 |   0 |
| `parking` |  60 | 0.417 | 0.967 | 0.958 | 0.920 | 0.939 | 0.920 |   0 |   0 |
| `baths` |  60 | 0.583 | 0.983 | 0.971 | 0.971 | 0.971 |    —  |   0 |   0 |
| `rent.let_type` |  60 | 0.133 | 0.983 | 1.000 | 0.875 | 0.933 |    —  |   0 |   0 |
| `rent.deposit_weeks` |  60 | 0.083 | 0.983 | 1.000 | 0.800 | 0.889 |    —  |   0 |   0 |
| `rent.deposit_amount` |  60 | 0.167 | 0.983 | 0.909 | 1.000 | 0.952 |    —  |   0 |   0 |
| `room.desired_occupation` |  60 | 0.067 | 0.983 | 0.750 | 0.750 | 0.750 |    —  |   0 |   0 |
| `room.couples_allowed` |  60 | 0.067 | 0.983 | 0.800 | 1.000 | 0.889 |    —  |   0 |   0 |
| `flags.is_build_to_rent` |  60 | 0.033 | 0.983 | 0.667 | 1.000 | 0.800 |    —  |   0 |   0 |
| `property_type` |  60 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `beds` |  60 | 0.700 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `living_rooms` |  60 | 0.550 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `total_floor_area_sqm` |  60 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `total_floors` |  60 | 0.083 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `floors_occupied` |  60 | 0.333 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `build_year_min` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `build_year_max` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `window_orientation` |  60 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.furnish_type` |  60 | 0.500 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.tenancy_min_months` |  60 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.tenancy_max_months` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.bills_included` |  60 | 0.383 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.bills_amount` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.pets_allowed` |  60 | 0.100 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.smokers_allowed` |  60 | 0.083 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.wifi_included` |  60 | 0.150 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.rental_support_accepted` |  60 | 0.050 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.student_friendly` |  60 | 0.133 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `rent.dps` |  60 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `sale.tenure_type` |  60 | 0.050 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `sale.length_of_lease_years` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `sale.ground_rent_amount` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.service_charge_amount` |  60 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `sale.is_shared_ownership` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `sale.purchase_scheme` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.room_type` |  60 | 0.250 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.is_ensuite` |  60 | 0.083 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.listing_for_number_of_rooms` |  60 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.current_number_of_flatmates` |  60 | 0.083 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.current_gender` |  60 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.current_age_min` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.current_age_max` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.current_occupation` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.current_pets` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.current_smokers` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.desired_age_min` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `room.desired_age_max` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.desired_gender` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.family_friendly` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.vegetarian_household` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.lgbt_household` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `room.shared_living_room` |  60 | 0.167 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `flags.is_chain_free` |  60 | 0.050 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `flags.is_cash_only` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `flags.is_hmo` |  60 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `flags.is_auction` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.has_planning_permission` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_tenanted` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.has_commercial_tenant` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_new_build` |  60 | 0.033 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `flags.is_warehouse_conversion` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.is_graded_building` |  60 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 |    —  |   0 |   0 |
| `flags.wheelchair_access` |  60 | 0.017 | 1.000 | 1.000 | 1.000 | 1.000 |    —  |   0 |   0 |
| `commute_claims` |  60 | 0.117 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |   0 |   0 |
| `school_mentions` |  60 | 0.150 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |   0 |   0 |

Notes: `cov` = fraction of gold rows with a value present; `acc` = exact-match over all rows (tn included); `F1` = field-presence F1 (treats partial-value mismatches as errors); `set_F1` = member-level F1 for list fields; `vocab!` = predictions outside the controlled vocabulary; `case!` = predictions that only differ in case.
