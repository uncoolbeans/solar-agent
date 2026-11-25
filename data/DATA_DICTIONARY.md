# Five-minute realised dataset dictionary

One row represents the half-open interval `[interval_start_utc, interval_end_utc)`. Rows are sorted by UTC start and are unique. The January smoke test has 8,928 rows, one for every five-minute interval in the Sydney-local calendar month. The CSV is a **realised-history table** for simulation and eventual forecast targets; its `realised_*` columns are not automatically available to an agent at the interval start.

| Column | Unit / meaning | Availability at decision time |
| --- | --- | --- |
| `interval_start_utc` | ISO 8601 UTC, primary key | Known calendar |
| `interval_end_utc` | ISO 8601 UTC, exclusive end | Known calendar |
| `interval_start_sydney` | ISO 8601 Australia/Sydney with offset | Known calendar |
| `local_date` | Sydney calendar date | Known calendar |
| `local_hour` | Sydney hour, 0–23; may repeat or skip on DST days | Known calendar |
| `local_minute` | 0, 5, …, 55 | Known calendar |
| `local_weekday` | Monday=0 through Sunday=6 | Known calendar |
| `utc_offset_minutes` | Sydney offset, normally 600 or 660 | Known calendar |
| `is_dst` | 1 when Sydney is observing daylight saving | Known calendar |
| `local_fold` | 1 for the second occurrence of a repeated local clock time when DST ends; otherwise 0 | Known calendar |
| `partition` | `train`, `validation`, or `held_out_test`, by Sydney date | Dataset governance only |
| `is_development_week` | 1 for 13–19 January 2025 | Dataset governance only |
| `include_in_validation_score` | 0 for the development week even though its partition is validation | Dataset governance only |
| `realised_aemo_rrp_aud_per_mwh` | AEMO NSW1 `RRP`, AUD/MWh, including negative and extreme values | Outcome; publication/revision timing must be checked before feature use |
| `realised_spot_aud_per_kwh` | `RRP / 1000`, AUD/kWh | Outcome |
| `realised_regional_demand_mw` | AEMO `TOTALDEMAND`, regional MW; **not household load** | Outcome |
| `realised_temperature_c` | Open-Meteo ERA5 reanalysis, 2 m temperature at the UTC hour start, °C | Ex-post weather reconstruction |
| `realised_shortwave_radiation_w_m2` | ERA5 preceding-hour mean, keyed by the UTC hour end, W/m² | Ex-post weather reconstruction |
| `realised_cloud_cover_pct` | ERA5 cloud cover at the UTC hour start, 0–100% | Ex-post weather reconstruction |
| `realised_solar_proxy_kw` | Existing `min(solar_kw, solar_kw × 0.83 × radiation / 1000)` AC proxy, kW | Derived outcome, not measured rooftop production |
| `realised_modeled_household_load_kw` | Existing temperature-responsive load equation with seed 42, kW | Derived outcome, not a meter reading |
| `realised_illustrative_buy_aud_per_kwh` | Existing `spot + 0.17` tariff example | Derived outcome, not a retailer quote |
| `realised_illustrative_sell_aud_per_kwh` | Existing `spot - 0.01` tariff example | Derived outcome, not a retailer quote |

The preceding-hour shortwave value is repeated over its twelve five-minute intervals to preserve the existing solar model. Temperature and cloud cover are likewise repeated within each UTC hour. This temporal expansion is for realised simulation data; it does **not** make the value available before the hour has elapsed. No missing observation is filled or interpolated into forecasting features. If source intervals are missing or invalid, preprocessing writes a quality report and refuses to publish a complete CSV.

## Chronological partitions

| Sydney-local dates | Partition | Use |
| --- | --- | --- |
| 2023-01-01 through 2024-12-31 | Train | Future model fitting, after data collection |
| 2025-01-01 through 2025-11-30 | Validation | Future tuning and comparison; exclude 13–19 January from reported scores |
| 2025-12-01 through 2026-02-28 | Held-out test | Isolated final evaluation; no fitting, selection, or exploratory performance comparisons |

Only January 2025 has been downloaded and processed in this milestone. The held-out test period has **not** been downloaded, examined, or used for controller/model selection here. A future decision-time feature builder must attach a verified publication or issuance timestamp to each market or weather forecast source, then join only records available before each decision. Calendar fields above are the only fields currently classified as known at the interval start.
