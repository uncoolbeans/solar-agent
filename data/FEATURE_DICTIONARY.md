# One-hour-ahead NSW1 price feature dictionary

Each row is a forecast made at the **start** of five-minute interval `t`, keyed by `decision_start_utc`. Its label is the realised NSW1 wholesale price of `[t+60 minutes, t+65 minutes)`, stored as `target_spot_aud_per_kwh` and keyed by `target_interval_start_utc`. AEMO's fixed-AEST `SETTLEMENTDATE` is an **interval end** label; ingestion subtracts five minutes and converts to UTC before making either key. The label is an outcome and is never a predictor.

**Availability assumption:** At `t`, we use the latest completed five-minute price and regional demand, from `[t-5, t)`, and assume they have been published with **zero extra latency**. Thus `lag_5m` is the newest reading and `lag_15m` starts at `t-15`. This is an optimistic benchmark. Monthly AEMO files may contain later revisions and do not establish the first public availability time for each record. A deployable model needs publication timestamps and an actual delay rule; these results are not a live information guarantee. No realised temperature, irradiance, cloud cover, solar proxy, household load, tariff price, or current/future wholesale observation is used as a feature.

All price features and the target are **AUD/kWh** (`RRP` AUD/MWh divided by 1,000). Regional demand is **MW**, not household kW. No clipping removes negative or extreme prices. A `lag_m` means the interval beginning `t-m` minutes, including throughout daylight saving changes. Trailing windows end just before `t`, use only completed intervals, and use population standard deviation (`ddof=0`).

| Feature | Unit | Construction and earliest required observation |
| --- | --- | --- |
| `hour_sin`, `hour_cos` | unitless | Sine/cosine of Sydney local hour plus minute divided by 24; calendar known at `t` |
| `weekday_sin`, `weekday_cos` | unitless | Sine/cosine of Sydney weekday (Monday 0) divided by 7; calendar known at `t` |
| `local_month` | 1–12 | Sydney month at `t` |
| `local_day_of_month` | 1–31 | Sydney day at `t` |
| `local_day_of_year` | 1–366 | Sydney ordinal day at `t` |
| `is_weekend` | 0/1 | Sydney Saturday or Sunday at `t` |
| `is_dst` | 0/1 | Sydney daylight saving status at `t` |
| `local_fold` | 0/1 | Second occurrence of repeated Sydney clock hour when DST ends |
| `price_lag_5m_aud_per_kwh` | AUD/kWh | Price for `[t-5,t)`; latest completed interval |
| `price_lag_15m_aud_per_kwh` | AUD/kWh | Price for `[t-15,t-10)` |
| `price_lag_30m_aud_per_kwh` | AUD/kWh | Price for `[t-30,t-25)` |
| `price_lag_60m_aud_per_kwh` | AUD/kWh | Price for `[t-60,t-55)` |
| `price_lag_180m_aud_per_kwh` | AUD/kWh | Price for `[t-180,t-175)` |
| `price_lag_360m_aud_per_kwh` | AUD/kWh | Price for `[t-360,t-355)` |
| `price_lag_720m_aud_per_kwh` | AUD/kWh | Price for `[t-720,t-715)` |
| `price_lag_1440m_aud_per_kwh` | AUD/kWh | Price for `[t-1440,t-1435)`; filled using Dec 31 2022 warm-up on Jan 1 2023 |
| `price_trailing_60m_mean_aud_per_kwh`, `price_trailing_60m_std_aud_per_kwh`, `price_trailing_60m_min_aud_per_kwh`, `price_trailing_60m_max_aud_per_kwh` | AUD/kWh | Mean, population standard deviation, minimum and maximum of 12 completed prices over `[t-60,t)` |
| `price_trailing_360m_mean_aud_per_kwh`, `price_trailing_360m_std_aud_per_kwh`, `price_trailing_360m_min_aud_per_kwh`, `price_trailing_360m_max_aud_per_kwh` | AUD/kWh | Same four statistics over 72 completed intervals, `[t-360,t)` |
| `price_trailing_1440m_mean_aud_per_kwh`, `price_trailing_1440m_std_aud_per_kwh`, `price_trailing_1440m_min_aud_per_kwh`, `price_trailing_1440m_max_aud_per_kwh` | AUD/kWh | Same four statistics over 288 completed intervals, `[t-1440,t)` |
| `price_change_5m_vs_15m_aud_per_kwh` | AUD/kWh | `price_lag_5m - price_lag_15m` |
| `price_change_5m_vs_60m_aud_per_kwh` | AUD/kWh | `price_lag_5m - price_lag_60m` |
| `price_change_5m_vs_1440m_aud_per_kwh` | AUD/kWh | `price_lag_5m - price_lag_1440m` |
| `demand_lag_5m_mw` | MW | NSW1 regional `TOTALDEMAND` for `[t-5,t)` |
| `demand_lag_15m_mw` | MW | NSW1 regional demand for `[t-15,t-10)` |
| `demand_lag_60m_mw` | MW | NSW1 regional demand for `[t-60,t-55)` |
| `demand_lag_1440m_mw` | MW | NSW1 regional demand for `[t-1440,t-1435)` |
| `demand_change_5m_vs_15m_mw` | MW | `demand_lag_5m - demand_lag_15m` |
| `demand_change_5m_vs_60m_mw` | MW | `demand_lag_5m - demand_lag_60m` |
| `price_target_aligned_previous_day_aud_per_kwh` | AUD/kWh | Price at `t+60-24h = t-23h`; same target clock time one elapsed day earlier, already completed |
| `price_target_aligned_previous_week_aud_per_kwh` | AUD/kWh | Price at `t+60-7×24h = t-167h`; same target clock time one elapsed week earlier, already completed |

The last two features use **elapsed UTC** days and weeks. Sydney clock time may differ by an hour across a daylight saving switch; local calendar and `fold` represent that distinction. Only Dec 31 2022 was downloaded as warm-up. It fills every required 24-hour lag for Jan 1, but cannot fill the prior-week feature until Jan 6 2023 at 23:00 local. The first 1,716 training forecast origins are therefore excluded rather than filled with future data or a fabricated value.

| Non-feature field | Meaning |
| --- | --- |
| `decision_start_utc` | Unique UTC forecast origin at interval start |
| `target_interval_start_utc` | Exactly 60 minutes after origin; label refers to following five-minute interval |
| `origin_local_date` | Sydney local date of forecast origin |
| `origin_partition`, `target_partition` | Chronological train or validation membership; cross-partition targets are excluded |
| `is_development_origin`, `is_development_target` | Flags for Jan 13–19 2025 |
| `include_in_validation_score` | 1 only if both validation origin and target are outside the development week |
| `target_spot_aud_per_kwh` | Realised target price, never a predictor |

The fixed baseline event rules are price `< 0` for negative prices and `>= $1.00/kWh` for spikes. They were set before validation scoring and have not been tuned against the held-out period. The training-only median baseline is fitted from **training targets only**; persistence copies `price_lag_5m`; the daily seasonal naive baseline copies `price_target_aligned_previous_day`.
