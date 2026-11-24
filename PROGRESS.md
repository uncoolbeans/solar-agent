# SolarAlpha progress

## 2026-11-24 — simulator review

- Ran `python solar_sim.py` with Python 3.14.7, default configuration, seed 42, and 14 synthetic summer days (4,032 five-minute intervals).
- Average per day: 39.8 kWh solar and 22.8 kWh household load.
- Self-consumption: 0.00 kWh imports, 209.51 kWh exports, $-10.56 cash flow, $7.40 wear, $-17.95 net reward, 78.99% final SOC, $-16.30 terminal-adjusted reward.
- Threshold trader: 2.29 kWh imports, 215.35 kWh exports, $5.12 cash flow, $8.20 wear, $-3.09 net reward, 62.32% final SOC, $-2.38 terminal-adjusted reward.
- Independently checked 16,128 intervals across both policies and extreme action requests: executed power and SOC limits, AC energy balance, export cap, import/export exclusivity, and reward accounting passed. Separate charge, discharge, and export-curtailment examples passed. These checks were run ad hoc and have not yet been saved as automated tests.
- Main limitations: synthetic tariffs and profiles; no forecast or real equipment parameters; terminal battery value uses an assumed $0.20/kWh sell price, so adjusted policy comparisons depend on that assumption. The abundant synthetic solar yielded zero imports under self-consumption.

## 2026-11-24 — repeatable simulator tests

- Added `test_solar_sim.py` with four standard-library tests for seeded data, charge/discharge efficiency and sign, action clipping at battery and grid limits, and energy/reward accounting on the same scenario across four action policies.
- `python -m unittest -v`: 4 tests passed. `python solar_sim.py`: baseline results unchanged.

## 2026-11-24 — one-day visualization

- Added `plot_day.py`, a standard-library SVG plotter with `--day`, `--seed`, and `--policy` options. It runs earlier days before plotting a later day so the battery's starting SOC carries forward.
- Generated `day_01_threshold.svg` for seed 42. Day 1 totals: 0.92 kWh import, 6.37 kWh export, 0.00 kWh curtailment, $-0.56 net reward, 77.0% final SOC.
- Parsed the SVG successfully: eight traces of 288 intervals each. All four simulator tests still pass.
- Removed `PROGRESS.md` from `.gitignore` so this log can be tracked when the project is under version control.

The user reviewed the one-day plots and found them suitable.

## 2026-11-24 — plot output folder

- `plot_day.py` now creates `plots/` and writes generated SVGs there. `--output` accepts a filename within that folder.
- Moved the existing day 1 threshold and day 2 self-consumption SVGs into `plots/`. Regenerated both successfully; all four simulator tests pass.

## 2026-11-24 — controlled low-solar stress case

- Added `compare_scenarios.py`: it halves only synthetic solar output across the same 14 days (seed 42), leaving load, prices, battery configuration, and initial SOC unchanged. This is a controlled stress test, not a weather forecast.
- Added minimum observed SOC to `evaluate` for reserve diagnostics. With 50% solar, both policies reach the 20% reserve; self-consumption imports 46.03 kWh and threshold trading imports 70.68 kWh over 14 days.
- The baseline printed results are unchanged. `python compare_scenarios.py` runs successfully and all four tests pass.

The user reviewed the stress-case results and requested a sensitivity check.

## 2026-11-24 — one-at-a-time sensitivity check

- Added `sensitivity_check.py`: ten fixed seeds (42-51), 14-day episodes, and separate probes for solar output, sell-price offset, discharge wear, and terminal energy value. Both policies see identical observations and initial SOC within each seed/case.
- Made `evaluate` accept `terminal_value_per_kwh` (default $0.20/kWh), preserving its prior default output. Added a test for the terminal adjustment; five tests pass.
- Threshold trading had a positive terminal-adjusted advantage in 10/10 seeds for every tested case. Mean advantage was $16.29 per 14-day episode in the reference case; changing the sell-price offset from -$0.10 to +$0.10/kWh moved it from $14.95 to $23.75. These probes are arbitrary and all use the same synthetic data-generating family, so they do not establish real-market profitability.

The user reviewed the sensitivity results and approved weather variation.

## 2026-11-24 — persistent weather and within-day storms

- `make_data` now uses a separate seeded weather stream. Clear, mixed, and overcast day regimes persist probabilistically across days; an occasional daytime storm sharply lowers solar and then recovers more gradually. `weather_variation=False` reproduces the earlier solar series, while load and prices remain identical in both modes.
- Seed 43, day 4 provides a clear-to-storm example: solar falls from 3.60 to 0.28 kW between 14:40 and 15:10. Generated `plots/day_04_seed43_threshold.svg` and refreshed the earlier day 1 and day 2 plots for the new default weather.
- New default seed 42 averages 24.3 kWh/day solar and 22.8 kWh/day load. Self-consumption: 38.99 kWh imports, $-18.94 net reward, 35.61% final SOC. Threshold trader: 61.55 kWh imports, $-5.10 net reward, 20.09% final SOC. Both policies reach the 20% reserve during the episode.
- Previous baseline and sensitivity figures above refer to the earlier generator. The six current tests pass; the prior baseline is reproducible with `weather_variation=False`.
- Weather regime probabilities, storm timing, and attenuation are illustrative. Prices and household load do not yet respond to storms, so financial results remain synthetic and uncalibrated.

Awaiting review of the storm plot and new baseline before another simulator milestone.

The user reviewed the storm plots and requested a small historical price/weather sample.

## 2026-11-24 — seven-day historical input replay

- Downloaded a fixed January 2025 AEMO NSW1 price CSV and Open-Meteo Sydney hourly weather snapshot into `data/raw/`; source URLs, SHA-256 hashes, units, and limitations are documented in `data/README.md`. The selected Sydney-local week is 13–19 January (2,016 five-minute intervals).
- Added `historical_replay.py`. AEMO's AEST interval-end labels are converted to Sydney AEDT interval starts, and AUD/MWh is divided by 1,000. Hourly reanalysis shortwave radiation is mapped to a bounded 10 kW AC solar proxy; household load remains synthetic. Buy/sell prices still use hypothetical wholesale-linked adjustments.
- The threshold rule receives the previous completed interval's price, while rewards use the current recorded interval price. This avoids giving it the current interval's settlement price in advance; publication latency and the hourly weather reconstruction remain limitations.
- Real NSW1 spot prices in this week range from -$48.29 to $17,500.00/MWh. Hybrid replay: self-consumption net $19.21, lagged threshold net $21.35; both results depend on the illustrative retail mapping and are not bill estimates. Generated spike-day and lower-irradiance day SVGs in `plots/`.
- Ten tests pass, including the full five-minute grid, price time/unit conversion, hourly solar mapping, and lagged price signal. Earlier candidate-week snapshots and hindsight plots were placed in ignored exploratory folders.

The user reviewed the historical replay, approved the solar proxy for now, and authorized an illustrative tariff comparison.

## 2026-11-24 — illustrative tariff modeling

- Added `tariffs.py` with fixed buy/sell rates ($0.30/$0.05 per kWh) and a spot-linked example (`spot + $0.17` / `spot - $0.01` per kWh). Both are explicitly hypothetical; the spot-linked example preserves the previous replay mapping.
- `historical_replay.py` now runs self-consumption and the lagged threshold policy under both tariffs on identical physical inputs and initial SOC. For the seven-day sample, fixed-tariff net reward is $3.03 for either policy; spot-linked net reward is $19.21 for self-consumption and $21.35 for lagged threshold. These are illustrative outcomes, not bill estimates.
- Added tariff calculation and fair-input tests; 13 tests pass. The replay writes tariff-labelled plots to `plots/`. Source notes and excluded tariff features are in `data/README.md`.

The user reviewed the tariff model and later approved temperature-responsive modeled demand for the historical replay.

## 2026-11-24 — clarified wholesale and modeled tariff prices

- Checked the 15 January AEMO source: median wholesale spot $0.096235/kWh; $17.50/kWh for three intervals (14:55–15:10 Sydney time). Immediately before the spike, at 14:45, spot was $0.0875/kWh while the illustrative spot-linked buy/sell rates were $0.2575/$0.0775/kWh.
- The historical SVG price panel now labels separate AEMO spot, modeled buy, and modeled sell traces. Replay output prints the selected day's wholesale median and peak plus the illustrative tariff formulas. The spike compresses ordinary values on the full-range axis.
- Added a fixture check for typical and spike prices. Fourteen tests pass; no tariff or reward calculations changed.

## 2026-11-24 — temperature-responsive household demand

- Historical replay demand now combines a 0.60 kW base, morning/evening peaks, seeded five-minute noise, and afternoon cooling that rises above 24°C using the stored Open-Meteo hourly temperature. Coefficients are illustrative; the default synthetic `make_data` demand is unchanged.
- For the seven-day sample, modeled demand averages 24.4 kWh/day. The warmer 15 January (28.2°C max) uses 28.9 kWh; the cooler 17 January (21.2°C max) uses 22.0 kWh. Hot-day and cool-day SVGs were regenerated.
- Fixed-tariff net reward is $2.54 for both policies. Spot-linked net reward is $17.88 for self-consumption and $20.79 for the lagged threshold policy. These remain illustrative because household load and tariff terms are modeled.
- Added reproducibility and temperature-response checks; all 15 tests pass.

Awaiting review before another simulator milestone.

The user noted that the load difference was hard to see in the combined solar/load panels. Added `plot_load_comparison.py`, which overlays the 15 and 17 January five-minute modeled loads on a shared load-only scale and labels daily energy. The underlying replay and policy results are unchanged.

## 2026-11-24 — package layout

- Moved simulator, tariff, replay, plotting, and analysis modules into `solaralpha/`; moved the three test modules into `tests/`. Root `data/` and `plots/` remain in place. Added a root README with the new `python -m solaralpha.<module>` commands.
- Updated imports and project-root paths for historical inputs and chart output. The synthetic simulator, historical replay, scenario comparison, and sensitivity command outputs match the pre-move results exactly. All 15 tests pass; both plot commands run.
- Earlier progress entries record commands as they existed at the time. Use the commands in `README.md` for the current layout.

Awaiting review before another simulator milestone.

## 2026-11-24 — price-blind evening schedule benchmark

- Added `evening_export_schedule`: request up to 5 kW discharge from 17:00 to 20:00 local time; otherwise follow self-consumption. It never reads prices or future observations. The environment still clips requested power at battery reserve, capacity, and grid export limits.
- The seven-day historical replay now compares this schedule with self-consumption and the lagged threshold trader under identical weather, demand, tariffs, initial SOC, and terminal valuation. For the illustrative fixed tariff, adjusted rewards are $4.41 for self-consumption and -$5.41 for the evening schedule. For the illustrative spot-linked tariff, they are $19.75 for self-consumption, $22.66 for lagged threshold, and $17.64 for the evening schedule.
- The schedule increases exports from 127.80 to 170.32 kWh but also causes 21.35 kWh of imports and ends at 39.1% SOC rather than 82.8%. Its spot-linked unadjusted net reward is $18.26 versus self-consumption's $17.88; the terminal adjustment reverses that comparison. The assumed $0.20/kWh terminal value remains a sensitivity, and the seven-day sample is too short for a general profitability claim.
- Added a clock/price-independence test and included this policy in the full energy balance test. Sixteen tests pass. Generated 15 and 17 January schedule plots under both illustrative tariffs.

Awaiting review before another simulator milestone.

## 2026-11-24 — historical spike sensitivity

- After review of the evening schedule, added `solaralpha/spike_sensitivity.py`. Its default $1.10/kWh counterfactual wholesale cap changes only the three $17.50/kWh intervals on 15 January. Both modeled buy and sell prices are recalculated; the raw AEMO snapshot, solar proxy, load, and battery setup remain unchanged.
- With the illustrative spot-linked tariff, terminal-adjusted rewards in the observed replay are $19.75 for self-consumption, $22.66 for lagged threshold, and $17.64 for the evening schedule. With the cap they fall to $0.78, $2.16, and -$1.34. The threshold advantage over self-consumption shrinks from $2.91 to $1.38.
- Physical import/export totals and final SOC remain identical across the observed and capped runs for each policy, so the difference here comes from price settlement. This is a controlled sensitivity check, not a historical correction or household bill prediction.
- Added a test for unchanged physical observations and dispatch plus exactly three price changes; all 17 tests pass.

Awaiting review before another simulator milestone.

## 2026-11-24 — decision-time agent view

- Added `solaralpha/decision_time.py` with an immutable `DecisionView` and a filtered evaluator. Policies receive current interval start time and SOC, plus solar, load, and modeled prices from the previous completed five-minute interval. They cannot read the current raw `Observation` through this runner; the existing `BatteryEnv` still settles energy and reward using the current interval's actual values.
- Added two small decision-time rules for demonstration: previous-interval self-consumption and the previous-price threshold with a previous-power fallback. Existing replay policies and outputs are preserved as reference benchmarks; some of those references use current recorded solar/load.
- On the seven-day illustrative spot-linked replay, terminal-adjusted rewards are $19.75 for the existing same-interval self-consumption reference, $18.12 for decision-time self-consumption, and $21.11 for the decision-time threshold. These are still modeled outcomes, not home bill estimates. The assumed one-interval price availability and historical five-minute power averages are limitations.
- Added tests showing that changing a hidden current interval leaves its action unchanged, becomes visible at the next decision, and changes settlement reward. The filtered runner reuses the existing physical accounting. All 19 tests pass.

Awaiting review before the model predictive trading-agent milestone.

## 2026-11-24 — first simulated model predictive controller

- Added `solaralpha/mpc_agent.py`: a one-hour (12 five-minute step) receding-horizon controller using persistence forecasts from the filtered decision-time view. It considers hold, forecast solar-minus-load power, full charge, and full discharge requests; its dynamic program merges battery states into 0.25 kWh bins. The objective sums modeled export revenue minus import cost and wear, plus `0.95 × $0.20 × ending battery kWh`. It executes only the first request and replans at every interval.
- Extracted `battery_transition` from `BatteryEnv.step` so the planner and physical environment use identical capacity, reserve, efficiency, export, curtailment, and reward calculations. The synthetic simulator, historical replay, and decision-time demonstration outputs matched their pre-refactor output exactly.
- On the seven-day illustrative spot-linked replay, adjusted reward is $15.29 for persistence MPC, $18.12 for decision-time self-consumption, and $21.11 for the decision-time threshold. MPC imports 39.93 kWh versus 8.55/8.50 kWh for the two rules. With the three extreme spot intervals counterfactually capped at $1.10/kWh, the corresponding adjusted rewards are -$4.50, -$0.85, and $0.61. The simple persistence forecast and assumed terminal energy value need scrutiny; this result is not a home savings estimate.
- Generated `plots/hybrid_2025-01-15_spot_linked_persistence_mpc.svg` for review. Tests cover forecast timing, initial missing history, action tradeoffs, and the hidden-current-price boundary. All 22 tests pass.

Awaiting review before improving forecasts or the controller.

## 2026-11-24 — offline forecast-oracle diagnostic

- Added a `ForecastProvider` interface and kept `mpc_action` wired to the original persistence forecast. Its seven-day output is unchanged. Only `oracle_diagnostic.py` can combine recorded future observations with persistence; ordinary policies still receive the filtered decision-time view.
- Compared decision-time self-consumption and threshold rules with MPC using persistence for everything, perfect future household buy/sell prices only, perfect future solar/load only, or all recorded future values. Every run uses the same seven-day episode, tariff, initial 50% SOC, battery parameters, one-hour horizon, 0.25 kWh planning bins, candidate actions, wear cost, and $0.20/kWh terminal value.

| Policy | Observed adjusted reward | Capped adjusted reward | Observed imports |
| --- | ---: | ---: | ---: |
| Decision-time self-consumption | $18.12 | -$0.85 | 8.55 kWh |
| Decision-time threshold | $21.11 | $0.61 | 8.50 kWh |
| Persistence MPC | $15.29 | -$4.50 | 39.93 kWh |
| Oracle prices | $15.98 | -$3.82 | 43.45 kWh |
| Oracle solar/load | $17.03 | -$3.47 | 37.93 kWh |
| Oracle all | $17.67 | -$2.83 | 38.06 kWh |

- In the observed replay, perfect prices alone add $0.69 adjusted reward to persistence MPC; perfect solar/load alone add $1.74; all perfect values add $2.38. Even all-perfect one-hour MPC trails the decision-time threshold by $3.44. Its 38.06 kWh of imports versus 8.50 kWh for the threshold rule, despite only $1.24 more export revenue, is a concrete source of the financial gap. Forecast error is therefore not the only issue; the short planning horizon, fixed terminal value, candidate actions, and energy-bin approximation remain possible contributors, not yet isolated causes.
- Repeated the experiment with the existing $1.10/kWh spot cap, changing only three spike intervals and recalculating the illustrative tariff. The policy ranking persists. The large fall in all rewards shows dependence on the extreme spike for absolute returns; it is not a prediction of actual household bills.
- The runner reports cash flow, net and terminal-adjusted reward, import/export energy, executed AC charge-plus-discharge throughput, discharge wear cost, and final SOC. Generated `plots/oracle_comparison_2025-01-15_observed.svg` and `plots/oracle_comparison_2025-01-15_capped.svg` with price, SOC, executed action, and cumulative daily reward panels. Parsed both SVGs successfully. All 27 tests pass, including forecast-field isolation, no future-data leak to persistence, benchmark agreement, and accounting checks.
- Limitations: the week is short; solar is a weather-derived proxy, load is modeled, and the spot-linked tariff, wear cost, and terminal value are illustrative. Partial-oracle cases hold at the first interval because there is no completed reading for their persistence fields. Forecast steps beyond the recorded episode fall back to persistence, and the all-oracle case is still an approximate, finite-horizon MPC rather than a whole-episode optimum.

Awaiting user review before another milestone.

## 2026-11-25 — all-oracle planning-horizon sensitivity

- Made the offline oracle comparison accept an explicit planning horizon while retaining 12 five-minute steps (one hour) as its default. Added `horizon_sensitivity.py` to compare 1, 3, 6, and 12 hours on the unchanged 13–19 January historical replay and the existing three-interval $1.10/kWh spot-cap counterfactual. Every case uses the same initial SOC, battery physics, candidate actions, 0.25 kWh state bins, spot-linked illustrative tariff, wear cost, and $0.20/kWh internal and evaluation terminal values. Ordinary `mpc_action` still uses decision-time persistence; recorded future data is accessible only in the offline oracle provider.

| Policy | Observed adjusted reward | Capped adjusted reward | Imports | Mean planning ms/decision observed / capped | Episode runtime observed / capped |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision-time self | $18.12 | -$0.85 | 8.55 kWh | <0.01 / <0.01 | 0.02 / 0.02 s |
| Decision-time threshold | $21.11 | $0.61 | 8.50 kWh | <0.01 / <0.01 | 0.02 / 0.03 s |
| All-oracle MPC, 1 h | $17.67 | -$2.83 | 38.06 kWh | 1.09 / 1.57 | 2.22 / 3.20 s |
| All-oracle MPC, 3 h | $19.42 | -$1.08 | 31.73 kWh | 13.79 / 14.08 | 27.87 / 28.46 s |
| All-oracle MPC, 6 h | $21.66 | $1.16 | 25.17 kWh | 28.64 / 27.78 | 57.78 / 56.04 s |
| All-oracle MPC, 12 h | $24.38 | $3.88 | 23.18 kWh | 75.44 / 81.90 | 152.13 / 165.18 s |

- The 12-hour oracle beats the threshold rule by $3.27 adjusted reward in both scenarios; the one-hour oracle trails it by $3.44. Longer planning reduces imports but increases battery throughput and wear. At 12 hours, throughput is 271.34 kWh and wear $6.19, compared with 160.89 kWh and $3.57 at one hour. Minimum SOC falls from 32.3% to 20.2%, still above the 20% reserve.
- Import attribution uses executed battery power, not requested action. At one hour, 17.31 kWh of imports occur in charge intervals, of which 15.16 kWh are incremental grid draw caused by charging; 20.75 kWh occur while holding. At 12 hours, those figures are 6.66, 5.84, and 16.52 kWh. No oracle imports occur while discharging, at the reserve, or while a discharge request is clipped in these replays. Thus depletion does not explain the excess; charging and hold decisions do. The household demand portion of imports during charging is separated from the charging increment.
- Full 24-hour runs were not completed because of the unchanged Python dynamic-programming search cost. Thirty-two spread-out decisions at fixed 50% SOC averaged 176.78 ms observed and 183.80 ms capped, suggesting roughly 356–371 s per 2,016-decision episode. This estimate is uncertain: full observed 12-hour runs varied from 152 to 230 s on this machine. These are runtime probes, not 24-hour financial backtests. The planner expands up to four action candidates across reachable 0.25 kWh battery bins at every forecast step, at every decision; no action, resolution, valuation, or optimization algorithm was changed to reduce runtime.
- Beyond the recorded week, forecast slots use persistence rather than unseen data. This affects the final 11, 35, 71, and 143 decisions for 1, 3, 6, and 12 hours respectively; the 24-hour case would affect 287 decisions. The oracle is perfect only for recorded intervals and remains an approximate finite-horizon planner, not a whole-week optimum. The single week, modeled solar/load, illustrative tariff, terminal value, and machine-dependent runtimes limit generalization.
- Generated `plots/horizon_sensitivity_2025-01-15_observed.svg` and its capped counterpart with daily cumulative net reward, SOC, imports, exports, and whole-week adjusted reward versus horizon. Both SVGs parsed successfully. All 32 tests pass, including horizon conversion, dataset-edge padding, unchanged one-hour benchmark, import attribution, and runtime sampling.

Awaiting user review before another milestone.
