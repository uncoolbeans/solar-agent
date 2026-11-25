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

## 2026-11-25 — internal terminal battery valuation sensitivity

- The planner now accepts an internal terminal value independently of evaluation. Its search objective remains forecast cash flow minus modeled wear plus `eta_discharge × internal_value × horizon-end stored kWh`. Evaluation always adds `eta_discharge × $0.20/kWh × (episode-end energy - initial energy)` to net reward. The original `$0.20/kWh` defaults, battery physics, candidate actions, 0.25 kWh bins, initial SOC, tariff, historical inputs, and oracle boundary behavior are unchanged. Ordinary `mpc_action` still uses only the filtered decision-time persistence forecast.
- Ran all 18 all-oracle combinations of 1, 6, and 12 hours with internal values `$0.00`, `$0.05`, `$0.10`, `$0.20`, `$0.30`, and `$0.40/kWh` on the observed-price episode. Matched `$0.20` results were reused from the completed horizon experiment as labeled, rounded references; config and episode hashes guard that reuse. Fresh runs and a selected-day trace were checkpointed after each setting in `data/derived/terminal_sensitivity_results.json`.

| Internal value | 1 h adjusted / imports | 6 h adjusted / imports | 12 h adjusted / imports |
| --- | ---: | ---: | ---: |
| $0.00 | $12.93 / 53.49 kWh | $23.60 / 27.26 kWh | $28.11 / 11.33 kWh |
| $0.05 | $22.72 / 28.85 kWh | $26.57 / 15.84 kWh | $29.58 / 5.51 kWh |
| $0.10 | $27.46 / 13.06 kWh | $28.68 / 8.37 kWh | $30.26 / 2.45 kWh |
| $0.20 (prior) | $17.67 / 38.06 kWh | $21.66 / 25.17 kWh | $24.38 / 23.18 kWh |
| $0.30 | $4.66 / 85.25 kWh | $11.75 / 66.45 kWh | $17.43 / 58.04 kWh |
| $0.40 | $0.53 / 95.41 kWh | $8.71 / 77.84 kWh | $15.34 / 66.53 kWh |

- Repeated ten representative configurations with the existing capped-price episode: `$0.20` at all three horizons, `$0.10` and `$0.40` at all three, and `$0.00` at one hour. Adjusted rewards for `$0.10` are $6.96, $8.18, and $9.76 at 1, 6, and 12 hours; the original `$0.20` cases give -$2.83, $1.16, and $3.88; `$0.40` gives -$19.97, -$11.79, and -$5.16. The capped one-hour `$0.00` case gives -$7.57. Every matched capped/observed pair has the same aggregate physical metrics and $20.50 less adjusted reward. Each fresh policy exports 1.25 kWh across the three capped spike intervals, so `1.25 kWh × ($17.50 - $1.10)/kWh = $20.50`; this cap does not independently validate a controller choice.
- In the observed run, `$0.10` gives the highest adjusted reward at every tested horizon. At 12 hours it has $35.23 cash flow, $28.65 net reward, $30.26 adjusted reward, 2.45 kWh imports, 139.08 kWh exports, zero incremental charging-related grid draw, 2.36 kWh imports while holding, 286.31 kWh AC throughput, $6.58 wear, 78.2% final SOC, 20.0% minimum SOC, and a 254.83 s episode runtime. The original `$0.20` twelve-hour case has $24.38 adjusted reward and 23.18 kWh imports; the `$0.40` case has $15.34 adjusted reward, 66.53 kWh imports, and 100% final SOC. Full metrics for all settings, including per-decision time, are in the checkpoint and console output.
- At 18:10 and 18:15 Sydney time on 15 January, the one-hour `$0.40` agent requested and executed hold at 100% SOC, importing 0.136 and 0.154 kWh. Feasible `-1.63` and `-1.85 kW` balance actions would cover the respective loads. At the same SOC and times, `$0.10` selects discharge. Buy prices are `$0.326` and `$0.425/kWh`; the highest sell price in either next-hour window is only `$0.295/kWh`. The `$0.40` internal value plus `$0.05/kWh` discharge wear makes retaining one AC kWh worth about `$0.45`, exceeding both avoided purchases. This supports terminal valuation as the cause of these holds; the action space and reserve do not force them. The 0.25 kWh state approximation remains a possible influence elsewhere and was not changed to isolate it.
- Generated observed and sparse capped adjusted-reward/import heatmaps (untested capped cells marked), observed SOC/action comparisons at 1 and 6 hours, and observed adjusted reward versus AC battery throughput. All seven SVGs parsed. Thirty-six tests pass; the original persistence MPC default numerical results remain unchanged.
- The last 11, 71, and 143 decisions at 1, 6, and 12 hours respectively still pad beyond the episode with persistence. No boundary rule was changed, and those padded forecasts may influence end-of-week actions. The seven-day replay is the same diagnostic dataset used to inspect and choose the tested values; it is not an independent validation set for selecting a final controller. Solar, household load, tariff, wear, and terminal value remain illustrative or modeled. Runtime depends on this machine and the reachable state bins.

Awaiting user review before another milestone.

## 2026-11-25 — whole-episode perfect-foresight benchmark

- Added an **offline-only** mixed-integer linear program in `solaralpha/perfect_foresight.py`, using optional pinned `highspy==1.16.0` (HiGHS; NumPy is a transitive dependency). `perfect_foresight_experiment.py` compares the six requested policies on the unchanged 13–19 January 2025 replay and the existing $1.10/kWh capped counterfactual. The ordinary filtered decision view and deployable agents were not changed.
- Before formulating, inspected `battery_transition`: positive AC kW charges; negative AC kW discharges. At each five-minute interval the model uses continuous charge/discharge, grid import/export, curtailed solar, and battery energy. It enforces the 30 kWh capacity, 20% reserve, 5 kW battery power and grid export caps, 95% charge/discharge efficiencies, and AC energy balance. Binary modes forbid simultaneous battery charge/discharge and grid import/export. Curtailment is allowed only after export reaches 5 kW, matching the simulator even when the sell price is negative. Discharge is additionally limited by the simulator's pre-action export headroom `max(0, export_kw - (solar_kw - load_kw))`.
- The single objective is `sum_t dt × (sell_price_t × export_kw_t - buy_price_t × import_kw_t - $0.05 × discharge_kw_t) + 0.95 × $0.20 × (E_final - E_initial)`. There is no horizon-end incentive at intermediate intervals. The initial energy is 15 kWh. This continuous-action MILP can choose actions outside the MPC's four-action candidate set and 0.25 kWh bins, but each selected net action is replayed through the unchanged `BatteryEnv`.

| Policy | Observed cash / net / adjusted (AUD) | Capped cash / net / adjusted (AUD) | Imports / exports / curtail (kWh) | Throughput (kWh) / wear (AUD) | Final / min SOC | Runtime observed / capped (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision-time self | 20.09 / 16.24 / 18.12 | 1.12 / -2.73 / -0.85 | 8.55 / 136.81 / 18.77 | 172.69 / 3.85 | 82.9% / 33.6% | 0.02 / 0.02 |
| Decision-time threshold | 23.31 / 19.23 / 21.11 | 2.81 / -1.27 / 0.61 | 8.50 / 137.02 / 18.04 | 182.10 / 4.07 | 82.9% / 33.6% | 0.02 / 0.02 |
| Original persistence MPC | 16.84 / 13.38 / 15.29 | -2.95 / -6.41 / -4.50 | 39.93 / 168.71 / 18.91 | 156.43 / 3.46 | 83.5% / 32.5% | 3.36 / 3.46 |
| 12 h oracle MPC, $0.20 internal | 28.66 / 22.47 / 24.38 | 8.16 / 1.97 / 3.88 | 23.18 / 159.17 / 5.86 | 271.34 / 6.19 | 83.4% / 20.2% | 256.10 / 257.53 |
| 12 h oracle MPC, $0.10 internal | 35.23 / 28.65 / 30.26 | 14.73 / 8.15 / 9.76 | 2.45 / 139.08 / 6.01 | 286.31 / 6.58 | 78.2% / 20.0% | 254.83 / 260.30 |
| Whole-episode MILP | 35.71 / 29.09 / 31.00 | 15.21 / 8.59 / 10.50 | 0.83 / 135.80 / 5.96 | 289.29 / 6.61 | 83.4% / 20.0% | 8.15 / 19.71 |

- HiGHS reported `kOptimal`, zero relative MIP gap, and matching incumbent/objective bound at **$30.9958638839** observed and **$10.4958638839** capped. Solver time was 8.08 s observed and 19.64 s capped; total benchmark wall time above includes model setup and simulator replay. The largest per-step energy/flow/action or total-objective discrepancy after replay was `2.82e-13` observed and `1.34e-12` capped. Thus the reported solver solution is achievable in the existing simulator, to numerical tolerance, rather than only a relaxed mathematical upper bound. Solver claims are to HiGHS' numerical tolerances.
- Remaining gap in terminal-adjusted AUD is `perfect - policy`. Opportunity capture is `(policy - threshold) / (perfect - threshold)`. Relative to threshold, the $0.10 twelve-hour oracle captures **92.55%** of the available improvement in each scenario, leaving **$0.737**; the $0.20 case captures **33.05%**, leaving **$6.619**. The threshold itself is **$9.887** below perfect in each scenario. Negative capture for self-consumption and persistence MPC means those policies underperform threshold. The observed best $0.10 oracle's close result is in-sample and not a final controller selection.
- The cap changes only three extreme-price intervals. The benchmark's aggregate physical metrics stay the same and its reward falls by $20.50, confirming that the absolute observed-dollar result depends heavily on those price spikes. In the optimal dispatch, imports are only 0.83 kWh; most are while holding the battery (0.81 kWh), not from grid charging. Keeping stored energy can be preferable to supplying low-priced household demand when later export opportunities and terminal energy value exceed the current purchase cost plus discharge wear.
- Added hand-verifiable tests for negative buy/sell prices and no grid arbitrage, full/reserve batteries, solar surplus/curtailment/export limits, a calculable discharge reward, two-step charge/discharge, terminal adjustment, and replay agreement. All **46** tests pass with the optional optimizer dependency installed; without it, optimizer-specific tests are skipped. Generated selected-day SOC/action/grid import/grid export/cumulative-net-reward SVGs for observed and capped prices; both SVGs parse. Metrics and solver diagnostics are saved in `data/derived/perfect_foresight_results.json`. The $0.10 oracle metrics and day-three trace were reused from the exact, input-hash-checked earlier checkpoint; the rounded $0.20 reference was recomputed.
- This is an **offline hindsight ceiling for the specified simulator and seven-day development replay**, not a forecastable policy or expected household savings. Solar and demand are proxies/models; the spot-linked tariff, wear, battery specification, and $0.20 terminal value are illustrative. The replay was already used to inspect MPC settings, so no independent test-set conclusion follows. The whole-week MILP has no forecast padding because its horizon ends exactly with the dataset; the existing twelve-hour MPC boundary padding remains unchanged.

Awaiting user review before another milestone.
