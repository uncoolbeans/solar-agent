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

Awaiting the user's review before adding actual tariff terms or another simulator milestone.

## 2026-11-24 — clarified wholesale and modeled tariff prices

- Checked the 15 January AEMO source: median wholesale spot $0.096235/kWh; $17.50/kWh for three intervals (14:55–15:10 Sydney time). Immediately before the spike, at 14:45, spot was $0.0875/kWh while the illustrative spot-linked buy/sell rates were $0.2575/$0.0775/kWh.
- The historical SVG price panel now labels separate AEMO spot, modeled buy, and modeled sell traces. Replay output prints the selected day's wholesale median and peak plus the illustrative tariff formulas. The spike compresses ordinary values on the full-range axis.
- Added a fixture check for typical and spike prices. Fourteen tests pass; no tariff or reward calculations changed.
