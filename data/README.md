# Historical input slice

**Current expansion:** the January 2023–November 2025 train/validation cache and one-hour forecasting baseline are documented in [FORECASTING_RESULTS.md](FORECASTING_RESULTS.md). The section below records the earlier seven-day replay and January 2025 smoke test.

The offline replay covers **13–19 January 2025**, Sydney local time (AEDT): 2,016 five-minute intervals. Run `python -m solaralpha.historical_replay --plot-day 3` from the project root to compare both example tariffs and write four SVG plots. `python -m solaralpha.fetch_historical_sample` downloads the two source snapshots if they are absent.

| Input | Local snapshot | Source and interpretation |
| --- | --- | --- |
| NSW1 wholesale price | `raw/PRICE_AND_DEMAND_202501_NSW1.CSV` | [AEMO January 2025 NSW1 price and demand CSV](https://www.nemweb.com.au/Reports/CURRENT/WebData/NEM/PRICEANDDEMAND/PRICE_AND_DEMAND_202501_NSW1.CSV). `RRP` is AUD/MWh and `SETTLEMENTDATE` labels the end of a five-minute interval in AEST (UTC+10, no daylight saving). The file contains the full month; the replay selects the stated week. SHA-256: `975e8603457f1370f818c880e18a8a7f0c09794410a04d83a0b6f677dca9906a`. |
| Sydney weather | `raw/sydney_weather_2025-01-12_20.json` | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), queried near 33.87°S, 151.21°E for hourly shortwave radiation, cloud cover, and temperature in UTC. This is **reanalysis**, not a measurement from the home. The radiation value is the preceding hour's mean in W/m². SHA-256: `3c6ddae4e6c7b59339a8b29be180944c05500ba12ccbe93d6256f57b4b45f853`. |

The loader converts AEMO interval-end labels to interval starts, then to AEDT (UTC+11) for this January week. It divides `RRP` by 1,000 to obtain AUD/kWh. The seven local days are continuous and have 288 intervals each.

The replay derives a **solar proxy**, not measured rooftop generation: `min(10 kW, 10 kW × 0.83 × hourly shortwave radiation / 1000 W/m²)`. It repeats each hourly radiation mean over twelve five-minute intervals. Panel tilt, shading, temperature losses, and fast within-hour changes are not represented. The stored cloud-cover field is not yet used.

Household demand is still **modeled**, now using the most recent hourly reanalysis temperature. At local hour `h` and temperature `T` in °C, its cooling term is `max(0, T - 24) × 0.25 × bump(h, 16, 4)` kW, where `bump` is the simulator's smooth Gaussian peak. That term is added to a 0.60 kW base, morning and evening peaks, and seeded five-minute noise (seed 42). The result is floored at 0.15 kW. These coefficients are illustrative, not fitted to the user's home. No future hourly temperature is used for the current five-minute load estimate.

`tariffs.py` contains two **illustrative** per-kWh examples:

| Example | Buy price (AUD/kWh) | Sell price (AUD/kWh) |
| --- | ---: | ---: |
| Fixed | `0.30` | `0.05` |
| Spot linked | `spot + 0.17` | `spot - 0.01` |

Here `spot` is the AEMO price after conversion to AUD/kWh. The spot-linked example preserves the replay's earlier price mapping. These are **not the user's retailer tariffs or bill estimates**. The first model excludes daily supply charges, time-of-use windows, retailer price floors or caps, and export fees; the actual contract rules can be added when available.

The replay's threshold policy sees the previous completed interval's tariff-derived price; the current interval's recorded price is used for reward. This one-interval signal is still idealized, and the weather series is an ex-post reconstruction, so the replay is not a deployable backtest.

The separate `solaralpha.decision_time` runner passes agents the current clock and battery SOC plus the previous completed interval's five-minute solar, load, and modeled prices. At the first interval those previous values are absent. This conservatively treats five-minute averages as available only after completion. It assumes the previous interval's price is published by the next decision; actual retailer publication timing and any live solar telemetry remain to be specified.

The first `solaralpha.mpc_agent` controller uses that filtered view and repeats the previous interval's readings across a one-hour planning horizon. In particular, a price spike is forecast to continue unless a later completed reading replaces it. This persistence forecast is intentionally simple and can be wrong; the controller is evaluated against the recorded current intervals only after it has selected each action.

The fixed AEST/AEDT offsets are correct for this January slice. The new historical pipeline below uses `zoneinfo` for year-round Australia/Sydney handling; it does not change this frozen replay.

Historical replay plots now show **AEMO wholesale spot**, **modeled household buy**, and **modeled household sell** as separate price lines. On 15 January the wholesale median is about $0.096/kWh (9.6 cents), while the $17.50/kWh peak spans three intervals from 14:55 to 15:10 Sydney time. A single full-range chart compresses ordinary prices when a spike this large occurs; the console replay prints the day's median and peak for context.

`python -m solaralpha.spike_sensitivity` makes an in-memory counterfactual by capping wholesale spot at **$1.10/kWh** before recalculating both illustrative spot-linked household prices. That cap changes exactly the three $17.50/kWh intervals in this sample. It leaves the stored AEMO file, solar proxy, modeled load, and battery configuration untouched. The cap is a sensitivity probe, not an estimate of any retailer's actual price cap.

## Derived diagnostic results

`derived/horizon_reference.json` records rounded `$0.20/kWh` all-oracle results from the completed horizon experiment, with hashes of the battery configuration and both replay episodes. The terminal-value sweep checks those hashes before reusing the results. `derived/terminal_sensitivity_results.json` is an incremental checkpoint containing newly computed metrics and one selected day's SOC/action trace per setting; reused references are labeled `prior_rounded` and have no trace. These files are outputs of the fixed historical replay, not additional independent observations or training data.

## January 2025 pipeline smoke test (earlier milestone)

`python -m solaralpha.historical_pipeline` defaults to the **Sydney-local January 2025** month. The approved expansion used `--start-month 2023-01 --end-month 2025-11`. `--plan` shows URLs without network access; `--process-only` verifies cached raw files and reprocesses without network access. The December 2025 through February 2026 held-out months were **not** downloaded.

Raw bytes remain in `raw/`. `raw/source_manifest.json` records URLs, SHA-256, byte counts, source type and response metadata. The old nine-day weather file and January AEMO file are marked as preexisting because their exact retrieval times are unknown. Re-running the downloader validates hashes and does not replace them. The new ERA5 request pins `models=era5`, latitude/longitude, hourly variables and GMT; historical API contents could still be revised upstream, so the local checksum and immutable snapshot establish reproducibility.

`processed/2025-01_realised.csv` has **8,928** unique five-minute UTC interval starts for Sydney-local January, sorted chronologically. `processed/2025-01_quality.json` reports zero missing AEMO or weather intervals, zero duplicates and zero invalid values. `processed/2025-01_summary.json` contains summary statistics and the processed CSV checksum. In this snapshot, median spot price is **$0.076/kWh**, the range is **−$0.99999 to $17.50/kWh**, and **1,355 intervals (15.18%)** have negative prices. The ten intervals above $1/kWh and other extremes are retained. The derived solar proxy and modeled household load use the same equations and seed as the frozen replay, now over a complete January time origin.

`python -m solaralpha.pipeline_diagnostics` writes `processed/2025-01_replay_comparison.json` and two SVGs in `plots/`. All **2,016** 13–19 January timestamps, AEMO spot prices and illustrative buy/sell rates match the frozen seven-day replay exactly. The solar proxy differs by **0.420 kW mean absolute difference** because the new weather source is explicitly ERA5 while the old response used Open-Meteo's unpinned default. Modeled load differs by **0.206 kW mean absolute difference** because temperature differs and the identical seed's random sequence now begins on January 1 instead of January 13. Prior benchmarks remain untouched; the new dataset should not be substituted into their recorded results without a separate, labeled experiment.

The processed `realised_*` fields are outcomes or ex-post reanalysis. The forecasting dataset selectively uses lagged historical price and regional-demand observations under the explicit availability assumption in [FEATURE_DICTIONARY.md](FEATURE_DICTIONARY.md). The January 13?19 development week is excluded from reported validation scores; the held-out test period remains untouched. See [FORECASTING_RESULTS.md](FORECASTING_RESULTS.md) for current coverage and results, [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for realised fields, and [SOURCE_RESEARCH.md](SOURCE_RESEARCH.md) for source timing.
