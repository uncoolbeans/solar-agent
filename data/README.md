# Historical input slice

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

The fixed AEST/AEDT offsets are correct for this January slice. A year-round loader will need daylight-saving-aware timezone handling.

Historical replay plots now show **AEMO wholesale spot**, **modeled household buy**, and **modeled household sell** as separate price lines. On 15 January the wholesale median is about $0.096/kWh (9.6 cents), while the $17.50/kWh peak spans three intervals from 14:55 to 15:10 Sydney time. A single full-range chart compresses ordinary prices when a spike this large occurs; the console replay prints the day's median and peak for context.

`python -m solaralpha.spike_sensitivity` makes an in-memory counterfactual by capping wholesale spot at **$1.10/kWh** before recalculating both illustrative spot-linked household prices. That cap changes exactly the three $17.50/kWh intervals in this sample. It leaves the stored AEMO file, solar proxy, modeled load, and battery configuration untouched. The cap is a sensitivity probe, not an estimate of any retailer's actual price cap.
