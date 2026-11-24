# SolarAlpha

Simulation-first solar and battery trading experiments for a Sydney household. The project currently uses a standard-library-only simulator and an offline historical replay; it does not control hardware.

## Layout

| Path | Purpose |
| --- | --- |
| `solaralpha/solar_sim.py` | Battery environment, synthetic observations, baseline policies, and evaluation |
| `solaralpha/tariffs.py` | Illustrative household tariff calculations |
| `solaralpha/historical_replay.py` | Offline AEMO price and weather replay |
| `solaralpha/plot_*.py` | SVG chart commands |
| `solaralpha/compare_scenarios.py`, `solaralpha/sensitivity_check.py` | Synthetic stress and sensitivity commands |
| `solaralpha/spike_sensitivity.py` | Counterfactual check of the historical price spike |
| `solaralpha/decision_time.py` | Filtered agent view and decision-time policy runner |
| `solaralpha/mpc_agent.py` | First simulation-only receding-horizon trading controller |
| `solaralpha/forecasting.py` | Forecast provider interface and persistence benchmark |
| `solaralpha/oracle_diagnostic.py` | Offline forecast-oracle comparison and private future-data provider |
| `solaralpha/horizon_sensitivity.py` | Offline all-oracle planning-horizon and runtime comparison |
| `solaralpha/fetch_historical_sample.py` | Fetches the fixed historical input snapshots if absent |
| `tests/` | Standard-library unit tests |
| `data/`, `plots/` | Historical inputs and generated SVG charts |

## Run from the project root

Use Python 3.10 or newer. The `-m` flag runs a module inside the `solaralpha` package, so its imports resolve from the project root.

```powershell
python -m unittest -v
python -m solaralpha.solar_sim
python -m solaralpha.plot_day --day 1 --policy threshold
python -m solaralpha.historical_replay --plot-day 3
python -m solaralpha.plot_load_comparison
python -m solaralpha.spike_sensitivity
python -m solaralpha.decision_time
python -m solaralpha.mpc_agent --plot-day 3
python -m solaralpha.oracle_diagnostic --plot-day 3
python -m solaralpha.horizon_sensitivity --hours 1 3 6 12 --plot-day 3
python -m solaralpha.horizon_sensitivity --hours 24 --scenario observed --timing-samples 32
```

The historical replay reads local files in `data/raw/`. If they are missing, run `python -m solaralpha.fetch_historical_sample` to download the pinned sample described in [data/README.md](data/README.md). Charts continue to be saved in `plots/`.

The replay compares self-consumption, a threshold rule using the previous completed price, and a price-blind evening export schedule (17:00–20:00 Sydney time). Its tariffs, solar generation, and household demand are still illustrative or modeled as described in [data/README.md](data/README.md).

`solaralpha.decision_time` demonstrates the agent-facing path: a policy receives current SOC and clock time plus only the previous completed interval's solar, load, and modeled prices. The original replay policies remain reference benchmarks; some use recorded current-interval solar/load. Future simulated trading agents should use the filtered runner.

## First trading controller

`solaralpha.mpc_agent` runs a simple model predictive controller through the filtered decision-time path. Every five minutes it repeats the last completed solar, load, and tariff prices across a one-hour forecast, evaluates candidate actions (hold, balance predicted solar and load, maximum charge, maximum discharge), executes only the first request, and plans again at the next interval. It merges planned battery states into 0.25 kWh bins. The planner and environment use the same battery transition function, including the reserve, efficiency, power, and export limits.

The planner maximizes forecast export revenue minus import cost and battery wear, plus an assumed value for energy remaining after its horizon: `0.95 × $0.20 × ending battery kWh`. This matches the terminal-value assumption in the existing evaluation, but is not a calibrated value for the user's home. The forecast is deliberately basic; the current seven-day replay is an engineering check, not evidence that this controller saves money. The `--horizon` option changes the number of five-minute planning steps.

## Offline forecast diagnostic

`solaralpha.oracle_diagnostic` runs the original decision-time self-consumption and threshold rules, the unchanged persistence MPC, and three diagnostic MPC variants. The variants substitute recorded future household buy/sell prices, recorded future solar/load, or both into the same one-hour planner. Only the offline diagnostic modules supply those values; `mpc_action` still receives only the last completed readings. The run repeats on the observed seven-day sample and a counterfactual version whose three extreme spot-price intervals are capped at $1.10/kWh.

The console reports cash flow, net and terminal-adjusted reward, grid imports and exports, AC battery throughput (charge plus discharge), discharge wear cost, and final SOC. It also writes a policy comparison for the chosen `--plot-day` to `plots/oracle_comparison_YYYY-MM-DD_observed.svg` and `plots/oracle_comparison_YYYY-MM-DD_capped.svg`. These oracle cases are diagnostic forecasts within a finite planning horizon, not deployable agents or a whole-episode optimum.

## Planning-horizon diagnostic

`solaralpha.horizon_sensitivity` compares the decision-time rules with all-oracle MPC at selected horizons. One hour is 12 five-minute steps; the original one-hour default and planner search settings remain unchanged. It runs each requested horizon on both the recorded-price replay and the existing counterfactual capped-price replay, then writes `plots/horizon_sensitivity_YYYY-MM-DD_observed.svg` and its capped counterpart. The plots show one selected day's cumulative net reward, SOC, cumulative imports and exports, plus whole-week terminal-adjusted reward versus horizon.

The output includes mean and 95th-percentile wall-clock planning time per decision, whole-episode runtime, and an import breakdown. Imports during battery charging include both household demand and charging; the separate "charging increment" estimates the additional grid draw caused by charging. Imports while holding or discharging supply remaining household demand. At the end of the seven-day snapshot, oracle forecasts use recorded data only through the last interval and persistence beyond it. The `--timing-samples N` option times spread-out decisions at a fixed 50% SOC without running a financial backtest, useful for estimating the cost of longer horizons.

The default horizon comparison runs 1, 3, 6, and 12 hours. A full 24-hour run remains available with `--hours 24`, but was not run for this milestone because the unchanged dynamic program's timing sample suggests roughly six minutes per seven-day scenario, with substantial wall-clock variability. No 24-hour financial result is inferred from that sample; details are in [PROGRESS.md](PROGRESS.md).
