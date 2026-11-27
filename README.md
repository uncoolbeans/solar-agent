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
| `solaralpha/terminal_sensitivity.py`, `solaralpha/inspect_terminal_holds.py` | Offline terminal-value sweep and interval diagnostics |
| `solaralpha/perfect_foresight.py`, `solaralpha/perfect_foresight_experiment.py` | Offline whole-episode MILP and six-policy comparison |
| `solaralpha/historical_pipeline.py`, `solaralpha/pipeline_diagnostics.py` | Resumable historical ingestion, validation, and development-week diagnostics |
| `solaralpha/price_forecasting/` | Offline one-hour price features, simple forecasting baselines, and validation plots |
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
python -m solaralpha.terminal_sensitivity --scenario observed
python -m solaralpha.plot_terminal_sensitivity
python -m solaralpha.inspect_terminal_holds --hours 1 --value 0.40
python -m solaralpha.historical_pipeline --start-month 2023-01 --end-month 2025-11 --process-only
python -m solaralpha.price_forecasting.run
python -m solaralpha.price_forecasting.xgb_v1
python -m solaralpha.price_forecasting.xgb_plots
python -m solaralpha.price_forecasting.xgb_v2_diagnostics
python -m solaralpha.price_forecasting.xgb_v2
python -m solaralpha.price_forecasting.xgb_v2_plots
python -m solaralpha.price_forecasting.tabular_comparison
python -m solaralpha.price_forecasting.tabular_plots
python -m solaralpha.price_forecasting.sequence_experiment --benchmark
python -m solaralpha.price_forecasting.sequence_experiment --model all
python -m solaralpha.price_forecasting.sequence_plots
```

The forecasting command requires the expanded 2023–November 2025 historical cache and optional Matplotlib from `requirements-forecast.txt` (`python -m pip install -r requirements-forecast.txt`). It produces train/validation CSV.gz files and baseline metrics in `data/features/`, plus PNG plots in `plots/forecasting/`. It never reads the held-out December 2025–February 2026 dates. Feature timing and the publication-delay assumption are documented in [data/FEATURE_DICTIONARY.md](data/FEATURE_DICTIONARY.md), with results in [data/README.md](data/README.md). This offline package is not connected to MPC.

For the fixed XGBoost V1 experiment, install pinned binary dependencies into the **existing** project environment with `.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-ml.txt` on Windows. Then run the two XGBoost commands above. They fit on 2023–2024, use January–June 2025 only for early stopping, and evaluate July–November 2025. They save the model and audit metadata in `data/models/price_xgb_v1/` and PNGs in `plots/forecasting/xgb_v1/`. The December 2025–February 2026 held-out period remains untouched; XGBoost is not wired into the simulator or MPC. See [forecasting results](data/FORECASTING_RESULTS.md) for metrics and limitations.

The three V2 commands audit the **frozen V1**, fit four fixed loss/target variants, and generate comparison plots. V2 writes to `data/models/price_xgb_v2/` and `plots/forecasting/xgb_v2/`; it does not rewrite V1. January–June MAE selects the V2 candidate. July–November scores are labeled **reused development evaluation** because that partition was already inspected during V1 review. V2-A (direct absolute-error regression) has the lowest tuning MAE and lowers reused evaluation MAE from V1's $0.04324/kWh to $0.02947/kWh, while still missing every >=$1/kWh evaluation spike. Full diagnostics and event tradeoffs are in [forecasting results](data/FORECASTING_RESULTS.md). The held-out summer data remains untouched.

The two tabular comparison commands fit Ridge, Lasso, Random Forest, LightGBM and CatBoost on the same 41 features, then compare them with all nine frozen V1/V2/baseline forecasts on matched rows. Install `requirements-ml.txt` into the **existing** virtual environment first; it pins Python 3.14.7/Windows-compatible LightGBM 4.7.0 and CatBoost 1.2.10 plus their resolved dependencies. Results, individual model artifacts and metadata are in `data/models/price_tabular_comparison/`; the machine-readable [forecasting leaderboard](data/models/forecasting_leaderboard.json) covers all 14 models, with plots in `plots/forecasting/tabular_comparison/`. July–November is reused development evaluation, and the held-out summer data remains untouched. See [forecasting results](data/FORECASTING_RESULTS.md) for methodology, scores and limits.

For the first sequence experiment, install pinned PyTorch into the **existing** Python 3.14.7 virtual environment with `.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements-neural.txt`. The installed `torch==2.15.0` wheel is CPU-only on this machine. The short benchmark command estimates runtime before full training. The second command trains fixed compact LSTM, GRU and causal TCN models on lazy 72-row windows, then compares their one-hour-ahead forecasts with the 14 preserved approaches on identical tuning and reused development evaluation origins. Individual `--model lstm`, `--model gru` or `--model tcn` runs can resume an interrupted training sequence; `--evaluate` uses saved models after all three are present. [The prespecified design](data/NEURAL_EXPERIMENT_SPEC.md) explains context windows, gates, receptive field, split rules and exact training settings. Saved results and checkpoints are in `data/models/price_sequence_comparison/`, with charts in `plots/forecasting/sequence_comparison/`. No model is connected to MPC.

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

## Internal terminal-value diagnostic

The planner's internal battery value can now vary independently of evaluation. The planner adds `0.95 × internal_value × ending stored kWh` to its forecast reward; the evaluation adjustment always uses `$0.20/kWh` and the change in stored energy over the whole episode. `mpc_agent` and `oracle_diagnostic` accept `--internal-terminal-value` while retaining the earlier `$0.20/kWh` default. This changes decisions, not the physical transition or evaluation formula.

`solaralpha.terminal_sensitivity` runs all-oracle MPC at 1, 6, and 12 hours for six internal values from `$0.00` to `$0.40/kWh`. It writes each result and a selected-day trace to `data/derived/terminal_sensitivity_results.json` as it completes, so an interrupted sweep resumes without rerunning completed settings. Identical `$0.20` cases reuse the rounded, input-hash-checked results in `data/derived/horizon_reference.json`. Use `--no-reference --pairs 1:0.20` to recompute a reference case when an exact trace is needed. The plots include reward and import heatmaps, SOC/action comparisons, and adjusted reward versus battery throughput. These are diagnostics on the same seven-day replay, not a separate validation set for choosing a final controller.

## Whole-episode perfect-foresight benchmark

This offline experiment uses HiGHS through `highspy` (an optional pinned dependency; it brings NumPy). Create a local virtual environment and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-optimizer.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe -m solaralpha.perfect_foresight
```

The runner solves a mixed-integer linear program over all 2,016 recorded five-minute intervals for observed and capped prices. It allows continuous battery power, enforces the simulator's energy balance, efficiencies, limits, exclusive battery and grid directions, and forced-only solar curtailment. It includes discharge wear and the same **single** $0.20/kWh final-minus-initial battery energy adjustment used by evaluation. It replays the proposed actions through `BatteryEnv`, reports solver status, objective bound, gap, runtime, and replay error, and compares the six policies. Exact prior 12-hour $0.10 MPC results are reused only after input hashes are checked; they are recomputed if that optional checkpoint is absent. The rounded $0.20 reference is always recomputed. Results are saved to `data/derived/perfect_foresight_results.json`, with selected-day SVGs in `plots/`.

The optimiser reads the complete replay and is not a deployable agent. These seven days are a development diagnostic, and the solar proxy, load, tariff, wear, and terminal value remain illustrative.

## Historical data pipeline (earlier January 2025 smoke test)

The default downloader fetches and processes **only Sydney-local January 2025**. It uses AEMO's December 2024 boundary file and January 2025 NSW1 monthly price/demand file, plus one explicitly pinned Open-Meteo ERA5 hourly response. Existing raw bytes are never overwritten; `data/raw/source_manifest.json` records source URLs, SHA-256 checksums, sizes, and download metadata. Repeated runs verify cached bytes and make no network requests. It uses the standard library plus `tzdata` on Windows for `zoneinfo` daylight-saving rules; install the pinned package with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
.\.venv\Scripts\python.exe -m solaralpha.historical_pipeline --plan
.\.venv\Scripts\python.exe -m solaralpha.historical_pipeline
.\.venv\Scripts\python.exe -m solaralpha.historical_pipeline --process-only
.\.venv\Scripts\python.exe -m solaralpha.pipeline_diagnostics
```

The processed UTC-ordered five-minute CSV, quality report, summary statistics and old-replay comparison are in `data/processed/`. The two diagnostic SVGs are in `plots/`. See [data/DATA_DICTIONARY.md](data/DATA_DICTIONARY.md) for units, availability and chronological train/validation/test boundaries, and [data/SOURCE_RESEARCH.md](data/SOURCE_RESEARCH.md) for possible future renewable and issued-forecast feeds. This section documents the earlier smoke test; the train/validation expansion and first baselines are described above and in [data/FORECASTING_RESULTS.md](data/FORECASTING_RESULTS.md).

## AEMO forecast archive feasibility

The offline [AEMO forecast feasibility report](data/AEMO_FORECAST_FEASIBILITY.md) documents official P5MIN and 30-minute predispatch archives, a January 2025 raw-file smoke test, timestamp and publication limitations, and proposed future features. It finds that P5MIN has no forecast for the existing exact five-minute target starting at `t+60` on the sampled decisions. The 30-minute monthly tables do not preserve the issued vintages required for a safe historical join. No forecast model, MPC agent or 17-model leaderboard entry was changed.

To reproduce the bounded smoke test with the existing Python environment (standard library only):

```powershell
.\.venv\Scripts\python.exe -c "from solaralpha.aemo_forecast_archive import fetch_tables; fetch_tables()"
.\.venv\Scripts\python.exe -m solaralpha.aemo_forecast_coverage
.\.venv\Scripts\python.exe -m solaralpha.aemo_forecast_diagnostic
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```
