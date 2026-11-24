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
```

The historical replay reads local files in `data/raw/`. If they are missing, run `python -m solaralpha.fetch_historical_sample` to download the pinned sample described in [data/README.md](data/README.md). Charts continue to be saved in `plots/`.
