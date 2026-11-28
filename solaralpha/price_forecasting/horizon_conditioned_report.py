"""Render the matched shared/direct/baseline diagnostic as Markdown."""

import json

from solaralpha.historical_pipeline import ROOT
from .horizon_conditioned import FORECAST_NAMES, OUTPUT_DIR
from .multi_horizon import HORIZONS


REPORT = ROOT / "data" / "HORIZON_CONDITIONED_RESULTS.md"


def _f(value, n=5):
    return "—" if value is None else f"{value:.{n}f}"


def make_report(path=REPORT):
    r = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    meta = r["model_metadata"]
    lines = ["# Shared horizon-conditioned XGBoost: detailed comparison", "",
             "Prices and errors are AUD/kWh. Every main result uses the **same 43,909 July–November 2025 decision origins** across ten representative horizons. This is reused development evaluation. The full JSON also contains horizon-specific eligible-subset results, source hashes, exact per-horizon inference timings, learning curves and all monthly/hourly and conditional scores.", "",
             "One model learns `price = f(41 frozen features, horizon_minutes)`. It was trained on 2,086,690 rows from 208,669 common 2023–2024 origins and tuned on 498,380 rows from 49,838 common January–June 2025 origins. Each origin supplies ten horizon-major rows. The input seasonal columns still refer to the old +60-minute clock; they were not redefined. The selected iteration minimizes the arithmetic mean of ten per-horizon tuning MAEs. The horizon-dependent hybrid's choice among persistence, direct XGBoost and shared XGBoost was frozen using the same common tuning set **before** reused evaluation.", "",
             "## Frozen tuning choice and reused evaluation MAE", "",
             "| Horizon min | Tuning choice | Persistence MAE | Direct MAE | Shared MAE | Hybrid MAE | Shared minus direct |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        models = r["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"]
        chosen = r["hybrid_selection"]["horizons"][str(h)]["chosen"]
        lines.append(f"| {h} | {chosen} | " + " | ".join(_f(models[name]["mae"]) for name in ("persistence", "direct", "shared", "hybrid")) + f" | {_f(models['shared']['mae'] - models['direct']['mae'])} |")
    lines += ["", "Mean of ten common-origin MAEs: " + ", ".join(
        f"{name} {_f(r['mean_common_mae']['evaluate'][name])}" for name in FORECAST_NAMES) + ".", "",
              "## Numeric accuracy on common origins", "",
              "| Horizon min | Model | MAE | RMSE | Median AE | Bias |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        models = r["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"]
        for name in FORECAST_NAMES:
            s = models[name]
            lines.append(f"| {h} | {name} | " + " | ".join(_f(s[k]) for k in ("mae", "rmse", "median_absolute_error", "mean_bias")) + " |")
    lines += ["", "## Negative-price and spike events on common origins", "",
              "Precision is undefined (—) when the model makes no event prediction. Onset detection is detected/actual new-event intervals relative to the latest completed price. False onset signals and all underlying counts are in the JSON.", "",
              "| Horizon min | Model | Negative P/R/F1 | Negative TP/FP | Negative onset | Spike P/R/F1 | Spike TP/FP | Spike onset |", "| ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for h in HORIZONS:
        models = r["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"]
        for name in FORECAST_NAMES:
            s = models[name]; negative = s["negative_price"]; spike = s["spike"]
            trio = lambda event: "/".join(_f(event[k], 3) for k in ("precision", "recall", "f1"))
            no = s["negative_onset"]; so = s["spike_onset"]
            lines.append(f"| {h} | {name} | {trio(negative)} | {negative['true_positive']}/{negative['false_positive']} | {no['detected_onsets']}/{no['actual_onsets']} | {trio(spike)} | {spike['true_positive']}/{spike['false_positive']} | {so['detected_onsets']}/{so['actual_onsets']} |")
    lines += ["", "## Monthly MAE on common origins", "",
              "| Horizon min | Sydney target month | Persistence | Direct | Shared | Hybrid |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        monthly = r["horizons"][str(h)]["splits"]["evaluate"]["common"]["monthly"]
        for month, models in monthly.items():
            lines.append(f"| {h} | {month} | " + " | ".join(_f(models[name]["mae"]) for name in ("persistence", "direct", "shared", "hybrid")) + " |")
    lines += ["", "## Conditional MAE by realised price", "",
              "Conditional sets can differ by target offset; their counts are shown. Every other baseline and the eligible-subset counterparts are in the JSON.", "",
              "| Horizon min | Actual price range | Count | Persistence | Direct | Shared | Hybrid |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        ranges = r["horizons"][str(h)]["splits"]["evaluate"]["common"]["conditional_price_ranges"]
        for label, models in ranges.items():
            if models:
                lines.append(f"| {h} | {label} | {models['shared']['count']} | " + " | ".join(_f(models[name]["mae"]) for name in ("persistence", "direct", "shared", "hybrid")) + " |")
    lines += ["", "## Feature-group total gain", "",
              "Percentages are the share of total tree split gain *within each model*. Group sizes differ, the shared model has one extra horizon feature, and these importances do not establish causality.", "",
              "| Model | Historical price | Regional demand | Calendar | Horizon |", "| --- | ---: | ---: | ---: | ---: |"]
    importance = r["importance"]
    for name, group in [(f"+{h}m direct", importance["direct"][str(h)]["fraction_by_group"]) for h in HORIZONS] + [("shared", importance["shared"]["fraction_by_group"])]:
        lines.append(f"| {name} | " + " | ".join(_f(100 * group[key], 2) + "%" for key in ("price_history", "regional_demand", "calendar", "horizon")) + " |")
    lines += ["", "Shared model's ten highest total-gain features: " + ", ".join(importance["shared"]["top_ten"]) + ".", "",
              "## Runtime, memory and size", "",
              f"Shared fit: {_f(meta['fit_seconds'], 2)} s; selected zero-based iteration {meta['best_iteration_zero_based']} of {meta['fitted_rounds']} fitted; selected balanced tuning MAE {_f(meta['best_tune_mean_mae'])}.", "",
              f"Saved shared model: {r['model_size_bytes']['shared']:,} bytes. Ten saved direct models: {r['model_size_bytes']['direct_ten_total']:,} bytes. Measured training-process peak working set: {_f(meta['process_peak_working_set_mib'], 1)} MiB. These times and sizes exclude historical feature construction, data loading, operating-system cache and a live service.", "",
              "Prepared-feature single-decision latency for all ten trained horizons: " + "; ".join(
                  f"{name} median {_f(v['median_ms'], 3)} ms, p95 {_f(v['p95_ms'], 3)} ms over {v['samples']} decisions"
                  for name, v in r["single_decision_inference"].items()) + ". The shared path forms one ten-row matrix; the direct path invokes ten already-loaded models. These local timings are diagnostic, not live latency promises.", "",
              "The ten offsets are sparse. No model was trained or validated at the other 134 five-minute offsets within a 12-hour trajectory. The held-out December 2025–February 2026 data, AEMO predispatch, MPC and hardware are outside this experiment.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(make_report())
