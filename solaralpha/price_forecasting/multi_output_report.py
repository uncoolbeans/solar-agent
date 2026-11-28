"""Render the matched ten-output neural diagnostics as Markdown."""

import json

from solaralpha.historical_pipeline import ROOT
from .multi_horizon import HORIZONS
from .multi_output_evaluation import ALL_NAMES, NEW_NAMES, OUTPUT_DIR, SINGLE_NAMES


REPORT = ROOT / "data" / "MULTI_OUTPUT_RESULTS.md"


def _f(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(path=REPORT):
    r = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    lines = ["# Ten-output sequence price forecasting: detailed comparison", "",
             "All prices and errors are AUD/kWh. Results use **43,909 common July–November 2025 decision origins** at every horizon. This partition has been examined repeatedly and is reused development evaluation, not an independent test. The held-out summer data was not accessed.", "",
             "Each input is a lazy 72 × 41 historical feature window ending at decision time. Each target is a ten-price vector for five-minute interval starts at +0, +5, +15, +30, +60, +120, +180, +360, +540 and +715 minutes. The +0 current settlement is a label only. The 41 features and original +60-aligned seasonal inputs are unchanged; the scaler uses original training feature rows only. The first 71 common training decisions lack a complete sequence, leaving **208,598 train / 49,838 tune / 43,909 evaluation** origins. Every comparison model is rescored on those exact origins.", "",
             "## Equal-horizon mean MAE", "",
             "| Approach | January–June tuning | Reused July–November evaluation |", "| --- | ---: | ---: |"]
    for name in ALL_NAMES:
        lines.append(f"| {name} | {_f(r['mean_mae']['tune'][name])} | {_f(r['mean_mae']['evaluate'][name])} |")
    lines += ["", f"Provisional neural selection by tuning: **{r['selection']['selected_neural']}**. Lowest tuning mean among all listed approaches: **{r['selection']['selected_all_approaches']}**. These choices were frozen before reused evaluation.", "",
              "## Per-horizon numeric accuracy", "",
              "| Horizon min | Approach | MAE | RMSE | Median AE | Bias |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        scores = r["splits"]["evaluate"]["horizons"][str(h)]["models"]
        for name in ALL_NAMES:
            s = scores[name]
            lines.append(f"| {h} | {name} | " + " | ".join(_f(s[k]) for k in ("mae", "rmse", "median_absolute_error", "mean_bias")) + " |")
    lines += ["", "## Negative-price and >=$1/kWh spike events", "",
              "Undefined precision is shown as — when there are no event predictions. Onset is detected/actual new-event intervals relative to the latest completed price. Full false-onset counts and conditional event MAE are in the JSON.", "",
              "| Horizon min | Approach | Negative P/R/F1 | Negative TP/FP | Negative onset | Spike P/R/F1 | Spike TP/FP | Spike onset |", "| ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for h in HORIZONS:
        scores = r["splits"]["evaluate"]["horizons"][str(h)]["models"]
        for name in ALL_NAMES:
            s = scores[name]; n = s["negative_price"]; spike = s["spike"]
            trio = lambda item: "/".join(_f(item[k], 3) for k in ("precision", "recall", "f1"))
            no = s["negative_onset"]; so = s["spike_onset"]
            lines.append(f"| {h} | {name} | {trio(n)} | {n['true_positive']}/{n['false_positive']} | {no['detected_onsets']}/{no['actual_onsets']} | {trio(spike)} | {spike['true_positive']}/{spike['false_positive']} | {so['detected_onsets']}/{so['actual_onsets']} |")
    lines += ["", "## Matched +60-minute single-output neural comparison", "",
              "The old single-output models and new ten-output heads see the same 72 historical feature rows on these origins. Their training target sets and head capacities differ.", "",
              "| Model | Tuning MAE | Reused evaluation MAE |", "| --- | ---: | ---: |"]
    for name in (*SINGLE_NAMES, *NEW_NAMES):
        lines.append(f"| {name} | {_f(r['splits']['tune']['matched_single_output_60'][name]['mae'])} | {_f(r['splits']['evaluate']['matched_single_output_60'][name]['mae'])} |")
    lines += ["", "## Training and inference resources", "",
              "| Architecture | Parameters | Selected epoch | Epochs fitted | Fit seconds | Peak working set MiB | Saved bytes | Prepared ten-output median ms |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("lstm", "gru", "tcn"):
        m = r["models"][name]
        lines.append(f"| {name} | {m['parameter_count']:,} | {m['best_epoch_one_based']} | {m['epochs_fitted']} | {_f(m['training_wall_seconds'], 2)} | {_f(m['process_peak_working_set_mib'], 1)} | {m['model_bytes']:,} | {_f(m['single_decision_inference']['median_ms'], 3)} |")
    lines += ["", "Single-decision times are forward passes on a preassembled, scaled 72 × 41 window; they exclude input construction and model loading. The result JSON has batch inference time, CPU time, the complete learning curves, source/model checksums and all per-horizon monthly, hourly and conditional price-range scores.", "",
              "## Monthly MAE on common origins", "",
              "| Horizon min | Sydney target month | Direct | Shared | Multi-LSTM | Multi-GRU | Multi-TCN |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        monthly = r["splits"]["evaluate"]["horizons"][str(h)]["monthly"]
        for month, scores in monthly.items():
            lines.append(f"| {h} | {month} | " + " | ".join(_f(scores[name]["mae"]) for name in ("direct", "shared", *NEW_NAMES)) + " |")
    lines += ["", "## Conditional MAE by realised price", "",
              "Counts are target-dependent and can differ across horizon offsets. All baseline conditional scores and the tuning counterparts are in the JSON.", "",
              "| Horizon min | Actual price range | Count | Direct | Shared | Multi-LSTM | Multi-GRU | Multi-TCN |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        ranges = r["splits"]["evaluate"]["horizons"][str(h)]["conditional_price_ranges"]
        for label, scores in ranges.items():
            if scores:
                lines.append(f"| {h} | {label} | {scores['direct']['count']} | " + " | ".join(_f(scores[name]["mae"]) for name in ("direct", "shared", *NEW_NAMES)) + " |")
    lines += ["", "The ten forecasts are sparse. Intervening five-minute offsets have not been trained or validated. No MPC backtest, AEMO predispatch integration, RL, hardware or held-out summer evaluation was performed.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(make_report())
