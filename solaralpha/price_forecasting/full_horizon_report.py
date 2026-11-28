"""Render full-horizon benchmark tables from saved offline diagnostics."""

import json
from statistics import mean

from solaralpha.historical_pipeline import ROOT
from .full_horizon_dataset import HORIZONS
from .full_horizon_direct import OUTPUT_DIR
from .full_horizon_evaluation import NAMES


REPORT = ROOT / "data/FULL_HORIZON_RESULTS.md"


def _f(value, digits=5):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(path=REPORT):
    r = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    selection = r["selection"]
    resources = r["resources"]
    lines = ["# Complete 144-point wholesale price benchmark", "",
             "Prices and errors are AUD/kWh. Each forecast covers five-minute **interval starts** at decision time plus 0, 5, ..., 715 minutes. The +0 settlement is a label only. Results use 43,909 identical July–November 2025 common origins for every horizon. This is reused development evaluation, not an independent test. The held-out summer period was not read.", "",
             "The original 41 decision-time-safe columns, including +60-aligned historical seasonal inputs, remain frozen. Direct models can train on horizon-specific eligible rows; common-origin tuning and all reported evaluation scores use the same full-trajectory intersection. The 72-row TCN excludes 71 additional early training origins. Ten frozen direct XGBoost models were reused after exact feature, parameter and label matching; 134 were fitted with the same V2-A settings.", "",
             "## Equal-horizon mean MAE", "",
             "| Model | January–June tuning | Reused July–November evaluation |", "| --- | ---: | ---: |"]
    for name in NAMES:
        lines.append(f"| {name} | {_f(r['mean_mae']['tune'][name])} | {_f(r['mean_mae']['evaluate'][name])} |")
    lines += ["", f"The lowest tuning mean is **{r['provisional_selection']['selected_by_equal_horizon_tune_mae']}**. The tuned hybrid selects persistence at {selection['choices'].count('persistence')} offsets and direct XGBoost at {selection['choices'].count('direct')}; choices were frozen before reused evaluation. The fixed alternative uses persistence at +0/+5 and direct at the other 142 offsets.", "",
              "## Per-horizon numerical accuracy on matched origins", "",
              "| Horizon min | Model | MAE | RMSE | Median AE | Bias |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        scores = r["splits"]["evaluate"]["horizons"][str(h)]["models"]
        for name in NAMES:
            s = scores[name]
            lines.append(f"| {h} | {name} | " + " | ".join(_f(s[k]) for k in ("mae", "rmse", "median_absolute_error", "mean_bias")) + " |")
    lines += ["", "## Negative and ≥$1/kWh spike detection", "",
              "Precision/F1 are undefined (—) when a model issues no event predictions. Onsets count new events relative to the latest completed price. Complete false-onset counts and tuning counterparts are in the JSON.", "",
              "| Horizon min | Model | Negative P/R/F1 | Negative TP/FP | Negative onset | Spike P/R/F1 | Spike TP/FP | Spike onset |",
              "| ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for h in HORIZONS:
        scores = r["splits"]["evaluate"]["horizons"][str(h)]["models"]
        for name in NAMES:
            s = scores[name]
            n, spike = s["negative_price"], s["spike"]
            trio = lambda item: "/".join(_f(item[k], 3) for k in ("precision", "recall", "f1"))
            lines.append(f"| {h} | {name} | {trio(n)} | {n['true_positive']}/{n['false_positive']} | "
                         f"{s['negative_onset']['detected_onsets']}/{s['negative_onset']['actual_onsets']} | "
                         f"{trio(spike)} | {spike['true_positive']}/{spike['false_positive']} | "
                         f"{s['spike_onset']['detected_onsets']}/{s['spike_onset']['actual_onsets']} |")
    lines += ["", "## Monthly equal-horizon MAE", "",
              "| Sydney target month | " + " | ".join(NAMES) + " |",
              "| --- | " + " | ".join("---:" for _ in NAMES) + " |"]
    for month in (202507, 202508, 202509, 202510, 202511):
        lines.append(f"| {month} | " + " | ".join(_f(mean(r["splits"]["evaluate"]["horizons"][str(h)]["monthly"][str(month)][name]["mae"] for h in HORIZONS)) for name in NAMES) + " |")
    lines += ["", "## Conditional realised-price MAE", "",
              "The following averages give each horizon equal weight within a price range. Counts and every horizon-specific conditional result are in `models/price_full_horizon_v1/results.json`.", "",
              "| Actual price range | " + " | ".join(NAMES) + " |",
              "| --- | " + " | ".join("---:" for _ in NAMES) + " |"]
    labels = tuple(r["splits"]["evaluate"]["horizons"]["0"]["conditional_price_ranges"])
    for label in labels:
        values = []
        for name in NAMES:
            available = [r["splits"]["evaluate"]["horizons"][str(h)]["conditional_price_ranges"][label][name]["mae"]
                         for h in HORIZONS if r["splits"]["evaluate"]["horizons"][str(h)]["conditional_price_ranges"][label]]
            values.append(_f(mean(available)) if available else "—")
        lines.append(f"| {label} | " + " | ".join(values) + " |")
    lines += ["", "## Earlier ten-horizon comparison on the same origins", "",
              "Direct XGBoost must reproduce its old common-origin MAE. The 144-output and ten-output TCNs are different trained heads, so compare their scores without treating them as identical models.", "",
              "| Horizon min | Frozen direct | Dense direct | Ten-output TCN | 144-output TCN |",
              "| ---: | ---: | ---: | ---: | ---: |"]
    for h, item in r["sparse_matched_comparison"].items():
        lines.append(f"| {h} | " + " | ".join(_f(item[k]) for k in ("sparse_direct_mae", "dense_direct_mae", "sparse_tcn_mae", "dense_tcn_mae")) + " |")
    lines += ["", "## Whole-trajectory diagnostics", "",
              "These compare complete 144-point forecasts issued at each decision. Adjacent-difference statistics diagnose direct-model discontinuity; they do not establish trading value.", "",
              "| Model | Mean predicted 12h range | Mean actual range | Range MAE | Peak timing MAE min | Trough timing MAE min | 5-min change MAE | Predicted >10c jumps | Negative-duration MAE min | Longest negative run MAE min |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in r["trajectory_diagnostics"]["twelve_hour_all_common_origins"].items():
        lines.append(f"| {name} | {_f(s['mean_predicted_range_aud_per_kwh'])} | {_f(s['mean_actual_range_aud_per_kwh'])} | {_f(s['mean_absolute_range_error'])} | {_f(s['peak_timing_mae_minutes'], 1)} | {_f(s['trough_timing_mae_minutes'], 1)} | {_f(s['five_minute_change_mae'])} | {_f(s['predicted_jumps_above_10c_fraction'], 4)} | {_f(s['negative_duration_mae_minutes'], 1)} | {_f(s['longest_negative_run_mae_minutes'], 1)} |")
    daily = r["trajectory_diagnostics"]["paired_midnight_noon_local_days"]
    lines += ["", f"For **{daily['count']}** Sydney-local development days with both a midnight and noon forecast, two 144-point trajectories cover the day. This is a paired issued-forecast diagnostic, not one forecast issued for all 24 hours.", "",
              "| Model | Mean predicted daily range | Mean actual daily range | Daily range MAE | Peak timing MAE min | Trough timing MAE min |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, s in daily["models"].items():
        lines.append(f"| {name} | {_f(s['mean_predicted_range_aud_per_kwh'])} | {_f(s['mean_actual_range_aud_per_kwh'])} | {_f(s['mean_absolute_range_error'])} | {_f(s['peak_timing_mae_minutes'],1)} | {_f(s['trough_timing_mae_minutes'],1)} |")
    latency = resources["single_decision_latency"]
    lines += ["", "## Computation", "",
              f"Direct: {resources['direct_models']['reused']} reused + {resources['direct_models']['new']} new models; new-fit time **{resources['direct_models']['new_fit_seconds']:.1f} s**, summed time across all 144 historical fits **{resources['direct_models']['sum_fit_seconds']:.1f} s**, combined saved size **{resources['direct_models']['sum_model_bytes']/2**20:.1f} MiB**. **{resources['direct_models']['selected_last_budget_round']}** direct models selected the final permitted boosting round, so their best attainable tuning score under a larger budget is unknown. TCN: **{resources['tcn']['trainable_parameters']:,} parameters**, selected epoch **{resources['tcn']['best_epoch_one_based']}**, fit **{resources['tcn']['fit_seconds']:.1f} s**, checkpoint **{resources['tcn']['model_bytes']:,} bytes**, training process peak **{resources['tcn']['training_peak_working_set_mib']:.1f} MiB**. Evaluation process peak **{r['process_peak_working_set_mib']:.1f} MiB**.", "",
              "| Model | Prepared 144-point median ms | P95 ms |", "| --- | ---: | ---: |"]
    for name, item in latency.items():
        lines.append(f"| {name} | {_f(item['median_ms'],3)} | {_f(item['p95_ms'],3)} |")
    lines += ["", "Latency excludes model loading and feature construction. These are local CPU measurements, not a live-service guarantee. Learning curves, timestamps, fingerprints, all monthly and conditional values, and compressed matched prediction matrices are saved under `models/price_full_horizon_v1/`.", "",
              "The completed trajectory supplies a shape-compatible offline price forecast for eventual 12-hour MPC research, but it has not been fed into MPC. Historical AEMO revisions and the optimistic completed-price publication assumption still limit live validity. No predispatch source, held-out summer data, RL or hardware was accessed.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(make_report())
