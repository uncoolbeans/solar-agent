"""Render comprehensive matched multi-horizon metrics from saved JSON."""

import json

from solaralpha.historical_pipeline import ROOT
from .multi_horizon import HORIZONS, MODEL_DIR, MODEL_NAMES


OUTPUT = ROOT / "data" / "MULTI_HORIZON_RESULTS.md"


def _number(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def make_report(output_path=OUTPUT):
    result = json.loads((MODEL_DIR / "results.json").read_text(encoding="utf-8"))
    lines = [
        "# Ten-horizon direct price experiment: detailed scores", "",
        "All prices and errors are AUD/kWh. Main tables use the **same 43,909 July–November 2025 decision origins** at all horizons; target intervals naturally differ by offset. This is reused development evaluation, not an independent test. The JSON beside the models also reports each horizon's full eligible subset, 50,000-ish tuning rows, monthly and hourly results, conditional price-range errors, learning curves, exact inference times, and source/model fingerprints.", "",
        "The models use the frozen 41 features, XGBoost V2-A direct absolute-error settings and training-only 2023–2024 labels. Baseline medians are refit on the training labels of each horizon. The +0-minute settlement price is never a feature. Every target is the price of a five-minute interval **starting** at the stated offset.", "",
        "## Eligibility and model runtimes", "",
        "| Horizon min | Train eligible | Tune eligible | Eval eligible | Common eval | Best iteration | XGB fit s | XGB eval batch s |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    manifest = result["target_manifest"]
    for h in HORIZONS:
        item = result["horizons"][str(h)]
        metadata = item["metadata"]
        lines.append(f"| {h} | {manifest['eligible_counts']['train'][str(h)]:,} | {manifest['eligible_counts']['tune'][str(h)]:,} | {manifest['eligible_counts']['evaluate'][str(h)]:,} | {manifest['common_counts']['evaluate']:,} | {metadata['best_iteration_zero_based']} | {_number(metadata['train_seconds'], 2)} | {_number(item['splits']['evaluate']['inference_seconds'], 4)} |")
    lines += ["", "Fit time excludes data loading. Inference is a prepared-matrix batch over each horizon's full eligible evaluation rows. Persistence and previous-day seasonal require no fit; their measured batch lookup times, plus training-only median fitting and lookup times, are saved under `baseline_runtimes_seconds` in `results.json`. These local batch timings do not include constructing live features.", "",
              "## Numeric accuracy on common origins", "",
              "| Horizon min | Model | MAE | RMSE | Median AE | Bias (prediction minus actual) |", "| ---: | --- | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        models = result["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"]
        for name in MODEL_NAMES:
            score = models[name]
            lines.append(f"| {h} | {name} | {_number(score['mae'], 5)} | {_number(score['rmse'], 5)} | {_number(score['median_absolute_error'], 5)} | {_number(score['mean_bias'], 5)} |")
    lines += ["", "## Negative prices and $1/kWh spikes on common origins", "",
              "Precision is undefined (—) when a model never predicts the event. F1 is zero in that case. Onset means a new event relative to the last completed interval; `detected/actual` counts event intervals with an available decision origin.", "",
              "| Horizon min | Model | Negative P/R/F1 | Negative TP/FP | Negative onset detected/actual | Spike P/R/F1 | Spike TP/FP | Spike onset detected/actual |", "| ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for h in HORIZONS:
        models = result["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"]
        for name in MODEL_NAMES:
            score = models[name]
            n, s = score["negative_price"], score["spike"]
            no, so = score["negative_onset"], score["spike_onset"]
            trio = lambda event: "/".join(_number(event[key], 3) for key in ("precision", "recall", "f1"))
            lines.append(f"| {h} | {name} | {trio(n)} | {n['true_positive']}/{n['false_positive']} | {no['detected_onsets']}/{no['actual_onsets']} | {trio(s)} | {s['true_positive']}/{s['false_positive']} | {so['detected_onsets']}/{so['actual_onsets']} |")
    lines += ["", "## Monthly common-origin MAE", "",
              "| Horizon min | Target local month | XGBoost | Persistence | Previous day | Train median | Hour-of-week median |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for h in HORIZONS:
        monthly = result["horizons"][str(h)]["splits"]["evaluate"]["common"]["monthly"]
        for month, models in monthly.items():
            lines.append(f"| {h} | {month} | " + " | ".join(_number(models[name]["mae"], 5) for name in MODEL_NAMES) + " |")
    lines += ["", "## Error conditional on realised price", "",
              "Conditional subsets differ in size across horizons because target times shift. The full JSON also records these counts and the same calculation for all five approaches.", "",
              "| Horizon min | Actual price range | Count | XGBoost MAE | Persistence MAE |", "| ---: | --- | ---: | ---: | ---: |"]
    for h in HORIZONS:
        ranges = result["horizons"][str(h)]["splits"]["evaluate"]["common"]["conditional_price_ranges"]
        for label, models in ranges.items():
            if models:
                lines.append(f"| {h} | {label} | {models['xgboost']['count']} | {_number(models['xgboost']['mae'], 5)} | {_number(models['persistence']['mae'], 5)} |")
    lines += ["", "Models/predictions: `models/price_multi_horizon_v1/`. Figures: `../plots/forecasting/multi_horizon_v1/`. Original +60-minute V2-A verification and the common-row score are in `results.json`.", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    print(make_report())
