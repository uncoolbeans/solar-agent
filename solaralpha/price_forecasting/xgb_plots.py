"""Matplotlib diagnostics for the single XGBoost V1 evaluation."""

import csv
from datetime import datetime
import gzip
import json
import os

from solaralpha.historical_pipeline import ROOT, sydney_zone
from .xgb_v1 import MODEL_DIR


PLOT_DIR = ROOT / "plots" / "forecasting" / "xgb_v1"
SERIES = {
    "xgboost_aud_per_kwh": ("XGBoost V1", "#0072B2"),
    "persistence_aud_per_kwh": ("Persistence", "#D55E00"),
    "daily_seasonal_aud_per_kwh": ("Previous day", "#CC79A7"),
    "train_median_aud_per_kwh": ("Training median", "#009E73"),
    "hour_of_week_median_aud_per_kwh": ("Hour-of-week median", "#56B4E9"),
}


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _target_local(row):
    return datetime.fromisoformat(row["target_interval_start_utc"].replace("Z", "+00:00"))\
        .astimezone(sydney_zone())


def _day_candidates(rows):
    by_day = {}
    for row in rows:
        by_day.setdefault(_target_local(row).date().isoformat(), []).append(row)
    # Select using realised outcomes only, never model errors.
    full = {day: values for day, values in by_day.items() if len(values) >= 275}
    normal = sorted((day for day, values in full.items()
                     if min(float(r["actual_aud_per_kwh"]) for r in values) >= 0
                     and max(float(r["actual_aud_per_kwh"]) for r in values) < 0.25))[:2]
    negative = sorted(full, key=lambda day: (
        -sum(float(r["actual_aud_per_kwh"]) < 0 for r in full[day]), day))
    negative = [day for day in negative
                if any(float(r["actual_aud_per_kwh"]) < 0 for r in full[day])][:2]
    spike = sorted(full, key=lambda day: (
        -max(float(r["actual_aud_per_kwh"]) for r in full[day]), day))
    spike = [day for day in spike
             if max(float(r["actual_aud_per_kwh"]) for r in full[day]) >= 1][:2]
    if len(normal) < 2 or len(negative) < 2 or len(spike) < 2:
        raise ValueError("Not enough representative evaluation days")
    return full, {"normal": normal, "negative": negative, "spike": spike}


def make_plots(model_dir=MODEL_DIR, output_dir=PLOT_DIR):
    cache_dir = ROOT / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads((model_dir / "results.json").read_text(encoding="utf-8"))
    rows = _rows(model_dir / "evaluate_predictions.csv.gz")
    actual = np.asarray([float(r["actual_aud_per_kwh"]) for r in rows])
    paths = []

    curve = report["learning_curve"]
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for key, label in (("validation_0", "Training"), ("validation_1", "Jan-Jun tuning")):
        ax.plot(curve[key]["mae"], label=label)
    ax.axvline(report["best_iteration_zero_based"], color="black", ls="--",
               label="Best iteration")
    ax.set(title="XGBoost V1 early stopping", xlabel="Boosting iteration (zero-based)",
           ylabel="MAE (AUD/kWh)")
    ax.legend()
    ax.grid(alpha=0.2)
    path = output_dir / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    sample = np.arange(0, len(rows), max(1, len(rows) // 15000))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, name in zip(axes, ("xgboost_aud_per_kwh",
                               "persistence_aud_per_kwh",
                               "hour_of_week_median_aud_per_kwh")):
        prediction = np.asarray([float(r[name]) for r in rows])
        label, color = SERIES[name]
        ax.scatter(actual[sample], prediction[sample], s=2, alpha=0.22,
                   color=color, rasterized=True)
        ax.plot([-25, 25], [-25, 25], color="black", lw=0.8)
        ax.set(xscale="symlog", yscale="symlog", xlim=(-2, 25), ylim=(-2, 25),
               title=label, xlabel="Actual (AUD/kWh)",
               ylabel="Forecast (AUD/kWh)")
        ax.grid(alpha=0.2)
    path = output_dir / "prediction_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name in ("xgboost_aud_per_kwh", "persistence_aud_per_kwh",
                 "hour_of_week_median_aud_per_kwh"):
        errors = np.asarray([float(r[name]) for r in rows]) - actual
        central = errors[np.abs(errors) <= 1]
        label, color = SERIES[name]
        ax.hist(central, bins=100, range=(-1, 1), histtype="step", density=True,
                lw=1.2, color=color,
                label=f"{label} ({len(errors)-len(central):,} outside ±1)")
    ax.set(title="Evaluation residuals (central ±$1/kWh window)",
           xlabel="Forecast minus actual (AUD/kWh)", ylabel="Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    path = output_dir / "residual_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    gain = report["gain_feature_importance"]
    top = sorted(gain, key=gain.get, reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.barh(top[::-1], [gain[name] for name in top[::-1]], color="#0072B2")
    ax.set(title="XGBoost split gain (association, not causal importance)",
           xlabel="Average training split gain")
    path = output_dir / "gain_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    by_day, chosen = _day_candidates(rows)
    for kind, days in chosen.items():
        for day in days:
            daily = by_day[day]
            times = [_target_local(r) for r in daily]
            fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
            ax.plot(times, [float(r["actual_aud_per_kwh"]) for r in daily],
                    color="black", label="Actual", lw=1.4)
            for name in ("xgboost_aud_per_kwh", "persistence_aud_per_kwh",
                         "hour_of_week_median_aud_per_kwh"):
                label, color = SERIES[name]
                ax.plot(times, [float(r[name]) for r in daily],
                        color=color, label=label, lw=1, alpha=0.85)
            if kind == "spike":
                ax.set_yscale("symlog", linthresh=0.5)
            ax.set(title=f"{kind.title()} evaluation day: {day}",
                   xlabel="Target interval start (Sydney local time)",
                   ylabel="Wholesale price (AUD/kWh)")
            ax.legend(ncol=4, fontsize=8)
            ax.grid(alpha=0.2)
            fig.autofmt_xdate()
            path = output_dir / f"day_{kind}_{day}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    selection = {"selection_rule": "Realised evaluation prices only; no forecast-error selection",
                 "normal": "Earliest two nearly complete days with all prices in [0, 0.25) AUD/kWh",
                 "negative": "Two nearly complete days with most negative-price intervals",
                 "spike": "Two nearly complete days with highest positive price",
                 "selected_dates": chosen,
                 "files": [path.relative_to(ROOT).as_posix() for path in paths]}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n",
                                                encoding="utf-8")
    return selection


if __name__ == "__main__":
    print(json.dumps(make_plots(), indent=2))
