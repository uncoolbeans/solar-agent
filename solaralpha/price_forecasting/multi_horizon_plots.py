"""Matplotlib summaries of the ten sparse, separately trained price horizons."""

import csv
import gzip
import json
import os

from solaralpha.historical_pipeline import ROOT
from .multi_horizon import HORIZONS, MODEL_DIR, TARGET_DIR


PLOT_DIR = ROOT / "plots" / "forecasting" / "multi_horizon_v1"


def make_plots(output_dir=PLOT_DIR):
    cache = ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads((MODEL_DIR / "results.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name):
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150)
        plt.close()
        files.append((output_dir / name).relative_to(ROOT).as_posix())

    def score(h, model):
        return result["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"][model]

    for metric, title, ylabel in (
        ("mae", "MAE on common origins", "AUD/kWh"),
        ("rmse", "RMSE on common origins", "AUD/kWh"),
        ("mean_bias", "Prediction bias on common origins", "AUD/kWh"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model in ("xgboost", "persistence", "daily_seasonal", "train_median", "hour_of_week_median"):
            ax.plot(HORIZONS, [score(h, model)[metric] for h in HORIZONS],
                    marker="o", label=model.replace("_", " "))
        ax.set(title=title, xlabel="Target interval start after decision (minutes)", ylabel=ylabel)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(f"{metric}_vs_horizon.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    improvement = [100 * (1 - score(h, "xgboost")["mae"] / score(h, "persistence")["mae"])
                   for h in HORIZONS]
    ax.plot(HORIZONS, improvement, marker="o")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="XGBoost MAE improvement over matched persistence",
           xlabel="Target interval start after decision (minutes)", ylabel="Relative MAE improvement (%)")
    ax.grid(alpha=.25)
    save("relative_mae_improvement.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in ("xgboost", "persistence", "daily_seasonal"):
        ax.plot(HORIZONS, [score(h, model)["negative_price"]["f1"] for h in HORIZONS],
                marker="o", label=model.replace("_", " "))
    ax.set(title="Negative-price F1 on common origins",
           xlabel="Target interval start after decision (minutes)", ylabel="F1")
    ax.grid(alpha=.25); ax.legend()
    save("negative_f1.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in ("xgboost", "persistence"):
        ax.plot(HORIZONS, [score(h, model)["spike"]["true_positive"] for h in HORIZONS],
                marker="o", label=f"{model} true positives")
        ax.plot(HORIZONS, [score(h, model)["spike"]["false_positive"] for h in HORIZONS],
                marker="x", linestyle="--", label=f"{model} false positives")
    ax.set(title="At-least-$1/kWh spike detection on common origins",
           xlabel="Target interval start after decision (minutes)", ylabel="Interval count")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save("spike_detection.png")

    with gzip.open(TARGET_DIR / "validation_targets.csv.gz", "rt", encoding="utf-8", newline="") as f:
        common = {row["decision_start_utc"] for row in csv.DictReader(f)
                  if row["period"] == "evaluate" and row["common_eligible"] == "1"}
    with gzip.open(MODEL_DIR / "h060" / "evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["decision_start_utc"] in common]
    ordinary = next(row for row in rows if .05 <= float(row["actual_aud_per_kwh"]) <= .15
                    and .05 <= float(row["xgboost_aud_per_kwh"]) <= .15)
    negative = min(rows, key=lambda row: float(row["actual_aud_per_kwh"]))
    spike = max(rows, key=lambda row: float(row["actual_aud_per_kwh"]))
    selected = {"ordinary": ordinary["decision_start_utc"],
                "negative": negative["decision_start_utc"],
                "spike": spike["decision_start_utc"]}
    samples = {label: {} for label in selected}
    for h in HORIZONS:
        with gzip.open(MODEL_DIR / f"h{h:03d}" / "evaluate_predictions.csv.gz",
                       "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for label, origin in selected.items():
                    if row["decision_start_utc"] == origin:
                        samples[label][h] = row
    for label, horizon_rows in samples.items():
        if len(horizon_rows) != len(HORIZONS):
            raise ValueError("Representative origin is missing a horizon")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, symbol in (("actual", "o"), ("xgboost", "x"), ("persistence", "+")):
            field = "actual_aud_per_kwh" if name == "actual" else f"{name}_aud_per_kwh"
            values = [float(horizon_rows[h][field]) for h in HORIZONS]
            ax.scatter(HORIZONS, values, marker=symbol, label=name, s=55)
        ax.set(title=f"{label.capitalize()} example: forecast issued at {selected[label]}\n"
                     "Ten sparse interval predictions; no intervening intervals predicted",
               xlabel="Target interval start after decision (minutes)", ylabel="AUD/kWh")
        if label == "spike":
            ax.set_yscale("symlog", linthresh=0.25)
            ax.set_ylabel("AUD/kWh (symmetric log scale)")
        ax.grid(alpha=.25); ax.legend()
        save(f"same_decision_{label}.png")
    return files


if __name__ == "__main__":
    print("\n".join(make_plots()))
