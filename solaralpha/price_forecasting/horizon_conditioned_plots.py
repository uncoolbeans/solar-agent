"""Plots for the ten trained sparse horizons; no intermediate prediction claims."""

import csv
import gzip
import json
import os

from solaralpha.historical_pipeline import ROOT
from .horizon_conditioned import OUTPUT_DIR, PLOT_DIR
from .multi_horizon import HORIZONS, TARGET_DIR


def make_plots(output_dir=PLOT_DIR):
    cache = ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(name):
        plt.tight_layout()
        path = output_dir / name
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(path.relative_to(ROOT).as_posix())

    def score(h, name):
        return result["horizons"][str(h)]["splits"]["evaluate"]["common"]["models"][name]

    models = ("persistence", "direct", "shared", "hybrid")
    for metric, label in (("mae", "MAE"), ("rmse", "RMSE"), ("mean_bias", "Prediction bias")):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for name in models:
            ax.plot(HORIZONS, [score(h, name)[metric] for h in HORIZONS],
                    marker="o", label=name)
        ax.set(title=f"{label} at ten trained horizons, common origins",
               xlabel="Target interval start after decision (minutes)", ylabel="AUD/kWh")
        ax.grid(alpha=.25); ax.legend()
        save(f"{metric}_vs_horizon.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for name in ("direct", "shared", "hybrid"):
        ax.plot(HORIZONS, [100 * (1 - score(h, name)["mae"] / score(h, "persistence")["mae"])
                           for h in HORIZONS], marker="o", label=name)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="MAE advantage over matched persistence",
           xlabel="Target interval start after decision (minutes)", ylabel="Relative improvement (%)")
    ax.grid(alpha=.25); ax.legend()
    save("relative_mae_improvement.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for name in models:
        ax.plot(HORIZONS, [score(h, name)["negative_price"]["f1"] for h in HORIZONS],
                marker="o", label=name)
    ax.set(title="Negative-price F1 at ten trained horizons",
           xlabel="Target interval start after decision (minutes)", ylabel="F1")
    ax.grid(alpha=.25); ax.legend()
    save("negative_f1.png")

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for name in ("direct", "shared", "hybrid"):
        ax.plot(HORIZONS, [score(h, name)["spike"]["true_positive"] for h in HORIZONS],
                marker="o", label=f"{name} true positive")
        ax.plot(HORIZONS, [score(h, name)["spike"]["false_positive"] for h in HORIZONS],
                marker="x", linestyle="--", label=f"{name} false positive")
    ax.set(title="At-least-$1/kWh spike intervals, common origins",
           xlabel="Target interval start after decision (minutes)", ylabel="Count")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save("spike_detection.png")

    groups = ("price_history", "regional_demand", "calendar", "horizon")
    importance = result["importance"]
    labels = [f"+{h}m direct" for h in HORIZONS] + ["shared"]
    records = [importance["direct"][str(h)]["fraction_by_group"] for h in HORIZONS]
    records.append(importance["shared"]["fraction_by_group"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(labels))
    for group in groups:
        values = np.asarray([record[group] for record in records]) * 100
        ax.bar(labels, values, bottom=bottom, label=group.replace("_", " "))
        bottom += values
    ax.set(title="Feature-group share of total split gain (descriptive)", ylabel="Share of total gain (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    save("feature_group_importance.png")

    with gzip.open(TARGET_DIR / "validation_targets.csv.gz", "rt", encoding="utf-8", newline="") as f:
        common = {row["decision_start_utc"] for row in csv.DictReader(f)
                  if row["period"] == "evaluate" and row["common_eligible"] == "1"}
    with gzip.open(OUTPUT_DIR / "h060_evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["decision_start_utc"] in common]
    ordinary = next(row for row in rows if .05 <= float(row["actual_aud_per_kwh"]) <= .15)
    selected = {"ordinary": ordinary["decision_start_utc"],
                "negative": min(rows, key=lambda row: float(row["actual_aud_per_kwh"]))["decision_start_utc"],
                "spike": max(rows, key=lambda row: float(row["actual_aud_per_kwh"]))["decision_start_utc"]}
    samples = {label: {} for label in selected}
    for h in HORIZONS:
        with gzip.open(OUTPUT_DIR / f"h{h:03d}_evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for label, origin in selected.items():
                    if row["decision_start_utc"] == origin:
                        samples[label][h] = row
    for label, horizon_rows in samples.items():
        if len(horizon_rows) != len(HORIZONS):
            raise ValueError("Representative origin lacks a trained horizon")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, symbol in (("actual", "o"), ("direct", "x"),
                             ("shared", "^"), ("persistence", "+")):
            field = "actual_aud_per_kwh" if name == "actual" else f"{name}_aud_per_kwh"
            ax.scatter(HORIZONS, [float(horizon_rows[h][field]) for h in HORIZONS],
                       marker=symbol, s=55, label=name)
        ax.set(title=f"{label.capitalize()} example issued at {selected[label]}\n"
                     "Ten sparse forecasts only; intervening intervals are untrained",
               xlabel="Target interval start after decision (minutes)", ylabel="AUD/kWh")
        if label == "spike":
            ax.set_yscale("symlog", linthresh=0.25)
            ax.set_ylabel("AUD/kWh (symmetric log scale)")
        ax.grid(alpha=.25); ax.legend()
        save(f"same_decision_{label}.png")
    return paths


if __name__ == "__main__":
    print("\n".join(make_plots()))
