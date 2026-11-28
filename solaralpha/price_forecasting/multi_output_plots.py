"""Matplotlib diagnostics for ten sparse multi-output neural forecasts."""

import csv
import gzip
import json
import os

from solaralpha.historical_pipeline import ROOT
from .multi_horizon import HORIZONS
from .multi_output_experiment import OUTPUT_DIR


PLOT_DIR = ROOT / "plots" / "forecasting" / "multi_output_neural_v1"
FOCUS = ("persistence", "direct", "shared", "multi_lstm", "multi_gru", "multi_tcn")


def make_plots(output_dir=PLOT_DIR):
    cache = ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    r = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name):
        plt.tight_layout()
        path = output_dir / name
        plt.savefig(path, dpi=150)
        plt.close()
        files.append(path.relative_to(ROOT).as_posix())

    def score(h, name):
        return r["splits"]["evaluate"]["horizons"][str(h)]["models"][name]

    for metric, title in (("mae", "MAE"), ("rmse", "RMSE"), ("mean_bias", "Prediction bias")):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name in FOCUS:
            ax.plot(HORIZONS, [score(h, name)[metric] for h in HORIZONS],
                    marker="o", label=name.replace("multi_", ""))
        ax.set(title=f"{title} on matched common origins",
               xlabel="Target interval start after decision (minutes)", ylabel="AUD/kWh")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(f"{metric}_vs_horizon.png")

    for reference in ("persistence", "direct"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name in ("multi_lstm", "multi_gru", "multi_tcn"):
            ax.plot(HORIZONS, [100 * (1 - score(h, name)["mae"] / score(h, reference)["mae"])
                               for h in HORIZONS], marker="o", label=name.replace("multi_", ""))
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(title=f"MAE improvement over {reference}",
               xlabel="Target interval start after decision (minutes)", ylabel="Relative MAE improvement (%)")
        ax.grid(alpha=.25); ax.legend()
        save(f"relative_mae_vs_{reference}.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in FOCUS:
        ax.plot(HORIZONS, [score(h, name)["negative_price"]["f1"] for h in HORIZONS],
                marker="o", label=name.replace("multi_", ""))
    ax.set(title="Negative-price F1 on matched common origins",
           xlabel="Target interval start after decision (minutes)", ylabel="F1")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save("negative_f1.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in ("direct", "multi_lstm", "multi_gru", "multi_tcn"):
        ax.plot(HORIZONS, [score(h, name)["spike"]["true_positive"] for h in HORIZONS],
                marker="o", label=f"{name} true positive")
        ax.plot(HORIZONS, [score(h, name)["spike"]["false_positive"] for h in HORIZONS],
                marker="x", linestyle="--", label=f"{name} false positive")
    ax.set(title="At-least-$1/kWh spike intervals on matched common origins",
           xlabel="Target interval start after decision (minutes)", ylabel="Count")
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    save("spike_detection.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, name in zip(axes, ("lstm", "gru", "tcn")):
        meta = r["models"][name]
        curve = meta["learning_curves"]
        epochs = np.arange(1, len(curve["train_mae"]) + 1)
        ax.plot(epochs, curve["train_mae"], marker="o", label="train")
        ax.plot(epochs, curve["tune_mae"], marker="o", label="tune")
        ax.axvline(meta["best_epoch_one_based"], color="black", linestyle="--", label="selected")
        ax.set(title=name.upper(), xlabel="Epoch", ylabel="Equal-horizon MAE (AUD/kWh)")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    save("learning_curves.png")

    with gzip.open(OUTPUT_DIR / "h060_evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ordinary = next(row for row in rows if .05 <= float(row["actual_aud_per_kwh"]) <= .15)
    selected = {"ordinary": ordinary["decision_start_utc"],
                "negative": min(rows, key=lambda row: float(row["actual_aud_per_kwh"]))["decision_start_utc"],
                "spike": max(rows, key=lambda row: float(row["actual_aud_per_kwh"]))["decision_start_utc"]}
    samples = {name: {} for name in selected}
    for h in HORIZONS:
        with gzip.open(OUTPUT_DIR / f"h{h:03d}_evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for label, origin in selected.items():
                    if row["decision_start_utc"] == origin:
                        samples[label][h] = row
    for label, values in samples.items():
        if len(values) != len(HORIZONS):
            raise ValueError("Example is missing a trained horizon")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, marker in (("actual", "o"), ("direct", "x"),
                             ("multi_lstm", "^"), ("multi_gru", "s"), ("multi_tcn", "+")):
            field = "actual_aud_per_kwh" if name == "actual" else f"{name}_aud_per_kwh"
            ax.scatter(HORIZONS, [float(values[h][field]) for h in HORIZONS],
                       marker=marker, label=name.replace("multi_", ""), s=45)
        ax.set(title=f"{label.capitalize()} example issued at {selected[label]}\n"
                     "Ten sparse predictions; no validated intervening intervals",
               xlabel="Target interval start after decision (minutes)", ylabel="AUD/kWh")
        if label == "spike":
            ax.set_yscale("symlog", linthresh=.25)
            ax.set_ylabel("AUD/kWh (symmetric log scale)")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(f"same_decision_{label}.png")
    return files


if __name__ == "__main__":
    print("\n".join(make_plots()))
