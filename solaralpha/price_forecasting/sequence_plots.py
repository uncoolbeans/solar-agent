"""Matplotlib diagnostics for the matched 17-model sequence comparison."""

import csv
import gzip
import json
import os

from solaralpha.historical_pipeline import ROOT
from .sequence_experiment import ALL_NAMES, NEURAL_NAMES, OUTPUT_DIR


PLOT_DIR = ROOT / "plots" / "forecasting" / "sequence_comparison"
FOCUS = ("actual", "persistence", "V2-A", "lightgbm", "catboost",
         "lstm", "gru", "tcn")


def make_plots(output_dir=PLOT_DIR):
    cache = ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    with gzip.open(OUTPUT_DIR / "evaluate_comparison.csv.gz", "rt", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    actual = np.asarray([float(row["actual_aud_per_kwh"]) for row in rows])
    pred = {name: np.asarray([float(row[f"{name}_aud_per_kwh"]) for row in rows])
            for name in ALL_NAMES}
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name):
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150)
        plt.close()
        files.append((output_dir / name).relative_to(ROOT).as_posix())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, name in zip(axes, NEURAL_NAMES):
        info = result["models"][name]
        curve = info["learning_curves"]
        epoch = np.arange(1, len(curve["train_mae"]) + 1)
        ax.plot(epoch, curve["train_mae"], marker=".", label="train MAE")
        ax.plot(epoch, curve["tune_mae"], marker=".", label="tune MAE")
        ax.axvline(info["best_epoch_one_based"], color="black", linestyle="--",
                   label="selected epoch")
        ax.set(title=name.upper(), xlabel="Epoch", ylabel="MAE (AUD/kWh)")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    save("learning_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(-.2, .4, 100)
    axes[0].hist(np.clip(actual, -.2, .4), bins=bins, histtype="step",
                 density=True, color="black", label="actual")
    for name in FOCUS[1:]:
        axes[0].hist(np.clip(pred[name], -.2, .4), bins=bins, histtype="step",
                     density=True, label=name)
    axes[0].set(title="Prediction distributions (display clipped)",
                xlabel="AUD/kWh", ylabel="Density")
    axes[0].legend(fontsize=7)
    bins = np.linspace(-.2, .2, 100)
    for name in FOCUS[1:]:
        axes[1].hist(np.clip(pred[name] - actual, -.2, .2), bins=bins,
                     histtype="step", density=True, label=name)
    axes[1].set(title="Residuals: predicted − actual (display clipped)",
                xlabel="AUD/kWh", ylabel="Density")
    axes[1].legend(fontsize=7)
    save("predictions_and_residuals.png")

    metrics = result["splits"]["evaluate"]["models"]
    order = sorted(ALL_NAMES, key=lambda name: metrics[name]["mae"])
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(order, [metrics[name]["mae"] for name in order])
    ax.invert_yaxis()
    ax.set(xlabel="MAE (AUD/kWh)", title="Matched reused development evaluation")
    ax.grid(axis="x", alpha=.25)
    save("seventeen_model_mae.png")

    days = json.loads((ROOT / "plots/forecasting/xgb_v2/selection.json").read_text(
        encoding="utf-8"))["selected_dates"]
    for kind, dates in days.items():
        for day in dates:
            chosen = [i for i, row in enumerate(rows) if row["origin_local_date"] == day]
            if not chosen:
                raise ValueError(f"Representative evaluation day missing: {day}")
            fig, ax = plt.subplots(figsize=(13, 4))
            hour = np.arange(len(chosen)) / 12
            ax.plot(hour, actual[chosen], color="black", linewidth=1.5, label="actual")
            for name in FOCUS[1:]:
                ax.plot(hour, pred[name][chosen], linewidth=1, alpha=.8, label=name)
            ax.set(title=f"{kind}: {day}", xlabel="Hours from first decision on local date",
                   ylabel="One-hour-ahead price (AUD/kWh)")
            ax.legend(ncol=4, fontsize=7)
            save(f"day_{kind}_{day}.png")

    manifest = {"evaluation_status": "Reused July-November 2025 development evaluation",
                "matched_rows": len(rows), "model_names": list(ALL_NAMES),
                "representative_days_selected_using_actual_prices_only": days,
                "display_clipping": "Distribution figures clip plotted values only; metrics retain extremes",
                "files": files}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(make_plots(), indent=2))
