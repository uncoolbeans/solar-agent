"""Matched Matplotlib diagnostics for the offline tabular comparison."""

import csv
import gzip
import json
from pathlib import Path

from solaralpha.historical_pipeline import ROOT
from .tabular_comparison import ALL_NAMES, NAMES, OUTPUT_DIR


PLOT_DIR = ROOT / "plots" / "forecasting" / "tabular_comparison"
FOCUS = ("persistence", "V1", "V2-A", "ridge", "lasso",
         "random_forest", "lightgbm", "catboost")
DAY_FOCUS = ("actual", "persistence", "V2-A", "random_forest", "lightgbm", "catboost")


def _rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def make_plots(output_dir=PLOT_DIR):
    import os
    cache = ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    metrics = results["splits"]["evaluate"]
    rows = _rows(OUTPUT_DIR / "evaluate_comparison.csv.gz")
    actual = np.array([float(row["actual_aud_per_kwh"]) for row in rows])
    predictions = {name: np.array([float(row[f"{name}_aud_per_kwh"]) for row in rows])
                   for name in ALL_NAMES}
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name):
        plt.tight_layout()
        plt.savefig(output_dir / name, dpi=150)
        plt.close()
        files.append((output_dir / name).relative_to(ROOT).as_posix())

    order = sorted(ALL_NAMES, key=lambda name: metrics["models"][name]["mae"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, metric in zip(axes, ("mae", "rmse")):
        ax.barh(order, [metrics["models"][n][metric] for n in order])
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.upper()} (AUD/kWh)")
        ax.grid(axis="x", alpha=.25)
    fig.suptitle("Matched July–November 2025 reused development evaluation")
    save("leaderboard_mae_rmse.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, event in zip(axes, ("negative_price", "spike")):
        ax.barh(order, [metrics["models"][n][event]["true_positive"] for n in order],
                label="true positive")
        ax.barh(order, [metrics["models"][n][event]["false_positive"] for n in order],
                left=[metrics["models"][n][event]["true_positive"] for n in order],
                label="false positive")
        ax.invert_yaxis()
        ax.set_title(event.replace("_", " "))
        ax.set_xlabel("Forecast interval count")
    axes[1].legend()
    save("event_detection.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    sample = np.arange(0, len(actual), max(1, len(actual) // 4000))
    for name in ("V2-A", "lightgbm", "catboost"):
        axes[0, 0].scatter(actual[sample], predictions[name][sample], s=2, alpha=.16,
                           label=name)
    axes[0, 0].plot([-0.2, 1.5], [-0.2, 1.5], color="black", lw=1)
    axes[0, 0].set(xlim=(-.2, 1.5), ylim=(-.2, 1.5),
                   xlabel="Actual AUD/kWh", ylabel="Predicted AUD/kWh",
                   title="Actual vs predicted (axes clip extreme prices)")
    axes[0, 0].legend()
    bins = np.linspace(-.2, .2, 90)
    for name in ("V2-A", "lightgbm", "catboost"):
        axes[0, 1].hist(np.clip(predictions[name] - actual, -.2, .2), bins=bins,
                        histtype="step", density=True, label=name)
    axes[0, 1].set(title="Residuals: prediction − actual (clipped)", xlabel="AUD/kWh")
    axes[0, 1].legend()
    bins = np.linspace(-.2, .4, 100)
    axes[1, 0].hist(np.clip(actual, -.2, .4), bins=bins, histtype="step",
                    density=True, color="black", label="actual")
    for name in ("V2-A", "lightgbm", "catboost"):
        axes[1, 0].hist(np.clip(predictions[name], -.2, .4), bins=bins,
                        histtype="step", density=True, label=name)
    axes[1, 0].set(title="Price distributions (clipped)", xlabel="AUD/kWh")
    axes[1, 0].legend()
    for name in FOCUS:
        delta = predictions[name] - actual
        axes[1, 1].plot([np.percentile(delta, p) for p in (5, 25, 50, 75, 95)],
                        label=name)
    axes[1, 1].set(title="Residual percentiles", xticks=range(5),
                   xticklabels=["5", "25", "50", "75", "95"], xlabel="Percentile",
                   ylabel="AUD/kWh")
    axes[1, 1].legend(fontsize=7)
    save("predictions_and_residuals.png")

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)
    for ax, key, xlabel in ((axes[0], "monthly", "Target local month"),
                           (axes[1], "hour_of_day", "Target local hour")):
        groups = metrics[key]
        for name in FOCUS:
            ax.plot(list(groups), [group[name]["mae"] for group in groups.values()],
                    label=name, marker=".", lw=1)
        ax.set(xlabel=xlabel, ylabel="MAE (AUD/kWh)")
        ax.grid(alpha=.25)
    axes[0].legend(ncol=4, fontsize=8)
    save("monthly_hourly_mae.png")

    days = json.loads((ROOT / "plots/forecasting/xgb_v2/selection.json").read_text(
        encoding="utf-8"))["selected_dates"]
    for kind, dates in days.items():
        for day in dates:
            chosen = [i for i, row in enumerate(rows) if row["origin_local_date"] == day]
            if not chosen:
                raise ValueError(f"Representative day absent: {day}")
            fig, ax = plt.subplots(figsize=(13, 4))
            x = np.arange(len(chosen)) / 12
            ax.plot(x, actual[chosen], color="black", label="actual", lw=1.5)
            for name in DAY_FOCUS[1:]:
                ax.plot(x, predictions[name][chosen], lw=1, alpha=.8, label=name)
            ax.set(xlabel="Hours from first decision on local date",
                   ylabel="One-hour-ahead AUD/kWh", title=f"{kind}: {day}")
            ax.legend(ncol=3, fontsize=8)
            save(f"day_{kind}_{day}.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    for name in NAMES:
        timing = results["models"][name]["inference"]["evaluate"]
        ax.scatter(timing["single_observation_median_microseconds"],
                   metrics["models"][name]["mae"], label=name, s=55)
    ax.set(xlabel="Median one-row inference (microseconds)",
           ylabel="MAE (AUD/kWh)", title="Accuracy vs model-only single-row latency")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=.25)
    save("accuracy_vs_inference.png")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for ax, name in zip(axes.flat, NAMES):
        importance = results["models"][name]["feature_importance"]
        top = sorted(importance, key=lambda feature: abs(importance[feature]),
                     reverse=True)[:10][::-1]
        ax.barh(top, [importance[k] for k in top])
        ax.set_title(f"{name}: " + ("standardised coefficient" if name in
                                      ("ridge", "lasso") else "native importance"))
        ax.tick_params(axis="y", labelsize=7)
    axes[-1, -1].axis("off")
    save("feature_importance.png")

    manifest = {"evaluation_status": "Reused development evaluation, not independent test",
                "matched_row_count": len(rows), "model_names": list(ALL_NAMES),
                "days_selected_from_realised_prices_only": days,
                "notes": "Clipped axes are labelled; raw metrics include all extreme prices. Native importance measures differ by model and are not causal.",
                "files": files}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(make_plots(), indent=2))
