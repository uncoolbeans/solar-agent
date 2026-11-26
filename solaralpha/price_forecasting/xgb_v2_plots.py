"""Comparative Matplotlib figures for fixed XGBoost V2 variants."""

import csv
from datetime import datetime
import gzip
import json
import os

from solaralpha.historical_pipeline import ROOT, sydney_zone
from .xgb_v2 import V2_DIR, VARIANTS


PLOT_DIR = ROOT / "plots" / "forecasting" / "xgb_v2"
COLOR = {"V1": "#333333", "V2-A": "#0072B2", "V2-B": "#D55E00",
         "V2-C": "#009E73", "V2-D": "#CC79A7",
         "persistence": "#E69F00", "actual": "#000000"}


def _read(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _local_target(row):
    return datetime.fromisoformat(row["target_interval_start_utc"].replace("Z", "+00:00"))\
        .astimezone(sydney_zone())


def make_plots(model_dir=V2_DIR, output_dir=PLOT_DIR):
    cache = ROOT / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads((model_dir / "v2_results.json").read_text(encoding="utf-8"))
    rows = _read(model_dir / "evaluate_comparison.csv.gz")
    actual = np.asarray([float(r["actual_aud_per_kwh"]) for r in rows])
    names = ("V1", *VARIANTS, "persistence")
    values = {name: np.asarray([float(r[f"{name}_aud_per_kwh"]) for r in rows])
              for name in names}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for ax, (name, variant) in zip(axes.flat, result["variants"].items()):
        curve = variant["learning_curve"]
        ax.plot(curve["validation_0"]["mae"], label="Train", lw=1)
        ax.plot(curve["validation_1"]["mae"], label="Jan-Jun tune", lw=1)
        ax.axvline(variant["best_iteration_zero_based"], ls="--", color="black",
                   lw=0.8, label="Selected")
        ax.set(title=f"{name}: {variant['objective']}", xlabel="Boosting iteration",
               ylabel="Absolute-price MAE (AUD/kWh)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    path = output_dir / "learning_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for name, series in (("actual", actual), *values.items()):
        central = series[(series >= -0.20) & (series <= 0.40)]
        axes[0].hist(central, bins=100, range=(-0.20, 0.40), histtype="step",
                     density=True, lw=1, color=COLOR[name], label=name)
    axes[0].set(title="Reused evaluation: central price distribution",
                xlabel="Price (AUD/kWh)", ylabel="Density")
    axes[0].legend(ncol=2, fontsize=7)
    axes[0].grid(alpha=0.2)
    labels = ("actual", *names)
    counts = [int(np.sum((actual if n == "actual" else values[n]) >= 1))
              for n in labels]
    axes[1].bar(labels, counts, color=[COLOR[n] for n in labels])
    axes[1].set(title="Predictions at or above $1/kWh", ylabel="Intervals")
    axes[1].tick_params(axis="x", labelrotation=35)
    path = output_dir / "prediction_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name in names:
        error = values[name] - actual
        central = error[np.abs(error) <= 0.5]
        ax.hist(central, bins=100, range=(-0.5, 0.5), histtype="step",
                density=True, color=COLOR[name], lw=1,
                label=f"{name} ({len(error)-len(central):,} outside +/-0.5)")
    ax.set(title="Reused evaluation residuals", xlabel="Forecast minus actual (AUD/kWh)",
           ylabel="Density")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(alpha=0.2)
    path = output_dir / "residual_distributions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # Reuse the earlier V1 day's outcome-only choices for direct comparison.
    prior = json.loads((ROOT / "plots/forecasting/xgb_v1/selection.json")
                       .read_text(encoding="utf-8"))
    selected = prior["selected_dates"]
    for kind in ("normal", "negative", "spike"):
        for day in selected[kind]:
            daily = [r for r in rows if _local_target(r).date().isoformat() == day]
            if not daily:
                raise ValueError(f"No evaluation targets on {day}")
            times = [_local_target(r) for r in daily]
            fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
            ax.plot(times, [float(r["actual_aud_per_kwh"]) for r in daily],
                    color=COLOR["actual"], lw=1.4, label="Actual")
            for name in names:
                ax.plot(times, [float(r[f"{name}_aud_per_kwh"]) for r in daily],
                        color=COLOR[name], lw=0.9, alpha=0.8, label=name)
            if kind == "spike":
                ax.set_yscale("symlog", linthresh=0.5)
            ax.set(title=f"{kind.title()} day {day} (reused development evaluation)",
                   xlabel="Target interval start (Sydney local)",
                   ylabel="Wholesale price (AUD/kWh)")
            ax.legend(ncol=4, fontsize=8)
            ax.grid(alpha=0.2)
            fig.autofmt_xdate()
            path = output_dir / f"day_{kind}_{day}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(path)
    selection = {"evaluation_status": "Reused development, not independent test",
                 "day_selection": "Same outcome-only dates as XGBoost V1",
                 "selected_dates": selected,
                 "files": [p.relative_to(ROOT).as_posix() for p in paths]}
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n",
                                                encoding="utf-8")
    return selection


if __name__ == "__main__":
    print(json.dumps(make_plots(), indent=2))
