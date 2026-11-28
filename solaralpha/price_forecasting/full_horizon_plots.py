"""Scientific plots for the dense 12-hour development benchmark."""

import json
import os

from solaralpha.historical_pipeline import ROOT
from .full_horizon_dataset import HORIZONS
from .full_horizon_direct import OUTPUT_DIR


PLOT_DIR = ROOT / "plots/forecasting/full_horizon_v1"
FOCUS = ("persistence", "direct", "hybrid_tuned", "hybrid_fixed", "tcn")


def make_plots(output_dir=PLOT_DIR):
    cache = ROOT / ".cache/matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    result = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    def save(name):
        plt.tight_layout()
        path = output_dir / name
        plt.savefig(path, dpi=150)
        plt.close()
        files.append(path)
    def scores(h, name):
        return result["splits"]["evaluate"]["horizons"][str(h)]["models"][name]

    for metric, title in (("mae", "MAE"), ("rmse", "RMSE")):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for name in FOCUS:
            ax.plot(HORIZONS, [scores(h, name)[metric] for h in HORIZONS],
                    label=name.replace("_", " "))
        ax.set(title=f"{title} on 43,909 matched development origins",
               xlabel="Target interval starts after decision (minutes)", ylabel="AUD/kWh")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(f"{metric}_all_144.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ("direct", "hybrid_tuned", "hybrid_fixed", "tcn"):
        ax.plot(HORIZONS, [100 * (1 - scores(h, name)["mae"] / scores(h, "persistence")["mae"])
                           for h in HORIZONS], label=name.replace("_", " "))
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(title="Relative MAE improvement over persistence", xlabel="Target start after decision (minutes)",
           ylabel="Improvement (%)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save("mae_improvement_vs_persistence.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in FOCUS:
        ax.plot(HORIZONS, [scores(h, name)["negative_price"]["f1"] for h in HORIZONS],
                label=name.replace("_", " "))
    ax.set(title="Negative-price F1 on matched development origins",
           xlabel="Target start after decision (minutes)", ylabel="F1")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save("negative_price_f1.png")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name in ("direct", "tcn", "hybrid_tuned"):
        ax.plot(HORIZONS, [scores(h, name)["spike"]["true_positive"] for h in HORIZONS],
                label=f"{name} true positive")
        ax.plot(HORIZONS, [scores(h, name)["spike"]["false_positive"] for h in HORIZONS],
                linestyle="--", label=f"{name} false positive")
    ax.set(title="≥$1/kWh spike detections", xlabel="Target start after decision (minutes)", ylabel="Intervals")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    save("spike_detections.png")

    tcn = json.loads((OUTPUT_DIR / "tcn/metadata.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, tcn["epochs_fitted"] + 1)
    ax.plot(epochs, tcn["learning_curves"]["train_mae"], marker="o", label="train")
    ax.plot(epochs, tcn["learning_curves"]["tune_mean_mae"], marker="o", label="Jan–Jun tuning")
    ax.axvline(tcn["best_epoch_one_based"], linestyle="--", color="black", label="selected")
    ax.set(title="144-output TCN learning curve", xlabel="Epoch", ylabel="Equal-horizon MAE (AUD/kWh)")
    ax.grid(alpha=.25); ax.legend()
    save("tcn_learning_curve.png")

    with np.load(OUTPUT_DIR / "evaluate_comparison.npz") as source:
        actual = source["actual"]
        forecasts = {name: source[name] for name in ("direct", "tcn", "hybrid_tuned", "persistence")}
        origin_indices = source["origin_point_indices"]
    from .features import load_training_validation_points
    points = load_training_validation_points()
    midpoint = HORIZONS.index(60)
    ordinary = next(i for i, row in enumerate(actual) if .05 <= row[midpoint] <= .15 and
                    np.max(row) < .30 and np.min(row) >= 0)
    examples = {"ordinary": ordinary,
                "negative": int(np.argmin(actual[:, midpoint])),
                "spike": int(np.argmax(actual[:, midpoint]))}
    for label, index in examples.items():
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(HORIZONS, actual[index], linewidth=1.4, label="realised")
        for name, values in forecasts.items():
            ax.plot(HORIZONS, values[index], linewidth=1.0, label=name.replace("_", " "))
        if label == "spike":
            ax.set_yscale("symlog", linthresh=.25)
        decision = points[int(origin_indices[index])].local_time.isoformat()
        ax.set(title=f"{label.capitalize()} development example issued {decision}\n"
                     "Selected by realised +60-minute price; illustrative, not representative",
               xlabel="Target interval starts after decision (minutes)", ylabel="AUD/kWh")
        ax.grid(alpha=.25); ax.legend(fontsize=8)
        save(f"trajectory_{label}.png")

    pairs = {}
    for i, index in enumerate(origin_indices):
        local = points[int(index)].local_time
        if local.minute == 0 and local.hour in (0, 12):
            pairs.setdefault(local.date().isoformat(), {})[local.hour] = i
    dates = sorted(date for date, pair in pairs.items() if 0 in pair and 12 in pair and
                   (points[int(origin_indices[pair[12]])].time_utc -
                    points[int(origin_indices[pair[0]])].time_utc).total_seconds() == 43200)
    actual_ranges = np.asarray([np.ptp(np.concatenate((actual[pairs[d][0]], actual[pairs[d][12]])))
                                for d in dates])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, name in zip(axes, ("direct", "tcn")):
        predicted = forecasts[name]
        predicted_ranges = np.asarray([np.ptp(np.concatenate((predicted[pairs[d][0]],
                                                               predicted[pairs[d][12]]))) for d in dates])
        ax.scatter(actual_ranges, predicted_ranges, s=12, alpha=.5)
        bound = max(float(actual_ranges.max()), float(predicted_ranges.max()))
        ax.plot([0, bound], [0, bound], color="black", linewidth=.8, linestyle="--")
        ax.set(xscale="symlog", yscale="symlog", title=f"{name}: {len(dates)} paired local days",
               xlabel="Realised daily price range (AUD/kWh)",
               ylabel="Predicted daily price range (AUD/kWh)")
        ax.grid(alpha=.25)
    save("paired_daily_price_ranges.png")

    resources = result["resources"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    labels = ("direct", "hybrid_tuned", "hybrid_fixed", "tcn")
    latencies = resources["single_decision_latency"]
    axes[0].bar(labels, [latencies[n]["median_ms"] for n in labels])
    axes[0].set(title="Prepared 144-point inference", ylabel="Median ms")
    axes[1].bar(("direct sum", "TCN"),
                [resources["direct_models"]["sum_fit_seconds"] / 60,
                 resources["tcn"]["fit_seconds"] / 60])
    axes[1].set(title="Summed model fit time", ylabel="Minutes")
    axes[2].bar(("direct sum", "TCN"),
                [resources["direct_models"]["sum_model_bytes"] / 2**20,
                 resources["tcn"]["model_bytes"] / 2**20])
    axes[2].set(title="Saved models", ylabel="MiB")
    axes[2].set_yscale("log")
    axes[2].set_ylim(.03, 1000)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=.25)
    save("computational_efficiency.png")
    return files


if __name__ == "__main__":
    for path in make_plots():
        print(path)
