"""Static scientific plots for the chronological validation baseline comparison."""

from datetime import datetime
import os

from solaralpha.historical_pipeline import ROOT
from .baselines import PREDICTION_FIELDS, read_prediction_rows


PLOT_DIR = ROOT / "plots" / "forecasting"
LABELS = {
    "persistence_aud_per_kwh": "Persistence",
    "daily_seasonal_aud_per_kwh": "Previous-day seasonal",
    "train_median_aud_per_kwh": "Training median",
}
COLORS = {"persistence_aud_per_kwh": "#0072B2",
          "daily_seasonal_aud_per_kwh": "#D55E00",
          "train_median_aud_per_kwh": "#009E73"}


def make_plots(prediction_path=None, output_dir=PLOT_DIR):
    cache_dir = ROOT / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(read_prediction_rows(prediction_path) if prediction_path
                else read_prediction_rows())
    if not rows:
        raise ValueError("No scored validation rows to plot")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = [float(r["actual_aud_per_kwh"]) for r in rows]
    names = PREDICTION_FIELDS[-3:]
    step = max(1, len(rows) // 18000)
    sample = range(0, len(rows), step)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, name in zip(axes, names):
        predicted = [float(r[name]) for r in rows]
        ax.scatter([actual[i] for i in sample], [predicted[i] for i in sample],
                   s=2, alpha=0.22, color=COLORS[name], rasterized=True)
        ax.plot([-25, 25], [-25, 25], color="black", lw=0.8)
        ax.set(xscale="symlog", yscale="symlog", xlim=(-2, 25), ylim=(-2, 25),
               title=LABELS[name], xlabel="Actual (AUD/kWh)",
               ylabel="Forecast (AUD/kWh)")
        ax.grid(alpha=0.2)
    scatter = output_dir / "prediction_vs_actual.png"
    fig.savefig(scatter, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    for name in names:
        errors = [float(r[name]) - a for r, a in zip(rows, actual)]
        central = [e for e in errors if -1 <= e <= 1]
        ax.hist(central, bins=100, range=(-1, 1), histtype="step", lw=1.3,
                label=f"{LABELS[name]} ({len(errors)-len(central):,} outside ±1)",
                color=COLORS[name], density=True)
    ax.set(title="Validation forecast errors (central ±$1/kWh window)",
           xlabel="Forecast minus actual (AUD/kWh)", ylabel="Density")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    error_plot = output_dir / "error_distributions.png"
    fig.savefig(error_plot, dpi=150)
    plt.close(fig)

    event_row = max(rows, key=lambda r: abs(float(r["actual_aud_per_kwh"])))
    dates = ("2025-03-10", event_row["origin_local_date"])
    daily_paths = []
    for label, day in zip(("ordinary", "extreme_event"), dates):
        daily = [r for r in rows if r["origin_local_date"] == day]
        if not daily:
            raise ValueError(f"No scored data for representative day {day}")
        times = [datetime.fromisoformat(r["target_interval_start_utc"].replace("Z", "+00:00"))
                 for r in daily]
        fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
        ax.plot(times, [float(r["actual_aud_per_kwh"]) for r in daily],
                label="Actual", color="black", lw=1.3)
        for name in names:
            ax.plot(times, [float(r[name]) for r in daily],
                    label=LABELS[name], color=COLORS[name], alpha=0.8, lw=1)
        ax.set(title=f"One-hour-ahead forecasts, origins {day} Sydney local",
               xlabel="Target interval start (UTC)", ylabel="Price (AUD/kWh)")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.2)
        fig.autofmt_xdate()
        path = output_dir / f"daily_{label}_{day}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        daily_paths.append(path)
    return {"prediction_vs_actual": scatter, "error_distributions": error_plot,
            "daily": daily_paths, "event_day_selection":
            "Validation day containing the scored interval with maximum absolute price",
            "event_day": event_row["origin_local_date"]}
