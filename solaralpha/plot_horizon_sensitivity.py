"""SVG plots for the offline planning-horizon experiment."""

from pathlib import Path
from xml.etree import ElementTree as ET

from solaralpha.plot_day import SVG, element, label
from solaralpha.plot_oracle_comparison import draw_panel


STYLES = {
    "decision_self": ("Decision self", "#2563a6"),
    "decision_threshold": ("Decision threshold", "#198754"),
    "oracle_1h": ("Oracle 1h", "#d18a00"),
    "oracle_3h": ("Oracle 3h", "#db6c35"),
    "oracle_6h": ("Oracle 6h", "#c34848"),
    "oracle_12h": ("Oracle 12h", "#7555b2"),
    "oracle_24h": ("Oracle 24h", "#202b40"),
}


def cumulative(rows, key):
    total = 0.0
    values = []
    for _, info, reward in rows:
        total += reward if key == "reward" else info[key]
        values.append(total)
    return values


def reward_horizon_panel(root, top, results):
    left, width, height = 90, 1050, 170
    baseline_keys = ("decision_self", "decision_threshold")
    horizon_keys = sorted((key for key in results if key.startswith("oracle_")),
                          key=lambda key: int(key.split("_")[1][:-1]))
    scores = [metrics["adjusted_reward"] for metrics, _ in results.values()]
    low, high = min(scores), max(scores)
    padding = 0.10 * (high - low or 1.0)
    low, high = low - padding, high + padding
    def y_position(value):
        return top + height * (high - value) / (high - low)

    def x_position(hour):
        return left + width * hour / 24

    label(root, left, top - 19, "Seven-day terminal-adjusted reward versus planning horizon",
          font_size=15, font_weight="bold", fill="#172339")
    label(root, 23, top + 12, "AUD", font_size=11, fill="#576579")
    for fraction in (0, 0.5, 1):
        value = high - fraction * (high - low)
        y = y_position(value)
        element(root, "line", x1=left, y1=y, x2=left + width, y2=y,
                stroke="#dbe2ea", stroke_width=1)
        label(root, left - 9, y + 4, f"{value:.1f}", text_anchor="end",
              font_size=11, fill="#576579")
    for hour in (1, 3, 6, 12, 24):
        x = x_position(hour)
        element(root, "line", x1=x, y1=top, x2=x, y2=top + height,
                stroke="#e8edf2", stroke_width=1)
        label(root, x, top + height + 21, str(hour), text_anchor="middle",
              font_size=11, fill="#576579")
    label(root, left + width / 2, top + height + 42, "Planning horizon (hours)",
          text_anchor="middle", font_size=12, fill="#576579")
    for key in baseline_keys:
        if key not in results:
            continue
        color = STYLES[key][1]
        y = y_position(results[key][0]["adjusted_reward"])
        element(root, "line", x1=left, y1=y, x2=left + width, y2=y,
                stroke=color, stroke_width=2, stroke_dasharray="7 5")
    points = [(x_position(int(key.split("_")[1][:-1])),
               y_position(results[key][0]["adjusted_reward"]))
              for key in horizon_keys]
    if points:
        element(root, "polyline", points=" ".join(f"{x:.2f},{y:.2f}" for x, y in points),
                fill="none", stroke="#202b40", stroke_width=2)
    for key, (x, y) in zip(horizon_keys, points):
        element(root, "circle", cx=x, cy=y, r=5, fill=STYLES[key][1],
                stroke="white", stroke_width=1)


def save_horizon_comparison(scenario, day, episode, results, config):
    """Show a selected day and full-week score across completed horizons."""
    intervals_per_day = round(24 / config.dt)
    start, end = (day - 1) * intervals_per_day, day * intervals_per_day
    selected = episode[start:end]
    if len(selected) != intervals_per_day:
        raise ValueError("Incomplete selected day")
    for key in results:
        if key not in STYLES or len(results[key][1][start:end]) != intervals_per_day:
            raise ValueError(f"Missing or incomplete plot records for {key}")
    date = selected[0].time.date().isoformat()
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": "0 0 1200 1230", "width": "1200", "height": "1230",
        "role": "img", "aria-label": "Battery planning horizon comparison",
    })
    element(root, "rect", x=0, y=0, width=1200, height=1230, fill="white")
    label(root, 90, 34, f"Planning-horizon sensitivity | {date} | {scenario}",
          font_size=21, font_weight="bold", fill="#172339")
    label(root, 90, 60,
          "Daily curves reset at midnight; battery SOC carries from earlier days.",
          font_size=13, fill="#576579")
    for index, key in enumerate(results):
        name, color = STYLES[key]
        x = 90 + (index % 4) * 265
        y = 88 + (index // 4) * 25
        element(root, "line", x1=x, y1=y, x2=x + 23, y2=y,
                stroke=color, stroke_width=3)
        label(root, x + 29, y + 4, name, font_size=12, fill="#344054")

    selected_rows = {key: pair[1][start:end] for key, pair in results.items()}
    draw_panel(root, 165, "Cumulative daily net reward (cash flow minus wear)",
               [(STYLES[key][1], cumulative(rows, "reward"))
                for key, rows in selected_rows.items()], "AUD", show_times=True)
    draw_panel(root, 360, "Battery state of charge",
               [(STYLES[key][1], [100 * info["soc"] for _, info, _ in rows])
                for key, rows in selected_rows.items()], "%", limits=(0, 100),
               show_times=True)
    draw_panel(root, 555, "Cumulative daily grid imports",
               [(STYLES[key][1], cumulative(rows, "import_kwh"))
                for key, rows in selected_rows.items()], "kWh", show_times=True)
    draw_panel(root, 750, "Cumulative daily grid exports",
               [(STYLES[key][1], cumulative(rows, "export_kwh"))
                for key, rows in selected_rows.items()], "kWh", show_times=True)
    reward_horizon_panel(root, 950, results)
    note = "Dashed score lines are decision-time baselines; points use all-oracle MPC."
    if "oracle_24h" not in results:
        note += " 24h was timed but not simulated."
    label(root, 90, 1210, note,
          font_size=12, fill="#576579")
    output = (Path(__file__).resolve().parent.parent / "plots"
              / f"horizon_sensitivity_{date}_{scenario}.svg")
    output.parent.mkdir(exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output
