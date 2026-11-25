"""SVG comparison of the six policies on one offline diagnostic day."""

from pathlib import Path
from xml.etree import ElementTree as ET

from solaralpha.plot_day import SVG, element, label


POLICY_STYLES = (
    ("decision_self", "Decision self", "#2563a6"),
    ("decision_threshold", "Decision threshold", "#198754"),
    ("persistence_mpc", "Persistence MPC", "#d18a00"),
    ("oracle_prices", "Oracle prices", "#c34848"),
    ("oracle_solar_load", "Oracle solar/load", "#7555b2"),
    ("oracle_all", "Oracle all", "#202b40"),
)


def draw_panel(root, top, title, series, unit, limits=None, clip=False,
               show_times=False):
    left, width, height = 90, 1050, 155
    values = [value for _, points in series for value in points]
    low, high = limits or (min(0.0, *values), max(0.0, *values))
    if limits is None:
        padding = 0.08 * (high - low or 1.0)
        low, high = low - padding, high + padding
    label(root, left, top - 19, title, font_size=15, font_weight="bold", fill="#172339")
    for fraction in (0, 0.5, 1):
        y = top + height * fraction
        value = high - fraction * (high - low)
        element(root, "line", x1=left, y1=y, x2=left + width, y2=y,
                stroke="#dbe2ea", stroke_width=1)
        label(root, left - 9, y + 4, f"{value:.2f}" if high - low < 1 else f"{value:.1f}",
              text_anchor="end", font_size=11, fill="#576579")
    label(root, 20, top + 12, unit, font_size=11, fill="#576579")
    for hour in (0, 6, 12, 18, 24):
        x = left + width * hour / 24
        element(root, "line", x1=x, y1=top, x2=x, y2=top + height,
                stroke="#e8edf2", stroke_width=1)
        if show_times:
            label(root, x, top + height + 21, f"{hour:02d}:00",
                  text_anchor="middle", font_size=11, fill="#576579")
    for color, points in series:
        coordinates = " ".join(
            f"{left + width * (index + 0.5) / len(points):.2f},"
            f"{top + height * (high - (min(high, max(low, value)) if clip else value)) / (high - low):.2f}"
            for index, value in enumerate(points)
        )
        element(root, "polyline", points=coordinates, fill="none", stroke=color,
                stroke_width=2, stroke_linejoin="round", stroke_linecap="round")


def save_comparison(scenario, day, episode, results, config, horizon_steps=12,
                    internal_terminal_value_per_kwh=0.20):
    intervals_per_day = round(24 / config.dt)
    start, end = (day - 1) * intervals_per_day, day * intervals_per_day
    selected = episode[start:end]
    if len(selected) != intervals_per_day:
        raise ValueError("Incomplete selected day")
    selected_rows = {name: results[name][1][start:end]
                     for name, _, _ in POLICY_STYLES}
    date = selected[0].time.date().isoformat()
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": "0 0 1200 1200", "width": "1200", "height": "1200",
        "role": "img", "aria-label": "Oracle forecast experiment policy comparison",
    })
    element(root, "rect", x=0, y=0, width=1200, height=1200, fill="white")
    label(root, 90, 34,
          f"Forecast-oracle comparison | {date} | {scenario} | {horizon_steps} steps"
          f" | value ${internal_terminal_value_per_kwh:.2f}/kWh",
          font_size=21, font_weight="bold", fill="#172339")
    price_note = ("Recorded spot-linked prices" if scenario == "observed"
                  else "Counterfactual spot-linked prices, spot capped at $1.10/kWh")
    label(root, 90, 60, price_note + " | SOC carries from earlier days",
          font_size=13, fill="#576579")
    for index, (_, name, color) in enumerate(POLICY_STYLES):
        x = 90 + (index % 3) * 350
        y = 88 + (index // 3) * 25
        element(root, "line", x1=x, y1=y, x2=x + 25, y2=y,
                stroke=color, stroke_width=3)
        label(root, x + 32, y + 4, name, font_size=13, fill="#344054")

    price_series = (("#c34848", [obs.buy_price for obs in selected]),
                    ("#29836c", [obs.sell_price for obs in selected]))
    draw_panel(root, 160, "Modeled household prices: buy red, sell green",
               price_series, "AUD/kWh")
    draw_panel(root, 360, "Price detail (values outside -$0.10 to $0.60/kWh clipped)",
               price_series, "AUD/kWh", limits=(-0.10, 0.60), clip=True)
    draw_panel(root, 560, "Battery state of charge", [
        (color, [100 * info["soc"] for _, info, _ in selected_rows[key]])
        for key, _, color in POLICY_STYLES], "%", limits=(0, 100))
    draw_panel(root, 760, "Executed battery action (+charge, -discharge)", [
        (color, [info["power_kw"] for _, info, _ in selected_rows[key]])
        for key, _, color in POLICY_STYLES], "kW", limits=(-5.5, 5.5))
    reward_series = []
    for key, _, color in POLICY_STYLES:
        total = 0.0
        cumulative = []
        for _, _, reward in selected_rows[key]:
            total += reward
            cumulative.append(total)
        reward_series.append((color, cumulative))
    draw_panel(root, 960, "Cumulative daily net reward (cash flow minus wear)",
               reward_series, "AUD", show_times=True)
    label(root, 90, 1180,
          "Daily reward starts at zero; seven-day terminal-adjusted results are in the console table.",
          font_size=12, fill="#576579")
    suffix = "" if horizon_steps == 12 else f"_{horizon_steps}step"
    if internal_terminal_value_per_kwh != 0.20:
        suffix += f"_{internal_terminal_value_per_kwh:.2f}value"
    output = (Path(__file__).resolve().parent.parent / "plots"
              / f"oracle_comparison_{date}_{scenario}{suffix}.svg")
    output.parent.mkdir(exist_ok=True)
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return output
