"""Standard-library SVG views of the offline terminal-valuation sweep."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from solaralpha.plot_day import SVG, element, label
from solaralpha.plot_oracle_comparison import draw_panel
from solaralpha.terminal_sensitivity import (HOURS, RESULT_PATH, VALUES,
                                             result_key)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS = PROJECT_ROOT / "plots"
HOUR_COLORS = {1: "#d18a00", 6: "#198754", 12: "#7555b2"}
VALUE_COLORS = {0.00: "#c34848", 0.05: "#d18a00", 0.10: "#198754",
                0.20: "#2563a6", 0.30: "#7555b2", 0.40: "#202b40"}


def load_results(path=RESULT_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["results"]


def svg_root(width, height, description):
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": f"0 0 {width} {height}",
        "width": str(width), "height": str(height),
        "role": "img", "aria-label": description,
    })
    element(root, "rect", x=0, y=0, width=width, height=height, fill="white")
    return root


def save_svg(root, filename):
    PLOTS.mkdir(exist_ok=True)
    path = PLOTS / filename
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def cell_color(fraction, kind):
    fraction = max(0.0, min(1.0, fraction))
    low, high = ((246, 231, 226), (45, 83, 151)) if kind == "adjusted" else (
        (236, 246, 239), (178, 61, 47))
    channels = [round(a + (b - a) * fraction) for a, b in zip(low, high)]
    return "#" + "".join(f"{value:02x}" for value in channels)


def save_heatmap(results, kind, scenario="observed", allow_missing=False):
    """Show a complete sweep, or mark untested capped cells explicitly."""
    metric = "adjusted_reward" if kind == "adjusted" else "import_kwh"
    numbers = []
    for hour in HOURS:
        row = []
        for value in VALUES:
            key = result_key(scenario, hour, value)
            if key not in results:
                if not allow_missing:
                    raise ValueError(f"Missing grid result {key}")
                row.append(None)
            else:
                row.append(results[key]["metrics"][metric])
        numbers.append(row)
    present = [number for row in numbers for number in row if number is not None]
    if not present:
        raise ValueError(f"No results for {scenario}")
    low, high = min(present), max(present)
    root = svg_root(1000, 480, f"{kind} by planning horizon and terminal value")
    title = ("Terminal-adjusted reward" if kind == "adjusted"
             else "Grid imports")
    unit = "AUD" if kind == "adjusted" else "kWh"
    label(root, 70, 45, f"{title} | {scenario} seven-day replay",
          font_size=23, font_weight="bold", fill="#172339")
    label(root, 70, 73,
          "Internal planner value varies; evaluation value remains $0.20/kWh.",
          font_size=13, fill="#576579")
    left, top, width, height = 195, 145, 118, 82
    for column, value in enumerate(VALUES):
        x = left + column * width + width / 2
        label(root, x, 120, f"${value:.2f}", text_anchor="middle",
              font_size=14, fill="#344054")
    label(root, left + 3 * width, 101, "Internal terminal value (AUD/kWh)",
          text_anchor="middle", font_size=13, fill="#576579")
    for row_index, hour in enumerate(HOURS):
        y = top + row_index * height
        label(root, left - 19, y + height / 2 + 5, f"{hour} h",
              text_anchor="end", font_size=15, fill="#344054")
        for column, number in enumerate(numbers[row_index]):
            x = left + column * width
            if number is None:
                element(root, "rect", x=x, y=y, width=width - 5, height=height - 5,
                        rx=7, fill="#e6e9ed")
                label(root, x + (width - 5) / 2, y + height / 2 + 5,
                      "not run", text_anchor="middle", font_size=12,
                      fill="#576579")
                continue
            fraction = (number - low) / (high - low or 1.0)
            element(root, "rect", x=x, y=y, width=width - 5, height=height - 5,
                    rx=7, fill=cell_color(fraction, kind))
            label(root, x + (width - 5) / 2, y + height / 2 + 5,
                  f"{number:.2f}", text_anchor="middle", font_size=16,
                  font_weight="bold", fill="white" if fraction > 0.55 else "#172339")
    note = (f"Light = lower {unit}; dark = higher {unit}. "
            "$0.20 cells reuse rounded prior results.")
    if allow_missing:
        note += " Grey = not run."
    label(root, 195, 425, note,
          font_size=12, fill="#576579")
    return save_svg(root, f"terminal_{kind}_heatmap_{scenario}.svg")


def save_soc_actions(results, hour, values, scenario="observed"):
    """Compare one day's realized SOC and battery action at fixed horizon."""
    entries = []
    for value in values:
        key = result_key(scenario, hour, value)
        entry = results.get(key)
        if entry is None or entry["trace"] is None:
            raise ValueError(f"Daily trace missing for {key}; recompute that setting")
        entries.append((value, entry["trace"]))
    date = entries[0][1][0]["time"][:10]
    root = svg_root(1200, 860, "SOC and actions under terminal-value choices")
    label(root, 90, 36,
          f"Battery behavior | {date} | {scenario} | {hour}-hour horizon",
          font_size=21, font_weight="bold", fill="#172339")
    label(root, 90, 63,
          "SOC carries from earlier days; actions are executed AC battery power.",
          font_size=13, fill="#576579")
    for index, (value, _) in enumerate(entries):
        x = 90 + index * 260
        element(root, "line", x1=x, y1=92, x2=x + 26, y2=92,
                stroke=VALUE_COLORS[value], stroke_width=3)
        label(root, x + 33, 96, f"Planner ${value:.2f}/kWh",
              font_size=12, fill="#344054")
    reference = entries[0][1]
    draw_panel(root, 160, "Recorded household buy/sell price (red/green)",
               [("#c34848", [point["buy_price"] for point in reference]),
                ("#198754", [point["sell_price"] for point in reference])],
               "AUD/kWh", show_times=True)
    draw_panel(root, 370, "Battery state of charge",
               [(VALUE_COLORS[value], [100 * point["soc_after"] for point in trace])
                for value, trace in entries], "%", limits=(0, 100),
               show_times=True)
    draw_panel(root, 580, "Executed action (+charge, -discharge)",
               [(VALUE_COLORS[value], [point["power_kw"] for point in trace])
                for value, trace in entries], "kW", limits=(-5.5, 5.5),
               show_times=True)
    label(root, 90, 822,
          "One selected day only; whole-week financial totals appear in the heatmaps.",
          font_size=12, fill="#576579")
    return save_svg(root, f"terminal_soc_actions_{hour}h_{date}_{scenario}.svg")


def save_reward_throughput(results, scenario="observed"):
    entries = [(hour, value, results[result_key(scenario, hour, value)]["metrics"])
               for hour in HOURS for value in VALUES]
    x_values = [metrics["throughput_kwh"] for _, _, metrics in entries]
    y_values = [metrics["adjusted_reward"] for _, _, metrics in entries]
    x_low, x_high = min(x_values) - 10, max(x_values) + 10
    y_low, y_high = min(y_values) - 3, max(y_values) + 3
    left, top, width, height = 100, 130, 990, 445
    x_pos = lambda value: left + width * (value - x_low) / (x_high - x_low)
    y_pos = lambda value: top + height * (y_high - value) / (y_high - y_low)
    root = svg_root(1200, 690, "Adjusted financial reward versus battery throughput")
    label(root, 100, 42,
          f"Reward versus battery throughput | {scenario} seven-day replay",
          font_size=22, font_weight="bold", fill="#172339")
    label(root, 100, 68,
          "Each point is one internal terminal value; reward uses fixed $0.20/kWh evaluation value.",
          font_size=13, fill="#576579")
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y_value = y_low + fraction * (y_high - y_low)
        y = y_pos(y_value)
        element(root, "line", x1=left, y1=y, x2=left + width, y2=y,
                stroke="#e8edf2", stroke_width=1)
        label(root, left - 12, y + 4, f"{y_value:.1f}",
              text_anchor="end", font_size=11, fill="#576579")
        x_value = x_low + fraction * (x_high - x_low)
        x = x_pos(x_value)
        element(root, "line", x1=x, y1=top, x2=x, y2=top + height,
                stroke="#e8edf2", stroke_width=1)
        label(root, x, top + height + 22, f"{x_value:.0f}",
              text_anchor="middle", font_size=11, fill="#576579")
    label(root, left + width / 2, 622, "Executed AC charge + discharge (kWh)",
          text_anchor="middle", font_size=13, fill="#576579")
    label(root, 20, 119, "AUD", font_size=12, fill="#576579")
    for hour, value, metrics in entries:
        x, y = x_pos(metrics["throughput_kwh"]), y_pos(metrics["adjusted_reward"])
        point = element(root, "circle", cx=x, cy=y, r=6,
                        fill=HOUR_COLORS[hour], stroke="white", stroke_width=1)
        element(point, "title").text = (
            f"{hour} h, planner ${value:.2f}/kWh: "
            f"{metrics['throughput_kwh']:.2f} kWh, ${metrics['adjusted_reward']:.2f}")
    for index, hour in enumerate(HOURS):
        x = 100 + index * 220
        element(root, "circle", cx=x, cy=660, r=6, fill=HOUR_COLORS[hour])
        label(root, x + 15, 665, f"{hour} h", font_size=12, fill="#344054")
    label(root, 770, 665, "Hover points for valuation and totals.",
          font_size=12, fill="#576579")
    return save_svg(root, f"terminal_reward_throughput_{scenario}.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULT_PATH)
    args = parser.parse_args()
    results = load_results(args.results)
    for kind in ("adjusted", "imports"):
        print(f"Wrote {save_heatmap(results, kind)}")
        if any(key.startswith("capped:") for key in results):
            print(f"Wrote {save_heatmap(results, kind, 'capped', allow_missing=True)}")
    print(f"Wrote {save_reward_throughput(results)}")
    one_hour_values = (0.00, 0.10, 0.40)
    original = results.get(result_key("observed", 1, 0.20))
    if original is not None and original["trace"] is not None:
        one_hour_values = (0.00, 0.10, 0.20, 0.40)
    for hour, values in ((1, one_hour_values),
                         (6, (0.00, 0.10, 0.40))):
        print(f"Wrote {save_soc_actions(results, hour, values)}")


if __name__ == "__main__":
    main()
