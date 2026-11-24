"""Plot one synthetic day as SVG using only the Python standard library.

Example: python plot_day.py --day 1 --policy threshold
"""

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET

from solar_sim import BatteryEnv, Config, make_data, self_consumption, threshold_trader


SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def element(parent, tag, **attributes):
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", {
        key.replace("_", "-"): str(value) for key, value in attributes.items()
    })


def label(parent, x, y, words, **attributes):
    node = element(parent, "text", x=x, y=y, **attributes)
    node.text = words
    return node


def draw_panel(svg, index, title, series, unit, fixed_range=None):
    left, width = 82, 920
    top = 48 + index * 155
    chart_top, chart_height = top + 48, 96
    values = [value for _, _, points in series for value in points]
    low, high = fixed_range or (min(0.0, *values), max(0.0, *values))
    if fixed_range is None:
        padding = 0.08 * (high - low or 1.0)
        low, high = low - padding, high + padding

    label(svg, left, top + 15, title, font_size=15, font_weight="bold", fill="#172339")
    for j, (name, color, _) in enumerate(series):
        x = left + j * 190
        element(svg, "line", x1=x, y1=top + 34, x2=x + 23, y2=top + 34,
                stroke=color, stroke_width=3)
        label(svg, x + 29, top + 38, name, font_size=12, fill="#344054")

    for fraction in (0, 0.5, 1):
        value = high - fraction * (high - low)
        y = chart_top + fraction * chart_height
        element(svg, "line", x1=left, y1=y, x2=left + width, y2=y,
                stroke="#dbe2ea", stroke_width=1)
        tick = f"{value:.2f}" if high - low < 1 else f"{value:.1f}"
        label(svg, left - 8, y + 4, tick, text_anchor="end",
              font_size=11, fill="#576579")
    label(svg, 12, chart_top + 12, unit, font_size=11, fill="#576579")

    for hour in (0, 6, 12, 18, 24):
        x = left + width * hour / 24
        element(svg, "line", x1=x, y1=chart_top, x2=x, y2=chart_top + chart_height,
                stroke="#e8edf2", stroke_width=1)
        if index == 4:
            label(svg, x, chart_top + chart_height + 17, f"{hour:02d}:00",
                  text_anchor="middle", font_size=11, fill="#576579")

    for _, color, points in series:
        coordinates = " ".join(
            f"{left + width * (i + 0.5) / len(points):.2f},"
            f"{chart_top + chart_height * (high - value) / (high - low):.2f}"
            for i, value in enumerate(points)
        )
        element(svg, "polyline", points=coordinates, fill="none", stroke=color,
                stroke_width=2, stroke_linejoin="round", stroke_linecap="round")


def collect_day(day, seed, policy):
    config = Config()
    intervals_per_day = round(24 / config.dt)
    data = make_data(config, days=day, seed=seed)
    env = BatteryEnv(config, data)
    obs, soc = env.reset()
    rows = []
    for index in range(len(data)):
        next_obs, reward, done, info = env.step(policy(obs, soc, config))
        if index >= (day - 1) * intervals_per_day:
            rows.append((obs, info, reward))
        if done:
            break
        obs, soc = next_obs
    return config, rows


def save_chart(path, day, seed, policy_name, rows, config, title=None,
               spot_prices=None):
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": "0 0 1040 875", "width": "1040", "height": "875",
        "role": "img", "aria-label": "One-day SolarAlpha simulation chart",
    })
    element(root, "rect", x=0, y=0, width=1040, height=875, fill="white")
    date = rows[0][0].time.date().isoformat()
    heading = title or f"SolarAlpha simulated day {day}: {date} | {policy_name} | seed {seed}"
    label(root, 82, 30, heading,
          font_size=20, font_weight="bold", fill="#172339")

    observations = [obs for obs, _, _ in rows]
    infos = [info for _, info, _ in rows]
    cumulative_reward = []
    total = 0.0
    for _, _, reward in rows:
        total += reward
        cumulative_reward.append(total)

    draw_panel(root, 0, "Solar and household load", [
        ("Solar", "#d18a00", [obs.solar_kw for obs in observations]),
        ("Load", "#2563a6", [obs.load_kw for obs in observations]),
    ], "kW")
    price_series = []
    if spot_prices is not None:
        price_series.append(("AEMO spot", "#555b68", spot_prices))
    price_series.extend([
        ("Buy (model)", "#c34848", [obs.buy_price for obs in observations]),
        ("Sell (model)", "#29836c", [obs.sell_price for obs in observations]),
    ])
    price_title = ("Wholesale spot and illustrative household prices"
                   if spot_prices is not None else "Electricity prices")
    draw_panel(root, 1, price_title, price_series, "AUD/kWh")
    draw_panel(root, 2, "Battery and grid power (+charge/import, -discharge/export)", [
        ("Battery", "#7555b2", [info["power_kw"] for info in infos]),
        ("Grid", "#3a677e", [(info["import_kwh"] - info["export_kwh"]) / config.dt
                              for info in infos]),
    ], "kW")
    draw_panel(root, 3, "Battery state of charge after each interval", [
        ("SOC", "#7555b2", [100 * info["soc"] for info in infos]),
    ], "%", fixed_range=(0, 100))
    draw_panel(root, 4, "Cumulative net reward (export revenue - import cost - wear)", [
        ("Net reward", "#29836c", cumulative_reward),
    ], "AUD")

    imports = sum(info["import_kwh"] for info in infos)
    exports = sum(info["export_kwh"] for info in infos)
    curtailed = sum(info["curtailed_kwh"] for info in infos)
    summary = (f"Day totals: import {imports:.2f} kWh  |  export {exports:.2f} kWh  |  "
               f"curtailed {curtailed:.2f} kWh  |  net reward ${total:.2f}  |  "
               f"final SOC {100 * infos[-1]['soc']:.1f}%")
    label(root, 82, 856, summary, font_size=12, fill="#344054")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=int, default=1, help="Summer day number, starting at 1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=("self", "threshold"), default="threshold")
    parser.add_argument("--output", help="SVG filename inside plots/ (defaults to day_NN_policy.svg)")
    args = parser.parse_args()
    if args.day < 1:
        parser.error("--day must be at least 1")
    policy = self_consumption if args.policy == "self" else threshold_trader
    filename = args.output or f"day_{args.day:02d}_{args.policy}.svg"
    if Path(filename).name != filename or not filename.lower().endswith(".svg"):
        parser.error("--output must be an SVG filename, not a path")
    output = Path(__file__).resolve().parent / "plots" / filename
    output.parent.mkdir(exist_ok=True)
    config, rows = collect_day(args.day, args.seed, policy)
    summary = save_chart(output, args.day, args.seed, args.policy, rows, config)
    print(f"Wrote {output}")
    print(summary)


if __name__ == "__main__":
    main()
