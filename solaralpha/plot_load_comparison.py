"""Compare modeled household load on two historical Sydney days.

Example: python -m solaralpha.plot_load_comparison
"""

import argparse
from datetime import date
from math import ceil
from pathlib import Path
from xml.etree import ElementTree as ET

from solaralpha.historical_replay import DAYS, START_LOCAL, build_observations
from solaralpha.plot_day import SVG, element, label
from solaralpha.solar_sim import Config


def day_load(observations, selected_date, intervals_per_day):
    day_index = (selected_date - START_LOCAL.date()).days
    if not 0 <= day_index < DAYS:
        raise ValueError(f"Choose a Sydney day from {START_LOCAL.date()} through "
                         f"{date.fromordinal(START_LOCAL.date().toordinal() + DAYS - 1)}")
    records = observations[day_index * intervals_per_day:(day_index + 1) * intervals_per_day]
    return [obs.load_kw for obs in records]


def save_comparison(path, dates, loads, dt):
    left, width = 82, 920
    top, height = 112, 260
    upper = ceil(max(max(series) for series in loads) * 2) / 2
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": "0 0 1040 440", "width": "1040", "height": "440",
        "role": "img", "aria-label": "Comparison of modeled household load on two Sydney days",
    })
    element(root, "rect", x=0, y=0, width=1040, height=440, fill="white")
    label(root, left, 33, "Modeled household load: warm and cool days",
          font_size=20, font_weight="bold", fill="#172339")
    colors = ("#c34848", "#2563a6")
    for index, (selected_date, series, color) in enumerate(zip(dates, loads, colors)):
        x = left + index * 370
        element(root, "line", x1=x, y1=65, x2=x + 28, y2=65,
                stroke=color, stroke_width=3)
        label(root, x + 35, 69,
              f"{selected_date}: {sum(series) * dt:.1f} kWh total",
              font_size=14, fill="#344054")

    # Draw both five-minute series against one load-only scale.
    element(root, "rect", x=left + width / 2, y=top, width=width / 3,
            height=height, fill="#fff5e9")  # noon to 20:00
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = top + height * fraction
        value = upper * (1 - fraction)
        element(root, "line", x1=left, y1=y, x2=left + width, y2=y,
                stroke="#dbe2ea", stroke_width=1)
        label(root, left - 8, y + 4, f"{value:.2f}", text_anchor="end",
              font_size=11, fill="#576579")
    label(root, 23, top + 12, "kW", font_size=11, fill="#576579")
    for hour in (0, 6, 12, 18, 24):
        x = left + width * hour / 24
        element(root, "line", x1=x, y1=top, x2=x, y2=top + height,
                stroke="#e8edf2", stroke_width=1)
        label(root, x, top + height + 23, f"{hour:02d}:00",
              text_anchor="middle", font_size=11, fill="#576579")
    for series, color in zip(loads, colors):
        points = " ".join(
            f"{left + width * (i + 0.5) / len(series):.2f},"
            f"{top + height * (upper - value) / upper:.2f}"
            for i, value in enumerate(series)
        )
        element(root, "polyline", points=points, fill="none", stroke=color,
                stroke_width=2.5, stroke_linejoin="round", stroke_linecap="round")
    label(root, left, 424,
          "Shaded: 12:00–20:00. Lines show five-minute modeled power; totals show daily energy.",
          font_size=12, fill="#576579")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=date.fromisoformat, default=date(2025, 1, 15))
    parser.add_argument("--second", type=date.fromisoformat, default=date(2025, 1, 17))
    args = parser.parse_args()
    config = Config()
    observations, _ = build_observations(config)
    intervals_per_day = round(24 / config.dt)
    dates = (args.first, args.second)
    try:
        loads = [day_load(observations, day, intervals_per_day) for day in dates]
    except ValueError as error:
        parser.error(str(error))
    output = (Path(__file__).resolve().parent.parent / "plots"
              / f"load_comparison_{dates[0]}_{dates[1]}.svg")
    output.parent.mkdir(exist_ok=True)
    save_comparison(output, dates, loads, config.dt)
    print(f"Wrote {output}")
    for day, series in zip(dates, loads):
        print(f"{day}: {sum(series) * config.dt:.1f} kWh; 16:00 {series[16 * 12]:.2f} kW")


if __name__ == "__main__":
    main()
