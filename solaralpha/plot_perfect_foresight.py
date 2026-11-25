"""One-day six-policy SVG comparison for the offline whole-episode benchmark."""

from pathlib import Path
from xml.etree import ElementTree as ET

from solaralpha.perfect_foresight_experiment import POLICIES
from solaralpha.plot_day import SVG, element, label
from solaralpha.plot_oracle_comparison import draw_panel


def save_comparison(scenario, day, episode, results, config):
    count = round(24 / config.dt)
    selected = episode[(day - 1) * count:day * count]
    if len(selected) != count:
        raise ValueError("Incomplete selected day")
    date = selected[0].time.date().isoformat()
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": "0 0 1200 1290", "width": "1200", "height": "1290",
        "role": "img", "aria-label": "Whole-episode policy comparison",
    })
    element(root, "rect", x=0, y=0, width=1200, height=1290, fill="white")
    label(root, 90, 35, f"Whole-episode foresight comparison | {date} | {scenario}",
          font_size=21, font_weight="bold", fill="#172339")
    label(root, 90, 61,
          "Observed or capped spot-linked tariff | SOC carries from earlier days",
          font_size=13, fill="#576579")
    for index, (key, name, color) in enumerate(POLICIES):
        x = 90 + (index % 3) * 350
        y = 88 + (index // 3) * 25
        element(root, "line", x1=x, y1=y, x2=x + 25, y2=y,
                stroke=color, stroke_width=3)
        label(root, x + 32, y + 4, name, font_size=13, fill="#344054")

    def series(field, scale=1):
        return [(color, [scale * row[field] for row in results[key]["trace"]])
                for key, _, color in POLICIES]

    draw_panel(root, 170, "Battery state of charge", series("soc", 100),
               "%", limits=(0, 100))
    draw_panel(root, 390, "Executed battery action (+charge, -discharge)",
               series("power_kw"), "kW", limits=(-5.5, 5.5))
    draw_panel(root, 610, "Grid imports", series("import_kw"), "kW")
    draw_panel(root, 830, "Grid exports", series("export_kw"), "kW",
               limits=(0, config.export_kw * 1.06))
    reward_series = []
    for key, _, color in POLICIES:
        cumulative = []
        amount = 0.0
        for row in results[key]["trace"]:
            amount += row["reward"]
            cumulative.append(amount)
        reward_series.append((color, cumulative))
    draw_panel(root, 1050, "Cumulative daily net reward (cash flow minus wear)",
               reward_series, "AUD", show_times=True)
    label(root, 90, 1275,
          "Daily reward starts at zero; whole-week terminal-adjusted scores are in the results file.",
          font_size=12, fill="#576579")
    path = (Path(__file__).resolve().parent.parent / "plots"
            / f"perfect_foresight_{date}_{scenario}.svg")
    path.parent.mkdir(exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path
