"""Compare the January smoke test with the frozen replay and plot its dev week."""

import csv
from datetime import datetime, timedelta, timezone
from math import copysign, log1p
from pathlib import Path
from statistics import mean, median
from xml.etree import ElementTree as ET

from solaralpha.historical_pipeline import PROCESSED, ROOT, _write_json_atomic
from solaralpha.historical_replay import build_observations, load_hourly_weather_field
from solaralpha.plot_day import SVG, element, label
from solaralpha.solar_sim import Config


def read_month(path=PROCESSED / "2025-01_realised.csv"):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _differences(new, old):
    absolute = [abs(a - b) for a, b in zip(new, old)]
    return {"exact_matches": sum(a == b for a, b in zip(new, old)),
            "mean_absolute_difference": mean(absolute),
            "max_absolute_difference": max(absolute)}


def compare_development_week(rows, path=PROCESSED / "2025-01_replay_comparison.json"):
    selected = [row for row in rows if row["is_development_week"] == "1"]
    old, old_prices_mwh = build_observations(Config())
    if len(selected) != len(old):
        raise ValueError("Development week has a different interval count")
    for row, observation in zip(selected, old):
        instant = datetime.fromisoformat(row["interval_start_utc"].replace("Z", "+00:00"))
        if instant != observation.time.astimezone(timezone.utc):
            raise ValueError("Development-week timestamp alignment differs")
    fields = {
        "spot_aud_per_kwh": (
            [float(row["realised_spot_aud_per_kwh"]) for row in selected],
            [old_prices_mwh[observation.time.astimezone(timezone.utc)] / 1000
             for observation in old]),
        "illustrative_buy_aud_per_kwh": (
            [float(row["realised_illustrative_buy_aud_per_kwh"]) for row in selected],
            [observation.buy_price for observation in old]),
        "illustrative_sell_aud_per_kwh": (
            [float(row["realised_illustrative_sell_aud_per_kwh"]) for row in selected],
            [observation.sell_price for observation in old]),
        "solar_proxy_kw": (
            [float(row["realised_solar_proxy_kw"]) for row in selected],
            [observation.solar_kw for observation in old]),
        "modeled_household_load_kw": (
            [float(row["realised_modeled_household_load_kw"]) for row in selected],
            [observation.load_kw for observation in old]),
    }
    results = {name: _differences(*pair) for name, pair in fields.items()}
    for field, column in (("temperature_2m", "realised_temperature_c"),
                          ("cloud_cover", "realised_cloud_cover_pct"),
                          ("shortwave_radiation", "realised_shortwave_radiation_w_m2")):
        old_weather = load_hourly_weather_field(field)
        new_values, old_values = [], []
        for row in selected:
            instant = datetime.fromisoformat(row["interval_start_utc"].replace("Z", "+00:00"))
            hour = instant.replace(minute=0, second=0, microsecond=0)
            if field == "shortwave_radiation":
                hour += timedelta(hours=1)
            new_values.append(float(row[column]))
            old_values.append(old_weather[hour])
        results[field] = _differences(new_values, old_values)
    payload = {
        "intervals": len(selected), "timestamps_match": True,
        "comparison": results,
        "source_note": ("New January weather is explicit ERA5; the earlier nine-day "
                        "snapshot used Open-Meteo's unpinned default. Both are "
                        "reanalysis, not decision-time forecasts."),
        "noise_note": ("The same seed and load equation now start at January 1 "
                       "rather than January 13, so the noise sequence differs."),
        "benchmark_inputs_modified": False,
    }
    _write_json_atomic(path, payload)
    return payload


def _root(title, height):
    root = ET.Element(f"{{{SVG}}}svg", {
        "viewBox": f"0 0 1200 {height}", "width": "1200", "height": str(height),
        "role": "img", "aria-label": title,
    })
    element(root, "rect", x=0, y=0, width=1200, height=height, fill="white")
    label(root, 80, 35, title, font_size=21, font_weight="bold", fill="#172339")
    return root


def _panel(root, top, heading, values, unit, color, low, high, labels):
    left, width, height = 90, 1040, 170
    label(root, left, top-18, heading, font_size=15, font_weight="bold", fill="#172339")
    label(root, 17, top+12, unit, font_size=11, fill="#576579")
    for fraction in (0, .5, 1):
        y = top + height*fraction
        element(root, "line", x1=left, y1=y, x2=left+width, y2=y,
                stroke="#dbe2ea", stroke_width=1)
        label(root, left-8, y+4, f"{high-fraction*(high-low):.2f}",
              text_anchor="end", font_size=11, fill="#576579")
    for fraction, words in labels:
        x = left + width*fraction
        element(root, "line", x1=x, y1=top, x2=x, y2=top+height,
                stroke="#e8edf2", stroke_width=1)
        label(root, x, top+height+19, words, text_anchor="middle",
              font_size=11, fill="#576579")
    coordinates = " ".join(
        f"{left+width*(i+.5)/len(values):.2f},"
        f"{top+height*(high-value)/(high-low):.2f}"
        for i, value in enumerate(values))
    element(root, "polyline", points=coordinates, fill="none", stroke=color,
            stroke_width=2, stroke_linejoin="round", stroke_linecap="round")


def _save(root, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def plot_development_prices(rows, path=ROOT / "plots" / "pipeline_dev_prices_2025-01.svg"):
    sample = [row for row in rows if row["is_development_week"] == "1"]
    prices = [float(row["realised_spot_aud_per_kwh"]) for row in sample]
    root = _root("January 13–19 development week | NSW1 realised spot prices", 1190)
    label(root, 90, 62, "AUD/kWh; historical outcomes, not decision-time forecasts",
          font_size=13, fill="#576579")
    bins = [(-float("inf"), -.1), (-.1, 0), (0, .05), (.05, .1),
            (.1, .2), (.2, .5), (.5, 1), (1, 5), (5, float("inf"))]
    names = ("<−.10", "−.10–0", "0–.05", ".05–.10", ".10–.20",
             ".20–.50", ".50–1", "1–5", ">=5")
    counts = [sum(lo <= value < hi for value in prices) for lo, hi in bins]
    top, left, width, height = 132, 90, 1040, 150
    label(root, left, top-21, "Distribution by price band", font_size=15,
          font_weight="bold", fill="#172339")
    for i, (name, count) in enumerate(zip(names, counts)):
        x = left + i*width/len(counts) + 8
        w = width/len(counts) - 16
        h = height*count/max(counts)
        element(root, "rect", x=x, y=top+height-h, width=w, height=h, fill="#2563a6")
        label(root, x+w/2, top+height-h-5, str(count), text_anchor="middle",
              font_size=10, fill="#344054")
        label(root, x+w/2, top+height+19, name, text_anchor="middle",
              font_size=11, fill="#576579")
    days = [prices[i*288:(i+1)*288] for i in range(7)]
    negative = [100*sum(value < 0 for value in day)/len(day) for day in days]
    day_labels = [((i+.5)/7, f"Jan {13+i}") for i in range(7)]
    _panel(root, 395, "Negative-price frequency by Sydney day", negative,
           "%", "#c34848", 0, max(10, max(negative)*1.1), day_labels)
    medians = [median(day[step] for day in days) for step in range(288)]
    _panel(root, 655, "Median price by local time of day", medians,
           "AUD/kWh", "#198754", -.1, max(.5, max(medians)*1.05),
           [(i/4, f"{6*i:02d}:00") for i in range(5)])
    transformed = [copysign(log1p(abs(value)/.1), value) for value in prices]
    bound = max(abs(min(transformed)), abs(max(transformed))) * 1.02
    _panel(root, 915, "Full week with extreme events (symmetric-log scale)",
           transformed, "symlog", "#7555b2", -bound, bound,
           [(i/7, f"Jan {13+i}") for i in range(8)])
    label(root, 90, 1175,
          "Symlog = sign(price) × ln(1 + |price| / $0.10); histogram bands are AUD/kWh.",
          font_size=12, fill="#576579")
    return _save(root, path)


def plot_development_weather(rows, path=ROOT / "plots" / "pipeline_dev_weather_2025-01.svg"):
    sample = [row for row in rows if row["is_development_week"] == "1"]
    root = _root("January 13–19 development week | ERA5 Sydney weather", 940)
    label(root, 90, 62, "Hourly reanalysis repeated across five-minute rows; not a live forecast",
          font_size=13, fill="#576579")
    days = [(i/7, f"Jan {13+i}") for i in range(8)]
    for top, title, field, unit, color, bounds in (
            (130, "Temperature at hour start", "realised_temperature_c", "°C",
             "#c34848", (0, 45)),
            (390, "Preceding-hour shortwave radiation", "realised_shortwave_radiation_w_m2",
             "W/m²", "#d18a00", (0, 1200)),
            (650, "Cloud cover at hour start", "realised_cloud_cover_pct", "%",
             "#2563a6", (0, 100))):
        _panel(root, top, title, [float(row[field]) for row in sample], unit,
               color, *bounds, days)
    return _save(root, path)


def main():
    rows = read_month()
    comparison = compare_development_week(rows)
    for name, result in comparison["comparison"].items():
        print(f"{name}: exact {result['exact_matches']}/2016, "
              f"MAE {result['mean_absolute_difference']:.6g}, "
              f"max {result['max_absolute_difference']:.6g}")
    for path in (plot_development_prices(rows), plot_development_weather(rows)):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
