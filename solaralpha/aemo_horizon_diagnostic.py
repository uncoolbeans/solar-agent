"""Exploratory matched January 2025 AEMO forecast-horizon diagnostics.

No agent imports this module. Historical release timestamps are unverified;
all availability checks are labeled sensitivity assumptions.
"""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path

from solaralpha.aemo_forecast_archive import (Forecast, ROOT, latest_available as latest_p5,
                                              read_forecasts as read_p5)
from solaralpha.aemo_predispatch_archive import (PredispatchForecast,
                                                 latest_available as latest_pd,
                                                 read_forecasts as read_pd)
from solaralpha.historical_pipeline import _write_json_atomic
from solaralpha.price_forecasting.ml_metrics import full_scores


JANUARY = ROOT / "data" / "processed" / "2025-01_realised.csv"
OUT = ROOT / "data" / "derived" / "aemo_horizon_2025_01.json"
P5_HORIZONS = (0, 5, 10, 15, 20, 30, 40, 45, 50, 55, 60)
PD_HOURS = (1, 3, 6, 12)


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    def at(q):
        index = (len(ordered) - 1) * q
        lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {"min": ordered[0], "p05": at(0.05), "median": at(0.5),
            "p95": at(0.95), "max": ordered[-1]}


def realised_prices(path=JANUARY):
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    prices = {datetime.fromisoformat(row["interval_start_utc"]):
              float(row["realised_spot_aud_per_kwh"]) for row in rows}
    if len(prices) != len(rows):
        raise ValueError("Duplicate realised interval starts")
    return prices


def half_hour_average(prices, start):
    slots = [start + timedelta(minutes=5 * i) for i in range(6)]
    if any(slot not in prices for slot in slots):
        return None
    return sum(prices[slot] for slot in slots) / 6


def scores(actual, prediction, persistence):
    return {"aemo": full_scores(actual, prediction),
            "matched_persistence": full_scores(actual, persistence)} if actual else None


def p5_diagnostic(prices):
    starts = sorted(prices)
    rows = defaultdict(list)
    for f in read_p5("P5MIN_REGIONSOLUTION", start=starts[0],
                     end=starts[-1] + timedelta(minutes=65)):
        if f.price_aud_per_kwh is not None:
            # Keep only fields needed for scoring, not 117 CSV values per vintage.
            rows[f.target_start_utc].append(Forecast(f.table, f.run_utc,
                f.target_start_utc, f.price_aud_per_kwh, f.intervention, {}))
    output = {}
    for delay_minutes in (0, 1, 5, 10, 15):
        by_horizon = {}
        delay = timedelta(minutes=delay_minutes)
        for horizon in P5_HORIZONS:
            offset = timedelta(minutes=horizon)
            eligible = [(t, t + offset) for t in starts
                        if t - timedelta(minutes=5) in prices and t + offset in prices]
            actual, predicted, persistence = [], [], []
            ages, source_leads, run_ages = [], [], []
            for decision, target in eligible:
                selected = latest_p5(rows[target], decision, target, delay=delay)
                if selected is None:
                    continue
                actual.append(prices[target])
                predicted.append(selected.price_aud_per_kwh)
                persistence.append(prices[decision - timedelta(minutes=5)])
                ages.append((decision - selected.run_utc - delay).total_seconds() / 60)
                run_ages.append((decision - selected.run_utc).total_seconds() / 60)
                source_leads.append((target - selected.run_utc).total_seconds() / 60)
            by_horizon[str(horizon)] = {
                "horizon_minutes": horizon,
                "eligible_origin_target_pairs": len(eligible),
                "matched_observations": len(actual),
                "coverage": len(actual) / len(eligible) if eligible else None,
                "forecast_age_since_assumed_availability_minutes": distribution(ages),
                "forecast_age_since_nominal_run_minutes": distribution(run_ages),
                "target_lead_from_nominal_run_minutes": distribution(source_leads),
                "scores": scores(actual, predicted, persistence),
            }
        output[str(delay_minutes)] = by_horizon
    return output


def pd_diagnostic(prices):
    starts = sorted(prices)
    first = starts[0]
    last = starts[-1] + timedelta(minutes=30)
    by_target = defaultdict(list)
    for row in read_pd("PREDISPATCHPRICE", target_start=first, target_end=last):
        if row.price_aud_per_kwh is not None:
            by_target[row.target_start_utc].append(PredispatchForecast(
                row.table, row.sequence, row.nominal_run_utc, row.lastchanged_utc,
                row.target_start_utc, row.price_aud_per_kwh, row.intervention,
                row.runno, {}))
    region_keys = set()
    region_fields_present = set()
    for row in read_pd("PREDISPATCHREGIONSUM", target_start=first, target_end=last):
        key = (row.sequence, row.target_start_utc, row.intervention, row.runno)
        if key in region_keys:
            raise ValueError("Duplicate regional predispatch vintage")
        region_keys.add(key)
        for name in ("TOTALDEMAND", "AVAILABLEGENERATION", "SS_SOLAR_AVAILABILITY",
                     "SS_WIND_AVAILABILITY", "NETINTERCHANGE"):
            if row.fields.get(name) not in (None, ""):
                region_fields_present.add(name)
    origins = [t for t in starts if t.minute in (0, 30)
               and t - timedelta(minutes=5) in prices]
    assumptions = (("lastchanged_plus_0", 0, False),
                   ("lastchanged_plus_5", 5, False),
                   ("lastchanged_plus_10", 10, False),
                   ("lastchanged_plus_30", 30, False),
                   ("lastchanged_plus_10_and_nominal_elapsed", 10, True))
    output = {}
    for name, minutes, require_nominal in assumptions:
        results = {}
        for hours in PD_HOURS:
            offset = timedelta(hours=hours)
            eligible = [(t, t + offset, half_hour_average(prices, t + offset))
                        for t in origins]
            eligible = [(t, target, actual) for t, target, actual in eligible
                        if actual is not None]
            actual, predicted, persistence = [], [], []
            ages, nominal_ages, source_leads = [], [], []
            joined_region = 0
            for decision, target, observed in eligible:
                selected = latest_pd(by_target[target], decision, target,
                                     delay=timedelta(minutes=minutes),
                                     require_nominal_run_elapsed=require_nominal)
                if selected is None:
                    continue
                actual.append(observed)
                predicted.append(selected.price_aud_per_kwh)
                persistence.append(prices[decision - timedelta(minutes=5)])
                ages.append((decision - selected.lastchanged_utc).total_seconds()/60 - minutes)
                nominal_ages.append((decision - selected.nominal_run_utc).total_seconds()/60)
                source_leads.append((target - selected.nominal_run_utc).total_seconds()/60)
                key = (selected.sequence, target, selected.intervention, selected.runno)
                joined_region += key in region_keys
            results[str(hours)] = {
                "horizon_hours": hours,
                "eligible_origin_target_pairs": len(eligible),
                "matched_observations": len(actual),
                "coverage": len(actual)/len(eligible) if eligible else None,
                "matched_regional_summary_rows": joined_region,
                "forecast_age_since_lastchanged_plus_delay_minutes": distribution(ages),
                "forecast_age_since_nominal_run_minutes": distribution(nominal_ages),
                "target_lead_from_nominal_run_minutes": distribution(source_leads),
                "scores": scores(actual, predicted, persistence),
            }
        output[name] = results
    return {"price_rows_with_target_in_january_window": sum(map(len, by_target.values())),
            "regional_vintage_keys_with_target_in_january_window": len(region_keys),
            "assumptions": output,
            "regional_fields_present": sorted(region_fields_present)}


def run():
    prices = realised_prices()
    result = {
        "dataset": "Sydney-local January 2025 exploratory development; not independent evaluation",
        "realised_five_minute_intervals": len(prices),
        "p5min": {"availability_proxy": "RUN_DATETIME + delay <= decision",
                  "target": "exact five-minute interval starting at decision + horizon",
                  "by_delay_minutes": p5_diagnostic(prices)},
        "predispatch": {"availability_proxy": "LASTCHANGED + delay <= decision; optional nominal-run-elapsed guard; neither is verified publication",
                        "target": "arithmetic mean of six realised five-minute NSW1 spot prices in forecast's 30-minute interval",
                        "by_horizon": pd_diagnostic(prices)},
    }
    _write_json_atomic(OUT, result)
    return result


if __name__ == "__main__":
    result = run()
    for delay, horizons in result["p5min"]["by_delay_minutes"].items():
        print("P5 delay", delay, "coverage", {h: horizons[str(h)]["matched_observations"]
                                             for h in P5_HORIZONS})
    for assumption, horizons in result["predispatch"]["by_horizon"]["assumptions"].items():
        print("PD", assumption, "coverage", {h: horizons[str(h)]["matched_observations"]
                                            for h in PD_HOURS})
