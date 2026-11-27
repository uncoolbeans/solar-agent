"""January 20-21, 2025 AEMO forecast availability audit (offline only)."""

import csv
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from statistics import mean
from zipfile import ZipFile

from solaralpha.aemo_forecast_archive import (TABLES, MANIFEST, ROOT, UTC,
                                              latest_available, read_forecasts, source_path)
from solaralpha.historical_pipeline import _write_json_atomic


SAMPLE_START = datetime(2025, 1, 19, 13, tzinfo=UTC)  # Jan 20 00:00 AEDT
SAMPLE_END = datetime(2025, 1, 21, 13, tzinfo=UTC)    # Jan 22 00:00 AEDT
OUT = ROOT / "data" / "derived" / "aemo_forecast_2025_01_smoke.json"
SAMPLE_CSV = ROOT / "data" / "derived" / "aemo_p5min_nsw1_2025-01-20_21.csv.gz"


def snapshot_summary(table):
    """Describe monthly 30-minute tables without treating snapshots as vintages."""
    with ZipFile(source_path(table)) as archive:
        member = archive.namelist()[0]
        uncompressed_bytes = archive.getinfo(member).file_size
        with archive.open(member) as file:
            reader = csv.reader(io.TextIOWrapper(file, encoding="utf-8-sig", newline=""))
            columns = None
            targets = []
            sequences = set()
            interventions = Counter()
            all_rows = 0
            for row in reader:
                if row[0] == "I" and row[1] == "PREDISPATCH":
                    columns = row[4:]
                elif row[0] == "D" and row[1] == "PREDISPATCH":
                    if columns is None:
                        raise ValueError("Missing predispatch header")
                    values = dict(zip(columns, row[4:]))
                    if values["REGIONID"] != "NSW1":
                        continue
                    all_rows += 1
                    targets.append(values["DATETIME"])
                    sequences.add(values["PREDISPATCHSEQNO"])
                    interventions[values["INTERVENTION"]] += 1
    return {"nsw_rows": all_rows, "unique_target_interval_ends": len(set(targets)),
            "unique_sequence_numbers": len(sequences),
            "intervention_flags": dict(interventions),
            "columns": columns, "uncompressed_bytes": uncompressed_bytes,
            "vintage_usable": False}


def run():
    if not MANIFEST.exists():
        raise FileNotFoundError("Download the three January 2025 table archives first")
    records = list(read_forecasts("P5MIN_REGIONSOLUTION", start=SAMPLE_START,
                                  end=SAMPLE_END + timedelta(hours=1)))
    SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(SAMPLE_CSV, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "run_utc", "target_interval_start_utc", "rrp_aud_per_kwh",
            "intervention", "lastchanged_aest", "totaldemand_mw",
            "availablegeneration_mw", "demandforecast_mw",
            "netinterchange_mw", "ss_solar_availability_mw",
            "ss_wind_availability_mw"])
        writer.writeheader()
        for f in sorted(records, key=lambda f: (f.run_utc, f.target_start_utc)):
            writer.writerow({
                "run_utc": f.run_utc.isoformat(),
                "target_interval_start_utc": f.target_start_utc.isoformat(),
                "rrp_aud_per_kwh": f.price_aud_per_kwh,
                "intervention": f.intervention,
                "lastchanged_aest": f.fields.get("LASTCHANGED"),
                "totaldemand_mw": f.fields.get("TOTALDEMAND"),
                "availablegeneration_mw": f.fields.get("AVAILABLEGENERATION"),
                "demandforecast_mw": f.fields.get("DEMANDFORECAST"),
                "netinterchange_mw": f.fields.get("NETINTERCHANGE"),
                "ss_solar_availability_mw": f.fields.get("SS_SOLAR_AVAILABILITY"),
                "ss_wind_availability_mw": f.fields.get("SS_WIND_AVAILABILITY"),
            })
    by_target = defaultdict(list)
    for forecast in records:
        by_target[forecast.target_start_utc].append(forecast)
    decisions = [SAMPLE_START + timedelta(minutes=5 * i) for i in range(576)]
    coverage = {}
    for delay_minutes in (0, 1, 5, 10):
        by_horizon = {}
        max_leads = []
        for horizon in range(5, 65, 5):
            count = 0
            for decision in decisions:
                target = decision + timedelta(minutes=horizon)
                selected = latest_available(by_target[target], decision, target,
                                            delay=timedelta(minutes=delay_minutes))
                count += selected is not None
            by_horizon[str(horizon)] = count
        for decision in decisions:
            leads = [(f.target_start_utc - decision).total_seconds() / 60
                     for target, group in by_target.items() for f in group
                     if f.run_utc + timedelta(minutes=delay_minutes) <= decision
                     and f.intervention == 0 and target > decision]
            max_leads.append(max(leads, default=None))
        coverage[str(delay_minutes)] = {
            "eligible_decisions_by_lead_minutes": by_horizon,
            "max_available_lead_minutes": dict(Counter(max_leads)),
        }
    p5 = {
        "nsw_forecast_rows_with_target_in_window_plus_one_hour": len(records),
        "unique_runs": len(set(f.run_utc for f in records)),
        "intervention_flags": dict(Counter(f.intervention for f in records)),
        "min_target_start_minus_run_minutes": min((f.target_start_utc - f.run_utc).total_seconds()/60 for f in records),
        "max_target_start_minus_run_minutes": max((f.target_start_utc - f.run_utc).total_seconds()/60 for f in records),
        "coverage_by_assumed_run_plus_delay_minutes": coverage,
    }
    with (ROOT / "data" / "processed" / "2025-01_realised.csv").open(
            encoding="utf-8", newline="") as file:
        realised = {datetime.fromisoformat(row["interval_start_utc"]):
                    float(row["realised_spot_aud_per_kwh"])
                    for row in csv.DictReader(file)}
    first_period_differences = [abs(f.price_aud_per_kwh - realised[f.target_start_utc])
                                for f in records
                                if f.run_utc == f.target_start_utc + timedelta(minutes=5)
                                and f.target_start_utc in realised]
    p5["first_period_vs_revised_realised"] = {
        "matched_rows": len(first_period_differences),
        "mean_absolute_difference_aud_per_kwh": mean(first_period_differences),
        "max_absolute_difference_aud_per_kwh": max(first_period_differences),
        "note": "Reconciliation check, not a forecast score; publication/revision differences unresolved.",
    }
    result = {
        "sample_origin_sydney_dates": ["2025-01-20", "2025-01-21"],
        "decision_origins": len(decisions), "target_offset_minutes": 60,
        "publication_timestamp_status": "not established in monthly archive; run plus delay is a proxy",
        "aemo_one_hour_price_metrics": None,
        "aemo_one_hour_metric_reason": "No P5MIN vintage covers the exact five-minute target interval at any decision origin; monthly predispatch tables do not retain usable vintages or five-minute resolution.",
        "matched_baseline_and_model_metrics": None,
        "matched_metric_reason": "Zero eligible AEMO one-hour rows, so no identical-row comparison exists.",
        "p5min": p5,
        "predispatchprice": snapshot_summary("PREDISPATCHPRICE"),
        "predispatchregionsum": snapshot_summary("PREDISPATCHREGIONSUM"),
        "raw_manifest": MANIFEST.relative_to(ROOT).as_posix(),
        "normalized_p5min_sample": SAMPLE_CSV.relative_to(ROOT).as_posix(),
    }
    _write_json_atomic(OUT, result)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"sample_decisions": result["decision_origins"],
                      "one_hour_coverage": result["p5min"]["coverage_by_assumed_run_plus_delay_minutes"],
                      "predispatch_rows": result["predispatchprice"]["nsw_rows"]}, indent=2))
