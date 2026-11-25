"""Download immutable monthly source snapshots and build realised five-minute data.

Default smoke test: python -m solaralpha.historical_pipeline
Plan without network: python -m solaralpha.historical_pipeline --plan
The default processes only Sydney-local January 2025. A wider date range must
be supplied explicitly; no trading agent imports this module.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from random import Random
from statistics import mean, median
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from solaralpha.historical_replay import (LOAD_SEED, SOLAR_DERATE,
                                           modeled_load_kw)
from solaralpha.fetch_historical_sample import WEATHER_FILE as LEGACY_WEATHER_FILE
from solaralpha.fetch_historical_sample import WEATHER_URL as LEGACY_WEATHER_URL
from solaralpha.solar_sim import Config
from solaralpha.tariffs import ILLUSTRATIVE_SPOT_LINKED


ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MANIFEST = RAW / "source_manifest.json"
STEP = timedelta(minutes=5)
AEST = timezone(timedelta(hours=10), "AEST")
UTC = timezone.utc
SOURCE_MODEL = "era5"
REGION = "NSW1"
TARGET_START = "2023-01"
TARGET_END = "2026-02"
DEV_START = date(2025, 1, 13)
DEV_END = date(2025, 1, 19)
MAX_BYTES = 10_000_000
CSV_FIELDS = (
    "interval_start_utc", "interval_end_utc", "interval_start_sydney",
    "local_date", "local_hour", "local_minute", "local_weekday",
    "utc_offset_minutes", "is_dst", "local_fold", "partition", "is_development_week",
    "include_in_validation_score", "realised_aemo_rrp_aud_per_mwh",
    "realised_spot_aud_per_kwh", "realised_regional_demand_mw",
    "realised_temperature_c", "realised_shortwave_radiation_w_m2",
    "realised_cloud_cover_pct", "realised_solar_proxy_kw",
    "realised_modeled_household_load_kw", "realised_illustrative_buy_aud_per_kwh",
    "realised_illustrative_sell_aud_per_kwh",
)


@dataclass(frozen=True)
class Source:
    path: Path
    url: str
    kind: str


def sydney_zone():
    try:
        return ZoneInfo("Australia/Sydney")
    except ZoneInfoNotFoundError as exc:
        raise RuntimeError("Install requirements-data.txt for Australia/Sydney time-zone rules") from exc


def month_date(text):
    try:
        parsed = datetime.strptime(text, "%Y-%m")
    except ValueError as exc:
        raise ValueError("Month must be YYYY-MM") from exc
    if parsed.strftime("%Y-%m") != text:
        raise ValueError("Month must be YYYY-MM")
    return date(parsed.year, parsed.month, 1)


def next_month(first):
    return date(first.year + (first.month == 12), first.month % 12 + 1, 1)


def previous_month(first):
    return date(first.year - (first.month == 1), (first.month - 2) % 12 + 1, 1)


def months_between(start, end):
    if not month_date(TARGET_START) <= start <= end <= month_date(TARGET_END):
        raise ValueError(f"Months must lie within {TARGET_START}..{TARGET_END}")
    months = []
    current = start
    while current <= end:
        months.append(current)
        current = next_month(current)
    return months


def local_month_bounds(month):
    zone = sydney_zone()
    start = datetime(month.year, month.month, 1, tzinfo=zone)
    end_day = next_month(month)
    end = datetime(end_day.year, end_day.month, 1, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


def source_plan(months):
    """Return deduplicated source files needed for Sydney-local months."""
    sources = {}
    for month in months:
        # During AEDT, early local hours belong to the previous AEST month.
        for market_month in (previous_month(month), month):
            stamp = market_month.strftime("%Y%m")
            filename = f"PRICE_AND_DEMAND_{stamp}_{REGION}.CSV"
            url = ("https://www.nemweb.com.au/Reports/CURRENT/WebData/NEM/"
                   f"PRICEANDDEMAND/{filename}")
            path = RAW / filename
            sources[path] = Source(path, url, "aemo_price_and_demand")
        start_utc, end_utc = local_month_bounds(month)
        params = {
            "latitude": "-33.87", "longitude": "151.21",
            "start_date": start_utc.date().isoformat(),
            "end_date": end_utc.date().isoformat(),
            "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
            "timezone": "GMT", "models": SOURCE_MODEL,
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)
        path = RAW / f"open_meteo_{SOURCE_MODEL}_sydney_local_{month:%Y%m}.json"
        sources[path] = Source(path, url, "open_meteo_reanalysis")
    return list(sources.values())


def _validate_source_bytes(source, content):
    if not content or len(content) > MAX_BYTES:
        raise ValueError(f"Empty or oversized source: {source.path.name}")
    if source.kind == "aemo_price_and_demand":
        header = content.lstrip(b"\xef\xbb\xbf").splitlines()[0].decode("ascii")
        if header != "REGION,SETTLEMENTDATE,TOTALDEMAND,RRP,PERIODTYPE":
            raise ValueError(f"Unexpected AEMO header: {source.path.name}")
    elif source.kind == "open_meteo_reanalysis":
        payload = json.loads(content)
        if "hourly" not in payload or "hourly_units" not in payload:
            raise ValueError(f"Missing weather arrays: {source.path.name}")


def _write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def ensure_sources(sources, manifest_path=MANIFEST, opener=urlopen):
    """Never overwrite raw bytes; verify cached files against their hashes."""
    manifest = ({"schema_version": 1, "files": {}}
                if not manifest_path.exists() else
                json.loads(manifest_path.read_text(encoding="utf-8")))
    for source in sources:
        key = source.path.relative_to(ROOT).as_posix()
        previous = manifest["files"].get(key)
        if previous is not None and previous["url"] != source.url:
            raise ValueError(f"Source URL changed for {key}")
        cached = source.path.exists()
        if cached:
            content = source.path.read_bytes()
            headers = {}
            origin = "preexisting_local" if previous is None else previous["origin"]
        else:
            request = Request(source.url, headers={"User-Agent": "SolarAlpha historical pipeline/1.0"})
            with opener(request, timeout=60) as response:
                content = response.read(MAX_BYTES + 1)
                headers = {name: response.headers.get(name)
                           for name in ("Last-Modified", "ETag", "Content-Type")}
            origin = "downloaded"
        _validate_source_bytes(source, content)
        digest = sha256(content).hexdigest()
        if previous is not None and previous["sha256"] != digest:
            raise ValueError(f"Raw source differs from manifest: {key}")
        if not cached:
            source.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = source.path.with_suffix(source.path.suffix + ".tmp")
            temporary.write_bytes(content)
            temporary.replace(source.path)
        if previous is None:
            manifest["files"][key] = {
                "url": source.url, "kind": source.kind, "sha256": digest,
                "bytes": len(content), "origin": origin,
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "retrieved_at_utc": (datetime.now(UTC).isoformat()
                                     if origin == "downloaded" else None),
                "response_headers": headers,
            }
            _write_json_atomic(manifest_path, manifest)
        print(f"{'cached' if cached else 'downloaded'} ({origin}): {key} "
              f"({len(content)} bytes; sha256 {digest})", flush=True)
    return manifest


def split_for_local_date(day):
    if date(2023, 1, 1) <= day <= date(2024, 12, 31):
        return "train"
    if date(2025, 1, 1) <= day <= date(2025, 11, 30):
        return "validation"
    if date(2025, 12, 1) <= day <= date(2026, 2, 28):
        return "held_out_test"
    raise ValueError(f"Date outside dataset plan: {day}")


def _aemo_rows(paths, start_utc, end_utc):
    values, duplicates, invalid = {}, [], []
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != (
                    "REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE"):
                raise ValueError(f"Unexpected AEMO columns in {path}")
            for line, row in enumerate(reader, 2):
                if row["REGION"] != REGION or row["PERIODTYPE"] != "TRADE":
                    continue
                try:
                    end = datetime.strptime(row["SETTLEMENTDATE"],
                                            "%Y/%m/%d %H:%M:%S").replace(tzinfo=AEST)
                    instant = end.astimezone(UTC) - STEP
                except ValueError:
                    invalid.append(f"{path.name}:{line}: timestamp")
                    continue
                if not start_utc <= instant < end_utc:
                    continue
                if instant in values:
                    duplicates.append(instant.isoformat())
                    continue
                try:
                    price = float(row["RRP"])
                    demand = float(row["TOTALDEMAND"])
                    if (not isfinite(price) or not isfinite(demand)
                            or not 100 <= demand <= 50_000):
                        raise ValueError
                except ValueError:
                    invalid.append(f"{path.name}:{line}: price/demand")
                    continue
                values[instant] = (price, demand)
    return values, duplicates, invalid


def _weather_rows(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    units = payload.get("hourly_units", {})
    if (payload.get("timezone") != "GMT" or payload.get("utc_offset_seconds") != 0
            or units.get("shortwave_radiation") != "W/m²"
            or units.get("temperature_2m") != "°C"
            or units.get("cloud_cover") != "%"):
        raise ValueError("Weather timezone or units differ from expected UTC/W/m²/°C/%")
    hourly = payload["hourly"]
    names = ("time", "temperature_2m", "shortwave_radiation", "cloud_cover")
    if len({len(hourly[name]) for name in names}) != 1:
        raise ValueError("Weather arrays have unequal lengths")
    values, duplicates, invalid = {}, [], []
    for index, stamp in enumerate(hourly["time"]):
        try:
            instant = datetime.fromisoformat(stamp).replace(tzinfo=UTC)
        except ValueError:
            invalid.append(f"weather row {index}: timestamp")
            continue
        if instant in values:
            duplicates.append(instant.isoformat())
            continue
        sample = [hourly[name][index] for name in names[1:]]
        if (any(value is None or not isinstance(value, (int, float))
                or not isfinite(value) for value in sample)
                or not -50 <= sample[0] <= 60
                or not 0 <= sample[1] <= 1500
                or not 0 <= sample[2] <= 100):
            invalid.append(f"weather row {index}: value")
            continue
        values[instant] = tuple(float(value) for value in sample)
    return values, duplicates, invalid


def _iso_utc(instant):
    return instant.isoformat().replace("+00:00", "Z")


def build_month(month, *, raw_dir=RAW, processed_dir=PROCESSED, config=None):
    """Write a strict local-calendar month or a quality report if incomplete."""
    config = Config() if config is None else config
    start_utc, end_utc = local_month_bounds(month)
    market_months = (previous_month(month), month)
    aemo_paths = [raw_dir / f"PRICE_AND_DEMAND_{part:%Y%m}_{REGION}.CSV"
                  for part in market_months]
    weather_path = raw_dir / f"open_meteo_{SOURCE_MODEL}_sydney_local_{month:%Y%m}.json"
    prices, price_duplicates, price_invalid = _aemo_rows(aemo_paths, start_utc, end_utc)
    weather, weather_duplicates, weather_invalid = _weather_rows(weather_path)
    count = int((end_utc - start_utc) / STEP)
    zone = sydney_zone()
    rng = Random(LOAD_SEED)
    rows = []
    missing_price, missing_weather = [], []
    for index in range(count):
        instant = start_utc + index * STEP
        local = instant.astimezone(zone)
        hour_start = instant.replace(minute=0, second=0, microsecond=0)
        hour_end = hour_start + timedelta(hours=1)
        price = prices.get(instant)
        conditions = weather.get(hour_start)
        radiation = weather.get(hour_end)
        if price is None:
            missing_price.append(_iso_utc(instant))
        if conditions is None or radiation is None:
            missing_weather.append(_iso_utc(instant))
        day = local.date()
        dev = DEV_START <= day <= DEV_END
        partition = split_for_local_date(day)
        noise = rng.gauss(0.0, 0.10)
        row = {
            "interval_start_utc": _iso_utc(instant),
            "interval_end_utc": _iso_utc(instant + STEP),
            "interval_start_sydney": local.isoformat(),
            "local_date": day.isoformat(), "local_hour": local.hour,
            "local_minute": local.minute, "local_weekday": local.weekday(),
            "utc_offset_minutes": int(local.utcoffset().total_seconds() / 60),
            "is_dst": int(local.dst() != timedelta(0)),
            "local_fold": local.fold,
            "partition": partition, "is_development_week": int(dev),
            "include_in_validation_score": int(partition == "validation" and not dev),
            "realised_aemo_rrp_aud_per_mwh": price[0] if price else None,
            "realised_spot_aud_per_kwh": price[0] / 1000 if price else None,
            "realised_regional_demand_mw": price[1] if price else None,
            "realised_temperature_c": conditions[0] if conditions else None,
            "realised_shortwave_radiation_w_m2": radiation[1] if radiation else None,
            "realised_cloud_cover_pct": conditions[2] if conditions else None,
            "realised_solar_proxy_kw": (min(config.solar_kw, config.solar_kw
                                            * SOLAR_DERATE * radiation[1] / 1000)
                                          if radiation else None),
            "realised_modeled_household_load_kw": (
                modeled_load_kw(local, conditions[0], noise) if conditions else None),
            "realised_illustrative_buy_aud_per_kwh": (
                ILLUSTRATIVE_SPOT_LINKED.prices(price[0] / 1000)[0]
                if price else None),
            "realised_illustrative_sell_aud_per_kwh": (
                ILLUSTRATIVE_SPOT_LINKED.prices(price[0] / 1000)[1]
                if price else None),
        }
        rows.append(row)
    report = {
        "month": month.strftime("%Y-%m"), "expected_intervals": count,
        "first_interval_utc": _iso_utc(start_utc),
        "last_interval_utc": _iso_utc(end_utc - STEP),
        "present_aemo_intervals": len(prices),
        "missing_aemo_intervals": len(missing_price),
        "missing_weather_intervals": len(missing_weather),
        "aemo_duplicate_intervals": len(price_duplicates),
        "weather_duplicate_hours": len(weather_duplicates),
        "invalid_aemo_rows": len(price_invalid),
        "invalid_weather_rows": len(weather_invalid),
        "examples": {"missing_aemo": missing_price[:20],
                     "missing_weather": missing_weather[:20],
                     "aemo_duplicates": price_duplicates[:20],
                     "weather_duplicates": weather_duplicates[:20],
                     "invalid_aemo": price_invalid[:20],
                     "invalid_weather": weather_invalid[:20]},
    }
    processed_dir.mkdir(parents=True, exist_ok=True)
    quality_path = processed_dir / f"{month:%Y-%m}_quality.json"
    _write_json_atomic(quality_path, report)
    defects = sum(report[key] for key in (
        "missing_aemo_intervals", "missing_weather_intervals",
        "aemo_duplicate_intervals", "weather_duplicate_hours",
        "invalid_aemo_rows", "invalid_weather_rows"))
    if defects:
        raise ValueError(f"Incomplete/invalid {month:%Y-%m} inputs; see {quality_path}")
    output_path = processed_dir / f"{month:%Y-%m}_realised.csv"
    temporary = output_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    return rows, report, output_path


def summarize_month(month, rows, processed_dir=PROCESSED):
    prices = [float(row["realised_spot_aud_per_kwh"]) for row in rows]
    demands = [float(row["realised_regional_demand_mw"]) for row in rows]
    temps = [float(row["realised_temperature_c"]) for row in rows]
    radiation = [float(row["realised_shortwave_radiation_w_m2"]) for row in rows]
    cloud = [float(row["realised_cloud_cover_pct"]) for row in rows]
    ordered = sorted(prices)
    summary = {
        "schema_version": 1,
        "month": month.strftime("%Y-%m"), "intervals": len(rows),
        "derived_csv_sha256": sha256((processed_dir / f"{month:%Y-%m}_realised.csv")
                                      .read_bytes()).hexdigest(),
        "derivation": {"weather_model": SOURCE_MODEL,
                       "solar_derate": SOLAR_DERATE,
                       "modeled_load_seed": LOAD_SEED,
                       "illustrative_tariff": ILLUSTRATIVE_SPOT_LINKED.name},
        "partition_counts": {part: sum(row["partition"] == part for row in rows)
                             for part in ("train", "validation", "held_out_test")},
        "development_week_intervals": sum(row["is_development_week"] for row in rows),
        "validation_score_eligible_intervals": sum(
            row["include_in_validation_score"] for row in rows),
        "spot_aud_per_kwh": {
            "min": min(prices), "median": median(prices),
            "mean": mean(prices), "p95": ordered[int(.95 * (len(ordered) - 1))],
            "p99": ordered[int(.99 * (len(ordered) - 1))], "max": max(prices),
            "negative_intervals": sum(price < 0 for price in prices),
            "negative_frequency": sum(price < 0 for price in prices) / len(prices),
            "above_1_aud_per_kwh_intervals": sum(price > 1 for price in prices),
        },
        "price_extreme_events": {
            "highest": [{"interval_start_utc": row["interval_start_utc"],
                         "interval_start_sydney": row["interval_start_sydney"],
                         "aud_per_kwh": row["realised_spot_aud_per_kwh"]}
                        for row in sorted(rows,
                                          key=lambda item: float(item["realised_spot_aud_per_kwh"]),
                                          reverse=True)[:10]],
            "lowest": [{"interval_start_utc": row["interval_start_utc"],
                        "interval_start_sydney": row["interval_start_sydney"],
                        "aud_per_kwh": row["realised_spot_aud_per_kwh"]}
                       for row in sorted(rows,
                                         key=lambda item: float(item["realised_spot_aud_per_kwh"]))[:10]],
        },
        "regional_demand_mw": {"min": min(demands), "mean": mean(demands),
                               "max": max(demands)},
        "temperature_c": {"min": min(temps), "mean": mean(temps), "max": max(temps)},
        "shortwave_radiation_w_m2": {"min": min(radiation),
                                      "mean": mean(radiation), "max": max(radiation)},
        "cloud_cover_pct": {"min": min(cloud), "mean": mean(cloud), "max": max(cloud)},
    }
    path = processed_dir / f"{month:%Y-%m}_summary.json"
    _write_json_atomic(path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-month", default="2025-01")
    parser.add_argument("--end-month", default="2025-01")
    parser.add_argument("--plan", action="store_true",
                        help="show files needed without network or processing")
    parser.add_argument("--process-only", action="store_true",
                        help="require cached raw files; never make network requests")
    args = parser.parse_args()
    months = months_between(month_date(args.start_month), month_date(args.end_month))
    sources = source_plan(months)
    if args.plan:
        for source in sources:
            print(f"{source.path.relative_to(ROOT)} <- {source.url}")
        return
    if args.process_only:
        absent = [str(source.path) for source in sources if not source.path.exists()]
        if absent:
            raise FileNotFoundError("Missing cached raw files: " + ", ".join(absent))
    # Even process-only verifies hashes/manifest before reading cached bytes.
    ensure_sources(sources)
    if LEGACY_WEATHER_FILE.exists():
        ensure_sources([Source(LEGACY_WEATHER_FILE, LEGACY_WEATHER_URL,
                               "open_meteo_reanalysis")])
    for month in months:
        rows, report, path = build_month(month)
        summary = summarize_month(month, rows)
        print(f"Processed {month:%Y-%m}: {len(rows)} intervals, "
              f"{summary['spot_aud_per_kwh']['negative_intervals']} negative prices, "
              f"{summary['development_week_intervals']} development intervals; "
              f"{path}", flush=True)


if __name__ == "__main__":
    main()
