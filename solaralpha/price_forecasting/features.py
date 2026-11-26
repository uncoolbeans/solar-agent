"""Decision-time-safe five-minute features for a one-hour-ahead price target."""

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
from math import cos, pi, sin, sqrt
from pathlib import Path
from collections import deque, Counter

from solaralpha.historical_pipeline import (
    AEST, DEV_END, DEV_START, MANIFEST, PROCESSED, RAW, ROOT, STEP,
    _aemo_rows, _write_json_atomic, local_month_bounds, month_date,
    months_between, sydney_zone,
)


FEATURE_DIR = ROOT / "data" / "features"
HORIZON_STEPS = 12
WARMUP_STEPS = 288
LAG_MINUTES = (5, 15, 30, 60, 180, 360, 720, 1440)
DEMAND_LAG_MINUTES = (5, 15, 60, 1440)
ROLL_MINUTES = (60, 360, 1440)
WEEKLY_TARGET_LAG_STEPS = 7 * 24 * 12
DAILY_TARGET_LAG_STEPS = 24 * 12
TRAIN_START = date(2023, 1, 1)
VALIDATION_END = date(2025, 11, 30)
TARGET_FIELD = "target_spot_aud_per_kwh"
METADATA_FIELDS = (
    "decision_start_utc", "target_interval_start_utc", "origin_local_date",
    "origin_partition", "target_partition", "is_development_origin",
    "is_development_target", "include_in_validation_score",
)
CALENDAR_FIELDS = (
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "local_month", "local_day_of_month", "local_day_of_year", "is_weekend",
    "is_dst", "local_fold",
)
LAG_FIELDS = tuple(f"price_lag_{minutes}m_aud_per_kwh" for minutes in LAG_MINUTES)
ROLL_FIELDS = tuple(
    f"price_trailing_{minutes}m_{stat}_aud_per_kwh"
    for minutes in ROLL_MINUTES for stat in ("mean", "std", "min", "max")
)
CHANGE_FIELDS = (
    "price_change_5m_vs_15m_aud_per_kwh",
    "price_change_5m_vs_60m_aud_per_kwh",
    "price_change_5m_vs_1440m_aud_per_kwh",
)
DEMAND_FIELDS = (
    tuple(f"demand_lag_{minutes}m_mw" for minutes in DEMAND_LAG_MINUTES)
    + ("demand_change_5m_vs_15m_mw", "demand_change_5m_vs_60m_mw")
)
SEASONAL_FIELDS = ("price_target_aligned_previous_day_aud_per_kwh",
                   "price_target_aligned_previous_week_aud_per_kwh")
FEATURE_FIELDS = (CALENDAR_FIELDS + LAG_FIELDS + ROLL_FIELDS + CHANGE_FIELDS
                  + DEMAND_FIELDS + SEASONAL_FIELDS)
OUTPUT_FIELDS = METADATA_FIELDS + FEATURE_FIELDS + (TARGET_FIELD,)


@dataclass(frozen=True, slots=True)
class PricePoint:
    time_utc: datetime
    local_time: datetime
    price_aud_per_kwh: float
    regional_demand_mw: float
    partition: str
    development_week: bool
    local_fold: int = 0


def _parse_utc(text):
    instant = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if instant.utcoffset() != timedelta(0):
        raise ValueError("Expected UTC timestamp")
    return instant.astimezone(timezone.utc)


def _assert_month_integrity(path):
    quality = json.loads(path.with_name(path.name.replace("_realised.csv", "_quality.json"))
                         .read_text(encoding="utf-8"))
    for field in ("missing_aemo_intervals", "missing_weather_intervals",
                  "aemo_duplicate_intervals", "weather_duplicate_hours",
                  "invalid_aemo_rows", "invalid_weather_rows"):
        if quality[field]:
            raise ValueError(f"Source quality failed for {path.name}: {field}")
    summary = json.loads(path.with_name(path.name.replace("_realised.csv", "_summary.json"))
                         .read_text(encoding="utf-8"))
    if sha256(path.read_bytes()).hexdigest() != summary["derived_csv_sha256"]:
        raise ValueError(f"Processed checksum mismatch: {path.name}")


def load_training_validation_points(processed_dir=PROCESSED, raw_dir=RAW):
    """Load only 2023-01..2025-11 plus one local day of *feature-only* warm-up."""
    first_local = datetime(2022, 12, 31, tzinfo=sydney_zone())
    end_local = datetime(2023, 1, 1, tzinfo=sydney_zone())
    start_utc, end_utc = first_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
    raw_path = raw_dir / "PRICE_AND_DEMAND_202212_NSW1.CSV"
    manifest_path = raw_dir / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_key = raw_path.relative_to(ROOT).as_posix()
    if sha256(raw_path.read_bytes()).hexdigest() != manifest["files"][raw_key]["sha256"]:
        raise ValueError("Warm-up source checksum mismatch")
    warmup, duplicates, invalid = _aemo_rows([raw_path], start_utc, end_utc)
    if duplicates or invalid or len(warmup) != WARMUP_STEPS:
        raise ValueError("One-day warm-up price/demand coverage is incomplete")
    zone = sydney_zone()
    points = [PricePoint(
        time_utc=start_utc + i * STEP,
        local_time=(start_utc + i * STEP).astimezone(zone),
        price_aud_per_kwh=warmup[start_utc + i * STEP][0] / 1000,
        regional_demand_mw=warmup[start_utc + i * STEP][1],
        partition="warmup", development_week=False,
        local_fold=(start_utc + i * STEP).astimezone(zone).fold,
    ) for i in range(WARMUP_STEPS)]
    months = months_between(month_date("2023-01"), month_date("2025-11"))
    expected_total = WARMUP_STEPS
    for month in months:
        path = processed_dir / f"{month:%Y-%m}_realised.csv"
        _assert_month_integrity(path)
        start_month_utc, end_month_utc = local_month_bounds(month)
        expected = int((end_month_utc - start_month_utc) / STEP)
        count = 0
        with path.open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                instant = _parse_utc(row["interval_start_utc"])
                local = instant.astimezone(zone)
                if instant != start_month_utc + count * STEP:
                    raise ValueError(f"Out-of-order or missing UTC interval: {path.name}")
                if row["local_date"] != local.date().isoformat():
                    raise ValueError(f"Sydney date mismatch: {path.name}")
                if row["local_fold"] != str(local.fold):
                    raise ValueError(f"Sydney fold mismatch: {path.name}")
                if row["partition"] not in ("train", "validation"):
                    raise ValueError("Held-out rows are forbidden in the feature loader")
                points.append(PricePoint(
                    instant, local, float(row["realised_spot_aud_per_kwh"]),
                    float(row["realised_regional_demand_mw"]),
                    row["partition"], row["is_development_week"] == "1",
                    int(row["local_fold"]),
                ))
                count += 1
        if count != expected:
            raise ValueError(f"Wrong interval count for {path.name}")
        expected_total += count
    if len(points) != expected_total:
        raise ValueError("Point count mismatch")
    for before, after in zip(points, points[1:]):
        if after.time_utc - before.time_utc != STEP:
            raise ValueError("Cross-month UTC gap or duplicate")
    return points


def trailing_stats(values, width):
    """Mean, population std, min, max over [i-width, i), O(n)."""
    result = [None] * (len(values) + 1)
    minimum, maximum = deque(), deque()
    running = squared = 0.0
    for j, value in enumerate(values):
        running += value
        squared += value * value
        while minimum and values[minimum[-1]] >= value:
            minimum.pop()
        while maximum and values[maximum[-1]] <= value:
            maximum.pop()
        minimum.append(j)
        maximum.append(j)
        if j >= width:
            removed = values[j - width]
            running -= removed
            squared -= removed * removed
        while minimum and minimum[0] <= j - width:
            minimum.popleft()
        while maximum and maximum[0] <= j - width:
            maximum.popleft()
        if j + 1 >= width:
            average = running / width
            variance = max(0.0, squared / width - average * average)
            result[j + 1] = (
                average, sqrt(variance), values[minimum[0]], values[maximum[0]]
            )
    return result


def _feature_values(points, prices, demands, rolling, i):
    """Only read indices < i. Caller keeps the future target separate."""
    if i < WARMUP_STEPS or i + HORIZON_STEPS - WEEKLY_TARGET_LAG_STEPS < 0:
        raise ValueError("Insufficient decision-time history")
    point = points[i]
    local = point.local_time
    hour = local.hour + local.minute / 60
    angle = 2 * pi * hour / 24
    weekday_angle = 2 * pi * local.weekday() / 7
    row = {
        "hour_sin": sin(angle), "hour_cos": cos(angle),
        "weekday_sin": sin(weekday_angle), "weekday_cos": cos(weekday_angle),
        "local_month": local.month, "local_day_of_month": local.day,
        "local_day_of_year": local.timetuple().tm_yday,
        "is_weekend": int(local.weekday() >= 5),
        "is_dst": int(local.dst() != timedelta(0)),
        "local_fold": point.local_fold,
    }
    for minutes in LAG_MINUTES:
        row[f"price_lag_{minutes}m_aud_per_kwh"] = prices[i - minutes // 5]
    for minutes in ROLL_MINUTES:
        stats = rolling[minutes][i]
        if stats is None:
            raise ValueError("Rolling history unavailable")
        for label, value in zip(("mean", "std", "min", "max"), stats):
            row[f"price_trailing_{minutes}m_{label}_aud_per_kwh"] = value
    latest = prices[i - 1]
    row["price_change_5m_vs_15m_aud_per_kwh"] = latest - prices[i - 3]
    row["price_change_5m_vs_60m_aud_per_kwh"] = latest - prices[i - 12]
    row["price_change_5m_vs_1440m_aud_per_kwh"] = latest - prices[i - 288]
    for minutes in DEMAND_LAG_MINUTES:
        row[f"demand_lag_{minutes}m_mw"] = demands[i - minutes // 5]
    row["demand_change_5m_vs_15m_mw"] = demands[i - 1] - demands[i - 3]
    row["demand_change_5m_vs_60m_mw"] = demands[i - 1] - demands[i - 12]
    # These are prices for the *target clock time* one day/week earlier.
    row[SEASONAL_FIELDS[0]] = prices[i + HORIZON_STEPS - DAILY_TARGET_LAG_STEPS]
    row[SEASONAL_FIELDS[1]] = prices[i + HORIZON_STEPS - WEEKLY_TARGET_LAG_STEPS]
    return row


def feature_values_at(points, i):
    """Small pure entry point for leakage tests and single-decision inspection."""
    prices = [point.price_aud_per_kwh for point in points]
    demands = [point.regional_demand_mw for point in points]
    rolling = {minutes: trailing_stats(prices, minutes // 5)
               for minutes in ROLL_MINUTES}
    return _feature_values(points, prices, demands, rolling, i)


def _gzip_writer(path):
    # mtime=0 and empty embedded filename make byte-for-byte reruns stable.
    binary = path.open("wb")
    compressed = gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0)
    text = TextIOWrapper(compressed, encoding="utf-8", newline="")
    return binary, compressed, text


def build_feature_dataset(points, output_dir=FEATURE_DIR):
    """Write chronological train/validation CSV.gz with no future features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = [point.price_aud_per_kwh for point in points]
    demands = [point.regional_demand_mw for point in points]
    rolling = {minutes: trailing_stats(prices, minutes // 5)
               for minutes in ROLL_MINUTES}
    paths = {name: output_dir / f"{name}_one_hour_price.csv.gz"
             for name in ("train", "validation")}
    handles = {}
    writers = {}
    counts = Counter()
    try:
        for name, path in paths.items():
            handles[name] = _gzip_writer(path.with_suffix(path.suffix + ".tmp"))
            writer = csv.DictWriter(handles[name][2], fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writers[name] = writer
        for i in range(WARMUP_STEPS, len(points) - HORIZON_STEPS):
            origin = points[i]
            target = points[i + HORIZON_STEPS]
            if target.time_utc - origin.time_utc != HORIZON_STEPS * STEP:
                raise ValueError("Target not exactly one hour after origin")
            if target.partition != origin.partition:
                counts["cross_partition_targets_excluded"] += 1
                continue
            if i + HORIZON_STEPS - WEEKLY_TARGET_LAG_STEPS < 0:
                counts["insufficient_weekly_history_excluded"] += 1
                continue
            row = {
                "decision_start_utc": origin.time_utc.isoformat().replace("+00:00", "Z"),
                "target_interval_start_utc": target.time_utc.isoformat().replace("+00:00", "Z"),
                "origin_local_date": origin.local_time.date().isoformat(),
                "origin_partition": origin.partition,
                "target_partition": target.partition,
                "is_development_origin": int(origin.development_week),
                "is_development_target": int(target.development_week),
                "include_in_validation_score": int(
                    origin.partition == "validation"
                    and not origin.development_week and not target.development_week),
            }
            row.update(_feature_values(points, prices, demands, rolling, i))
            row[TARGET_FIELD] = target.price_aud_per_kwh
            writers[origin.partition].writerow(row)
            counts[origin.partition] += 1
            if row["include_in_validation_score"]:
                counts["validation_scored"] += 1
            elif origin.partition == "validation":
                counts["validation_development_excluded"] += 1
    finally:
        for binary, compressed, text in handles.values():
            text.close()
            compressed.close()
            binary.close()
    for path in paths.values():
        path.with_suffix(path.suffix + ".tmp").replace(path)
    result = {
        "schema_version": 1,
        "forecast_horizon_minutes": 60,
        "source_coverage": {
            "first_interval_start_utc": points[WARMUP_STEPS].time_utc.isoformat(),
            "last_interval_start_utc": points[-1].time_utc.isoformat(),
            "processed_intervals": len(points) - WARMUP_STEPS,
            "train_intervals": sum(p.partition == "train" for p in points),
            "validation_intervals": sum(p.partition == "validation" for p in points),
            "negative_price_intervals": sum(
                p.price_aud_per_kwh < 0 for p in points[WARMUP_STEPS:]),
            "at_or_above_one_aud_per_kwh_intervals": sum(
                p.price_aud_per_kwh >= 1 for p in points[WARMUP_STEPS:]),
            "minimum_price_aud_per_kwh": min(
                p.price_aud_per_kwh for p in points[WARMUP_STEPS:]),
            "maximum_price_aud_per_kwh": max(
                p.price_aud_per_kwh for p in points[WARMUP_STEPS:]),
            "sydney_repeated_hour_fold_intervals": sum(
                p.local_fold == 1 for p in points[WARMUP_STEPS:]),
        },
        "decision_price_assumption": ("At decision interval start t, the interval "
            "ending at t (start t-5 minutes) is assumed published with zero "
            "additional delay. Revised monthly AEMO files are hindsight values."),
        "warmup_local_date": "2022-12-31",
        "warmup_intervals": WARMUP_STEPS,
        "weekly_history_note": ("A one-day source warm-up fills the 24-hour lags. "
            "The target-aligned previous-week feature requires seven days; "
            "early January 2023 origins without it are excluded."),
        "counts": dict(counts),
        "feature_count": len(FEATURE_FIELDS),
        "files": {name: {"path": path.relative_to(ROOT).as_posix(),
                          "sha256": sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size}
                  for name, path in paths.items()},
    }
    _write_json_atomic(output_dir / "build_report.json", result)
    return result


def read_feature_rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)
