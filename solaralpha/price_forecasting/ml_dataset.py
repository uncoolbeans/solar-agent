"""Strict chronological XGBoost V1 slices from the frozen feature CSVs."""

import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from .features import (FEATURE_DIR, FEATURE_FIELDS, OUTPUT_FIELDS, TARGET_FIELD,
                       read_feature_rows)
from solaralpha.historical_pipeline import sydney_zone


TUNE_START = date(2025, 1, 1)
EVALUATE_START = date(2025, 7, 1)
EVALUATE_END = date(2025, 11, 30)


def _utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.utcoffset() != timedelta(0):
        raise ValueError("Feature timestamp must be UTC")
    return result.astimezone(timezone.utc)


def period_for_row(row):
    """Return a usable split only if both origin and target stay in it."""
    origin = _utc(row["decision_start_utc"])
    target = _utc(row["target_interval_start_utc"])
    if target - origin != timedelta(hours=1):
        raise ValueError("Target is not exactly one hour ahead")
    zone = sydney_zone()
    origin_date = origin.astimezone(zone).date()
    target_date = target.astimezone(zone).date()
    if row["origin_local_date"] != origin_date.isoformat():
        raise ValueError("Origin local date mismatch")
    if row["origin_partition"] != row["target_partition"]:
        raise ValueError("Original feature partition crossing")
    if row["origin_partition"] == "train":
        if origin_date > date(2024, 12, 31) or target_date > date(2024, 12, 31):
            raise ValueError("Training target crosses into 2025")
        return "train"
    if row["origin_partition"] != "validation":
        raise ValueError("Unknown or held-out partition")
    if row["include_in_validation_score"] != "1":
        if row["is_development_origin"] != "1" and row["is_development_target"] != "1":
            raise ValueError("Unexplained validation exclusion")
        return None
    if row["is_development_origin"] != "0" or row["is_development_target"] != "0":
        raise ValueError("Development week reached scored rows")
    if TUNE_START <= origin_date < EVALUATE_START and target_date < EVALUATE_START:
        return "tune"
    if EVALUATE_START <= origin_date <= EVALUATE_END and target_date <= EVALUATE_END:
        return "evaluate"
    if origin_date < EVALUATE_START <= target_date:
        return None  # 12 origins at the June/July boundary.
    raise ValueError("Unrecognised validation date or crossing into held-out test")


def target_hour_of_week(row):
    local = _utc(row["target_interval_start_utc"]).astimezone(sydney_zone())
    return local.weekday() * 24 + local.hour


def assert_feature_header(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        fields = tuple(next(csv.reader(source)))
    if fields != OUTPUT_FIELDS:
        raise ValueError(f"Feature ordering/schema mismatch in {path}")


def verify_feature_files(feature_dir=FEATURE_DIR):
    report = json.loads((feature_dir / "build_report.json").read_text(encoding="utf-8"))
    if report["feature_count"] != len(FEATURE_FIELDS) or report["forecast_horizon_minutes"] != 60:
        raise ValueError("Unexpected feature build schema")
    for split in ("train", "validation"):
        path = feature_dir / f"{split}_one_hour_price.csv.gz"
        assert_feature_header(path)
        if sha256(path.read_bytes()).hexdigest() != report["files"][split]["sha256"]:
            raise ValueError(f"Feature fingerprint mismatch: {split}")
    return report


@dataclass
class Slice:
    x: object
    y: object
    rows: list


@dataclass
class Slices:
    train: Slice
    tune: Slice
    evaluate: Slice
    constant_median: float
    hour_of_week_medians: tuple
    counts: dict
    feature_report: dict


def load_slices(feature_dir=FEATURE_DIR):
    """No held-out paths exist here; validate split and feature order while loading."""
    import numpy as np

    report = verify_feature_files(feature_dir)
    capacity = {"train": report["counts"]["train"],
                "tune": report["counts"]["validation"],
                "evaluate": report["counts"]["validation"]}
    x = {part: np.empty((n, len(FEATURE_FIELDS)), dtype=np.float32)
         for part, n in capacity.items()}
    y = {part: np.empty(n, dtype=np.float64) for part, n in capacity.items()}
    rows = {part: [] for part in capacity}
    sizes = {part: 0 for part in capacity}
    training_groups = [[] for _ in range(7 * 24)]
    excluded = {"development": 0, "tune_evaluate_boundary": 0}
    for split in ("train", "validation"):
        path = feature_dir / f"{split}_one_hour_price.csv.gz"
        previous = None
        for row in read_feature_rows(path):
            origin = _utc(row["decision_start_utc"])
            if previous is not None and origin <= previous:
                raise ValueError("Feature origins are not strictly chronological")
            previous = origin
            part = period_for_row(row)
            if part is None:
                key = ("development" if row["include_in_validation_score"] == "0"
                       else "tune_evaluate_boundary")
                excluded[key] += 1
                continue
            if part != split and not (split == "validation" and part in ("tune", "evaluate")):
                raise ValueError("Feature file and derived split disagree")
            index = sizes[part]
            x[part][index, :] = [float(row[field]) for field in FEATURE_FIELDS]
            value = float(row[TARGET_FIELD])
            y[part][index] = value
            if part == "train":
                training_groups[target_hour_of_week(row)].append(value)
            else:
                rows[part].append({
                    "decision_start_utc": row["decision_start_utc"],
                    "target_interval_start_utc": row["target_interval_start_utc"],
                    "origin_local_date": row["origin_local_date"],
                    "target_local_month": _utc(row["target_interval_start_utc"])
                        .astimezone(sydney_zone()).strftime("%Y-%m"),
                    "target_local_hour": _utc(row["target_interval_start_utc"])
                        .astimezone(sydney_zone()).hour,
                    "target_hour_of_week": target_hour_of_week(row),
                    "persistence": float(row["price_lag_5m_aud_per_kwh"]),
                    "daily_seasonal": float(
                        row["price_target_aligned_previous_day_aud_per_kwh"]),
                })
            sizes[part] += 1
    if sizes["train"] != capacity["train"]:
        raise ValueError("Training row count changed")
    if sizes["tune"] + sizes["evaluate"] + sum(excluded.values()) != report["counts"]["validation"]:
        raise ValueError("Validation row accounting mismatch")
    if any(not group for group in training_groups):
        raise ValueError("Missing training hour-of-week bin")
    return Slices(
        *(Slice(x[part][:sizes[part]], y[part][:sizes[part]], rows[part])
          for part in ("train", "tune", "evaluate")),
        constant_median=median(y["train"][:sizes["train"]].tolist()),
        hour_of_week_medians=tuple(median(group) for group in training_groups),
        counts={**sizes, **excluded}, feature_report=report,
    )
