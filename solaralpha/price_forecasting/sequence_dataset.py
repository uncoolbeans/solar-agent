"""Lazy, chronological 72-row inputs for the unchanged one-hour price target."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

from solaralpha.historical_pipeline import PROCESSED, ROOT, sydney_zone
from .features import (FEATURE_DIR, FEATURE_FIELDS, ROLL_MINUTES, PricePoint,
                       _assert_month_integrity, _feature_values, read_feature_rows,
                       trailing_stats)
from .ml_dataset import load_slices, period_for_row


WINDOW = 72
STEP_MINUTES = 5
BRIDGE_START = "2024-12-31T12:00:00Z"
BRIDGE_COUNT = 12


def _minute(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.utcoffset() != timedelta(0):
        raise ValueError("Sequence feature time must be UTC")
    return int(instant.timestamp() // 60)


def _bridge_context(processed_dir=PROCESSED):
    """Reconstruct only the 12 unlabelled 2024-12-31 feature rows."""
    path = processed_dir / "2024-12_realised.csv"
    _assert_month_integrity(path)
    points = []
    with path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            utc = datetime.fromisoformat(row["interval_start_utc"].replace("Z", "+00:00"))
            local = utc.astimezone(sydney_zone())
            points.append(PricePoint(utc, local,
                                     float(row["realised_spot_aud_per_kwh"]),
                                     float(row["realised_regional_demand_mw"]),
                                     "train", False, local.fold))
    prices = [point.price_aud_per_kwh for point in points]
    demands = [point.regional_demand_mw for point in points]
    rolling = {minutes: trailing_stats(prices, minutes // STEP_MINUTES)
               for minutes in ROLL_MINUTES}
    by_time = {point.time_utc.strftime("%Y-%m-%dT%H:%M:%SZ"): i
               for i, point in enumerate(points)}
    last_known = "2024-12-31T11:55:00Z"
    first_bridge = datetime.fromisoformat(BRIDGE_START.replace("Z", "+00:00"))
    missing = [(first_bridge + i * timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
               for i in range(BRIDGE_COUNT)]
    if last_known not in by_time or any(t not in by_time for t in missing):
        raise ValueError("December 2024 context does not cover train/validation bridge")
    def at(timestamp):
        i = by_time[timestamp]
        if i < 7 * 24 * 12:
            raise ValueError("Bridge lacks weekly feature history")
        feature = _feature_values(points, prices, demands, rolling, i)
        return [float(feature[field]) for field in FEATURE_FIELDS]
    return last_known, at(last_known), [(t, at(t)) for t in missing], sha256(path.read_bytes()).hexdigest()


@dataclass
class SequenceData:
    x: object
    minutes: object
    segments: object
    indices: dict
    targets: dict
    rows: dict
    scaler_mean: object
    scaler_scale: object
    feature_files: dict
    bridge_source_sha256: str
    exclusions: dict

    def windows(self, part):
        return WindowDataset(self.x, self.indices[part], self.targets[part])


class WindowDataset:
    """Retain one matrix; each item views its 72 chronological feature rows."""

    def __init__(self, scaled_matrix, end_indices, labels):
        if len(end_indices) != len(labels):
            raise ValueError("Window ends and targets differ")
        self.x = scaled_matrix
        self.ends = end_indices
        self.labels = labels

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, item):
        import torch
        end = int(self.ends[item])
        window = self.x[end - WINDOW + 1:end + 1]
        if window.shape != (WINDOW, len(FEATURE_FIELDS)):
            raise ValueError("Incomplete feature history")
        return torch.from_numpy(window), torch.tensor(self.labels[item], dtype=torch.float32)


def load_sequence_data(feature_dir=FEATURE_DIR, processed_dir=PROCESSED):
    """Fingerprint features, reconstruct boundary context, and forbid split-crossing labels."""
    import numpy as np

    frozen = load_slices(feature_dir)
    capacities = frozen.feature_report["counts"]
    total = capacities["train"] + capacities["validation"] + BRIDGE_COUNT
    x = np.empty((total, len(FEATURE_FIELDS)), dtype=np.float32)
    minutes = np.empty(total, dtype=np.int64)
    selected = {part: [] for part in ("train", "tune", "evaluate")}
    targets = {part: [] for part in selected}
    last_known, last_features, bridge, bridge_hash = _bridge_context(processed_dir)
    position = 0
    for split in ("train", "validation"):
        path = feature_dir / f"{split}_one_hour_price.csv.gz"
        for row in read_feature_rows(path):
            if split == "validation" and position == capacities["train"]:
                if minutes[position - 1] != _minute(last_known):
                    raise ValueError("Training feature end changed")
                if not np.allclose(x[position - 1], last_features, rtol=1e-5, atol=1e-6):
                    raise ValueError("Rebuilt bridge formula differs from frozen feature formula")
                for time, features in bridge:
                    x[position] = features
                    minutes[position] = _minute(time)
                    position += 1
            x[position] = [float(row[field]) for field in FEATURE_FIELDS]
            minutes[position] = _minute(row["decision_start_utc"])
            part = period_for_row(row)
            if part is not None:
                selected[part].append(position)
                targets[part].append(float(row["target_spot_aud_per_kwh"]))
            position += 1
    if position != total or np.any(np.diff(minutes) != STEP_MINUTES):
        raise ValueError("Sequence context has missing, duplicate or unordered five-minute rows")
    segments = np.zeros(total, dtype=np.int32)
    for i in range(1, total):
        segments[i] = segments[i - 1] if minutes[i] - minutes[i - 1] == STEP_MINUTES else i
    exclusions = {}
    for part in selected:
        original = len(selected[part])
        keep = [k for k, end in enumerate(selected[part]) if end - WINDOW + 1 >= segments[end]]
        selected[part] = np.asarray([selected[part][k] for k in keep], dtype=np.int32)
        targets[part] = np.asarray([targets[part][k] for k in keep], dtype=np.float64)
        exclusions[part] = original - len(keep)
    if exclusions != {"train": 71, "tune": 0, "evaluate": 0}:
        raise ValueError(f"Unexpected sequence warm-up losses: {exclusions}")
    if len(selected["train"]) != 208729 or len(selected["tune"]) != 50100 or len(selected["evaluate"]) != 44040:
        raise ValueError("Sequence split counts differ from prespecified experiment")
    for part in ("tune", "evaluate"):
        portion = getattr(frozen, part)
        if not np.array_equal(targets[part], portion.y):
            raise ValueError(f"{part} targets no longer match frozen comparisons")
        if not np.array_equal(x[selected[part]], portion.x):
            raise ValueError(f"{part} feature rows no longer match frozen comparisons")
    if not np.array_equal(targets["train"], frozen.train.y[71:]):
        raise ValueError("Training labels differ after sequence warm-up")
    mean = frozen.train.x.mean(axis=0, dtype=np.float64)
    scale = frozen.train.x.std(axis=0, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    scaled = ((x - mean) / scale).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite scaled sequence feature")
    return SequenceData(scaled, minutes, segments, selected, targets,
                        {"tune": frozen.tune.rows, "evaluate": frozen.evaluate.rows},
                        mean, scale, frozen.feature_report["files"], bridge_hash, exclusions)
