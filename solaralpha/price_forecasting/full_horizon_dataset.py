"""Offline labels for every five-minute start in a 12-hour price trajectory.

This module may inspect realised future prices to create labels. Forecast inputs
continue to come exclusively from the frozen 41-column feature files.
"""

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from solaralpha.historical_pipeline import ROOT, STEP, _write_json_atomic
from .features import FEATURE_FIELDS, load_training_validation_points
from .ml_dataset import _utc, verify_feature_files
from .multi_horizon import HORIZONS as SPARSE_HORIZONS, load_matrices


HORIZONS = tuple(range(0, 720, 5))
TARGET_DIR = ROOT / "data/features/full_horizon_v1"
PARTS = ("train", "tune", "evaluate")


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class FullHorizonData:
    matrices: dict
    targets: dict
    seasonal: dict
    common: dict
    manifest: dict


def build_targets(output_dir=TARGET_DIR):
    """Write checksum-verified, row-aligned float64 matrices without window copies."""
    if len(HORIZONS) != 144 or HORIZONS[-1] != 715:
        raise ValueError("Dense horizon definition changed")
    matrices, sparse_manifest = load_matrices()
    points = load_training_validation_points()
    if any(points[i].time_utc - points[i - 1].time_utc != STEP for i in range(1, len(points))):
        raise ValueError("Realised price grid is not UTC-contiguous")
    prices = np.asarray([point.price_aud_per_kwh for point in points], dtype=np.float64)
    # -1 marks the held-out/excluded development week or feature-only warm-up.
    codes = np.full(len(points), -1, dtype=np.int8)
    for i, point in enumerate(points):
        if point.development_week:
            continue
        if point.partition == "train":
            codes[i] = 0
        elif point.partition == "validation":
            codes[i] = 1 if point.local_time.date().month < 7 else 2
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "horizons_minutes": list(HORIZONS),
                "units": "AUD/kWh", "target_semantics": "five-minute interval start",
                "feature_order": list(FEATURE_FIELDS),
                "feature_files": verify_feature_files()["files"],
                "sparse_target_manifest_sha256": _sha(ROOT / "data/features/multi_horizon_v1/manifest.json"),
                "partitions": {}, "files": {},
                "feature_note": "frozen +60-minute-aligned seasonal input columns unchanged",
                "held_out": "December 2025-February 2026 not loaded"}
    for part in PARTS:
        matrix = matrices[part]
        origins = [row["decision_start_utc"] for row in matrix.rows]
        indices = np.fromiter((int((_utc(t) - points[0].time_utc) / STEP) for t in origins),
                              dtype=np.int32, count=len(origins))
        if np.any(np.diff(indices) <= 0) or np.any(indices < 288):
            raise ValueError("Feature origins lack ordered historical context")
        code = PARTS.index(part)
        if np.any((codes[indices] != code) & (codes[indices] != -1)):
            raise ValueError("Origin partition changed")
        paths = {name: output_dir / f"{part}_{name}.npy" for name in ("prices", "seasonal")}
        temporary = {name: path.with_suffix(".npy.tmp") for name, path in paths.items()}
        arrays = {name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float64,
                                                  shape=(len(indices), len(HORIZONS)))
                  for name, path in temporary.items()}
        counts = {}
        for col, horizon in enumerate(HORIZONS):
            target_indices = indices + horizon // 5
            inside = target_indices < len(points)
            safe = np.minimum(target_indices, len(points) - 1)
            eligible = inside & (codes[indices] == code) & (codes[safe] == code)
            if np.any(target_indices[eligible] - 288 >= indices[eligible]):
                raise ValueError("Seasonal baseline crosses decision time")
            for array in arrays.values():
                array[:, col] = np.nan
            arrays["prices"][eligible, col] = prices[target_indices[eligible]]
            arrays["seasonal"][eligible, col] = prices[target_indices[eligible] - 288]
            counts[str(horizon)] = int(eligible.sum())
            if horizon in SPARSE_HORIZONS:
                old = SPARSE_HORIZONS.index(horizon)
                if (not np.array_equal(np.asarray(arrays["prices"][:, col]), matrix.y[:, old],
                                       equal_nan=True) or
                    not np.array_equal(np.asarray(arrays["seasonal"][:, col]), matrix.seasonal[:, old],
                                       equal_nan=True)):
                    raise ValueError(f"Dense target differs from frozen sparse +{horizon}m")
        common = np.isfinite(np.asarray(arrays["prices"])).all(axis=1)
        if not np.array_equal(common, matrix.common):
            raise ValueError("Full-trajectory intersection differs from frozen sparse intersection")
        for array in arrays.values():
            array.flush()
        del array
        arrays.clear()
        for name in paths:
            temporary[name].replace(paths[name])
            manifest["files"][f"{part}_{name}"] = {
                "path": paths[name].relative_to(ROOT).as_posix(),
                "sha256": _sha(paths[name]), "bytes": paths[name].stat().st_size}
        manifest["partitions"][part] = {"origins": len(indices), "common": int(common.sum()),
                                         "eligible_counts": counts,
                                         "origin_sha256": sha256("\n".join(origins).encode()).hexdigest()}
    if manifest["partitions"]["train"]["common"] != sparse_manifest["common_counts"]["train"]:
        raise ValueError("Frozen common training count changed")
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def load_targets(target_dir=TARGET_DIR, matrices=None):
    """Memory-map dense targets and verify their frozen-feature alignment."""
    matrices = matrices or load_matrices()[0]
    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    if (tuple(manifest["horizons_minutes"]) != HORIZONS or
            tuple(manifest["feature_order"]) != FEATURE_FIELDS or
            manifest["feature_files"] != verify_feature_files()["files"]):
        raise ValueError("Dense target schema or feature fingerprint differs")
    targets, seasonal, common = {}, {}, {}
    for part in PARTS:
        matrix = matrices[part]
        expected = manifest["partitions"][part]
        origins = [row["decision_start_utc"] for row in matrix.rows]
        if (len(origins) != expected["origins"] or
                sha256("\n".join(origins).encode()).hexdigest() != expected["origin_sha256"]):
            raise ValueError("Dense target origin order differs")
        for name, destination in (("prices", targets), ("seasonal", seasonal)):
            record = manifest["files"][f"{part}_{name}"]
            path = ROOT / record["path"]
            if _sha(path) != record["sha256"]:
                raise ValueError("Dense target checksum differs")
            destination[part] = np.load(path, mmap_mode="r")
            if destination[part].shape != (len(origins), len(HORIZONS)):
                raise ValueError("Dense target shape differs")
        common[part] = np.isfinite(targets[part]).all(axis=1)
        if not np.array_equal(common[part], matrix.common):
            raise ValueError("Dense common-origin selection changed")
        for col, horizon in enumerate(HORIZONS):
            if int(np.isfinite(targets[part][:, col]).sum()) != expected["eligible_counts"][str(horizon)]:
                raise ValueError("Dense target eligibility count changed")
    return FullHorizonData(matrices, targets, seasonal, common, manifest)
