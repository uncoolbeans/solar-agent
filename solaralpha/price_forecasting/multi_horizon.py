"""Offline direct multi-horizon experiment on the frozen 41-feature origin rows.

Only this diagnostic builder reads realised prices at/after a decision. Models
receive FEATURE_FIELDS exclusively; targets and baseline labels stay separate.
"""

import csv
from dataclasses import dataclass
from datetime import date, timedelta
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter
from itertools import zip_longest

from solaralpha.historical_pipeline import ROOT, STEP, _write_json_atomic, sydney_zone
from .features import (FEATURE_DIR, FEATURE_FIELDS, WARMUP_STEPS,
                       load_training_validation_points, read_feature_rows)
from .ml_dataset import _utc, verify_feature_files
from .ml_metrics import full_scores, grouped_scores, numeric_scores
from .xgb_v1 import PARAMETERS
from .xgb_v2 import _conditional_scores


HORIZONS = (0, 5, 15, 30, 60, 120, 180, 360, 540, 715)
TARGET_DIR = ROOT / "data" / "features" / "multi_horizon_v1"
MODEL_DIR = ROOT / "data" / "models" / "price_multi_horizon_v1"
TUNE_START = date(2025, 1, 1)
EVAL_START = date(2025, 7, 1)
EVAL_END = date(2025, 11, 30)
MODEL_NAMES = ("xgboost", "persistence", "daily_seasonal", "train_median",
               "hour_of_week_median")
TARGET_FIELDS = tuple(f"price_h{h}_aud_per_kwh" for h in HORIZONS)
TARGET_HEADER = ("decision_start_utc", "period", "common_eligible", *TARGET_FIELDS,
                 *(f"seasonal_h{h}_aud_per_kwh" for h in HORIZONS))
LATEST_INDEX = FEATURE_FIELDS.index("price_lag_5m_aud_per_kwh")


def eligible_period(origin, target):
    """Both five-minute interval starts must remain in the same allowed slice."""
    if origin.partition != target.partition or origin.development_week or target.development_week:
        return None
    d0, d1 = origin.local_time.date(), target.local_time.date()
    if origin.partition == "train":
        return "train" if d0 < TUNE_START and d1 < TUNE_START else None
    if origin.partition != "validation":
        return None
    if TUNE_START <= d0 < EVAL_START and d1 < EVAL_START:
        return "tune"
    if EVAL_START <= d0 <= EVAL_END and EVAL_START <= d1 <= EVAL_END:
        return "evaluate"
    return None


def target_for_horizon(points, index, horizon_minutes):
    """Return (target price, prior-24-hour price) or None; never a feature."""
    if horizon_minutes not in HORIZONS:
        raise ValueError("Unconfigured horizon")
    target_index = index + horizon_minutes // 5
    if target_index >= len(points):
        return None
    origin, target = points[index], points[target_index]
    if target.time_utc - origin.time_utc != timedelta(minutes=horizon_minutes):
        raise ValueError("Target interval is not aligned")
    if eligible_period(origin, target) is None:
        return None
    seasonal_index = target_index - 288
    if seasonal_index < 0 or points[seasonal_index].time_utc >= origin.time_utc:
        raise ValueError("Seasonal baseline is not decision-time available")
    return target.price_aud_per_kwh, points[seasonal_index].price_aud_per_kwh


def _writer(path):
    binary = path.open("wb")
    compressed = gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0)
    stream = TextIOWrapper(compressed, encoding="utf-8", newline="")
    return binary, compressed, stream


def build_targets(output_dir=TARGET_DIR, feature_dir=FEATURE_DIR, points=None):
    """Versioned labels keyed to frozen feature origins, without altering them."""
    report = verify_feature_files(feature_dir)
    if points is None:
        points = load_training_validation_points()
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {part: {str(h): 0 for h in HORIZONS} for part in ("train", "tune", "evaluate")}
    common = {part: 0 for part in counts}
    feature_origins = {"train": 0, "validation": 0}
    files = {}
    for source_split in ("train", "validation"):
        path = output_dir / f"{source_split}_targets.csv.gz"
        temporary = path.with_suffix(path.suffix + ".tmp")
        handles = _writer(temporary)
        try:
            writer = csv.DictWriter(handles[2], fieldnames=TARGET_HEADER)
            writer.writeheader()
            for feature in read_feature_rows(feature_dir / f"{source_split}_one_hour_price.csv.gz"):
                origin_time = _utc(feature["decision_start_utc"])
                delta = origin_time - points[0].time_utc
                if delta < timedelta(0) or delta % STEP:
                    raise ValueError("Feature origin is outside five-minute price grid")
                index = int(delta / STEP)
                origin = points[index]
                if origin.time_utc != origin_time or origin.partition != source_split:
                    raise ValueError("Feature/source origin mismatch")
                feature_origins[source_split] += 1
                origin_date = origin.local_time.date()
                period = ("train" if source_split == "train" else
                          "tune" if origin_date < EVAL_START else "evaluate")
                values = {h: target_for_horizon(points, index, h) for h in HORIZONS}
                if values[60] is not None and abs(values[60][0] - float(feature["target_spot_aud_per_kwh"])) > 1e-10:
                    raise ValueError("Frozen one-hour target differs from source")
                all_eligible = all(value is not None for value in values.values())
                output = {"decision_start_utc": feature["decision_start_utc"],
                          "period": period, "common_eligible": int(all_eligible)}
                for h, value in values.items():
                    output[f"price_h{h}_aud_per_kwh"] = "" if value is None else value[0]
                    output[f"seasonal_h{h}_aud_per_kwh"] = "" if value is None else value[1]
                    if value is not None:
                        counts[period][str(h)] += 1
                if all_eligible:
                    common[period] += 1
                writer.writerow(output)
        finally:
            handles[2].close()
            handles[1].close()
            handles[0].close()
        temporary.replace(path)
        files[source_split] = {"path": path.relative_to(ROOT).as_posix(),
                               "sha256": sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size}
    result = {"schema_version": 1, "horizons_minutes": list(HORIZONS),
              "feature_files": report["files"], "feature_count": len(FEATURE_FIELDS),
              "feature_origins": feature_origins, "eligible_counts": counts,
              "common_counts": common, "files": files,
              "source_last_interval_start_utc": points[-1].time_utc.isoformat(),
              "seasonal_rule": "target interval start minus 24 hours UTC (288 five-minute intervals)",
              "feature_note": "The two frozen seasonal input columns remain aligned to +60 minutes at every horizon; the baseline is aligned separately."}
    _write_json_atomic(output_dir / "manifest.json", result)
    return result


@dataclass
class Matrix:
    x: object
    y: object
    seasonal: object
    common: object
    rows: list


def load_matrices(target_dir=TARGET_DIR, feature_dir=FEATURE_DIR):
    """Strictly align separate target artifacts to ordered frozen feature rows."""
    import numpy as np
    manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    report = verify_feature_files(feature_dir)
    if manifest["feature_files"] != report["files"] or tuple(manifest["horizons_minutes"]) != HORIZONS:
        raise ValueError("Target manifest and source feature version disagree")
    for split in ("train", "validation"):
        path = target_dir / f"{split}_targets.csv.gz"
        if sha256(path.read_bytes()).hexdigest() != manifest["files"][split]["sha256"]:
            raise ValueError("Target artifact checksum mismatch")
    parts = {p: {"x": [], "y": [], "seasonal": [], "common": [], "rows": []}
             for p in ("train", "tune", "evaluate")}
    for split in ("train", "validation"):
        target_path = target_dir / f"{split}_targets.csv.gz"
        with gzip.open(target_path, "rt", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames) != TARGET_HEADER:
                raise ValueError("Multi-horizon target header changed")
            for f, t in zip_longest(read_feature_rows(feature_dir / f"{split}_one_hour_price.csv.gz"), reader):
                if f is None or t is None or f["decision_start_utc"] != t["decision_start_utc"]:
                    raise ValueError("Target and feature origins differ")
                period = t["period"]
                if period not in parts or (split == "train") != (period == "train"):
                    raise ValueError("Target period mismatch")
                bucket = parts[period]
                bucket["x"].append([float(f[field]) for field in FEATURE_FIELDS])
                bucket["y"].append([float(t[field]) if t[field] else np.nan for field in TARGET_FIELDS])
                bucket["seasonal"].append([float(t[f"seasonal_h{h}_aud_per_kwh"])
                                           if t[f"seasonal_h{h}_aud_per_kwh"] else np.nan for h in HORIZONS])
                bucket["common"].append(t["common_eligible"] == "1")
                bucket["rows"].append({"decision_start_utc": t["decision_start_utc"]})
    matrices = {name: Matrix(np.asarray(b["x"], dtype=np.float32),
                             np.asarray(b["y"], dtype=np.float64),
                             np.asarray(b["seasonal"], dtype=np.float64),
                             np.asarray(b["common"], dtype=bool), b["rows"])
                for name, b in parts.items()}
    for name, matrix in matrices.items():
        if int(matrix.common.sum()) != manifest["common_counts"][name]:
            raise ValueError("Common eligible count mismatch")
        for col, h in enumerate(HORIZONS):
            if int(np.isfinite(matrix.y[:, col]).sum()) != manifest["eligible_counts"][name][str(h)]:
                raise ValueError("Horizon eligible count mismatch")
    return matrices, manifest


def predict_saved(model_path, metadata_path, matrix, feature_names, horizon_minutes):
    import numpy as np
    import xgboost as xgb
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if (tuple(feature_names) != FEATURE_FIELDS or tuple(metadata["feature_order"]) != FEATURE_FIELDS
            or metadata["horizon_minutes"] != horizon_minutes or matrix.ndim != 2
            or matrix.shape[1] != len(FEATURE_FIELDS)):
        raise ValueError("Multi-horizon inference schema or horizon mismatch")
    if sha256(Path(model_path).read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Multi-horizon model checksum mismatch")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return np.asarray(model.predict(matrix))


def _baseline_predictions(matrices, part, col, mask):
    import numpy as np
    train = matrices["train"]
    train_mask = np.isfinite(train.y[:, col])
    labels = train.y[train_mask, col]
    groups = [[] for _ in range(168)]
    for row, value in zip(np.asarray(train.rows, dtype=object)[train_mask], labels):
        target = (_utc(row["decision_start_utc"]) + timedelta(minutes=HORIZONS[col])).astimezone(sydney_zone())
        groups[target.weekday() * 24 + target.hour].append(float(value))
    overall = median(labels.tolist())
    if any(not g for g in groups):
        raise ValueError("Missing training hour-of-week target bin")
    portion = matrices[part]
    how = []
    for row in np.asarray(portion.rows, dtype=object)[mask]:
        target = (_utc(row["decision_start_utc"]) + timedelta(minutes=HORIZONS[col])).astimezone(sydney_zone())
        how.append(median(groups[target.weekday() * 24 + target.hour]))
    return {"persistence": portion.x[mask, LATEST_INDEX].astype(float),
            "daily_seasonal": portion.seasonal[mask, col],
            "train_median": np.full(int(mask.sum()), overall),
            "hour_of_week_median": np.asarray(how)}


def _event_onset(actual, predicted, latest, negative):
    import numpy as np
    threshold = 0.0 if negative else 1.0
    actual_event = actual < threshold if negative else actual >= threshold
    prior_event = latest < threshold if negative else latest >= threshold
    prediction_event = predicted < threshold if negative else predicted >= threshold
    onset = actual_event & ~prior_event
    return {"actual_onsets": int(onset.sum()),
            "detected_onsets": int((onset & prediction_event).sum()),
            "false_onset_signals": int((~actual_event & ~prior_event & prediction_event).sum())}


def _scores(matrices, part, col, mask, forecasts):
    import numpy as np
    portion = matrices[part]
    actual = portion.y[mask, col]
    latest = portion.x[mask, LATEST_INDEX].astype(float)
    rows = []
    for row in np.asarray(portion.rows, dtype=object)[mask]:
        local = (_utc(row["decision_start_utc"]) + timedelta(minutes=HORIZONS[col])).astimezone(sydney_zone())
        rows.append({"target_local_month": local.strftime("%Y-%m"),
                     "target_local_hour": local.hour})
    return {"count": len(actual),
            "models": {name: {**full_scores(actual, forecast),
                              "negative_onset": _event_onset(actual, forecast, latest, True),
                              "spike_onset": _event_onset(actual, forecast, latest, False)}
                       for name, forecast in forecasts.items()},
            "conditional_price_ranges": _conditional_scores(actual, forecasts),
            "monthly": grouped_scores(actual, forecasts, rows, "target_local_month"),
            "hourly": grouped_scores(actual, forecasts, rows, "target_local_hour")}


def _prediction_file(path, matrices, part, col, mask, forecasts):
    portion = matrices[part]
    temporary = path.with_suffix(path.suffix + ".tmp")
    handles = _writer(temporary)
    try:
        fields = ("decision_start_utc", "target_interval_start_utc", "actual_aud_per_kwh",
                  *(f"{name}_aud_per_kwh" for name in MODEL_NAMES))
        writer = csv.DictWriter(handles[2], fieldnames=fields)
        writer.writeheader()
        indices = [i for i, keep in enumerate(mask) if keep]
        for j, i in enumerate(indices):
            instant = _utc(portion.rows[i]["decision_start_utc"]) + timedelta(minutes=HORIZONS[col])
            writer.writerow({"decision_start_utc": portion.rows[i]["decision_start_utc"],
                             "target_interval_start_utc": instant.isoformat().replace("+00:00", "Z"),
                             "actual_aud_per_kwh": portion.y[i, col],
                             **{f"{name}_aud_per_kwh": forecast[j] for name, forecast in forecasts.items()}})
    finally:
        handles[2].close(); handles[1].close(); handles[0].close()
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def run_experiment(target_dir=TARGET_DIR, output_dir=MODEL_DIR):
    import numpy as np
    import xgboost as xgb
    matrices, manifest = load_matrices(target_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {**PARAMETERS, "objective": "reg:absoluteerror"}
    result = {"schema_version": 1, "task": "direct ten-horizon NSW1 five-minute interval-start price",
              "horizons_minutes": list(HORIZONS), "target_manifest": manifest,
              "parameters": params, "july_november_status": "reused development evaluation",
              "held_out_status": "December 2025-February 2026 untouched",
              "horizons": {}}
    for col, horizon in enumerate(HORIZONS):
        print(f"Training +{horizon}m", flush=True)
        train, tune = matrices["train"], matrices["tune"]
        train_mask = np.isfinite(train.y[:, col])
        tune_mask = np.isfinite(tune.y[:, col])
        model = xgb.XGBRegressor(**params)
        start = perf_counter()
        model.fit(train.x[train_mask], train.y[train_mask, col],
                  eval_set=[(train.x[train_mask], train.y[train_mask, col]),
                            (tune.x[tune_mask], tune.y[tune_mask, col])], verbose=False)
        train_seconds = perf_counter() - start
        folder = output_dir / f"h{horizon:03d}"
        folder.mkdir(exist_ok=True)
        model_path = folder / "model.json"
        model.save_model(model_path)
        metadata = {"horizon_minutes": horizon, "feature_order": list(FEATURE_FIELDS),
                    "feature_files": manifest["feature_files"], "target_files": manifest["files"],
                    "parameters": params, "python": platform.python_version(),
                    "xgboost": xgb.__version__, "best_iteration_zero_based": int(model.best_iteration),
                    "best_tune_mae": float(model.best_score), "train_seconds": train_seconds,
                    "fitted_rounds": len(model.evals_result()["validation_1"]["mae"]),
                    "model_sha256": sha256(model_path.read_bytes()).hexdigest()}
        _write_json_atomic(folder / "metadata.json", metadata)
        record = {"metadata": metadata, "learning_curve": model.evals_result(), "splits": {}}
        for part in ("tune", "evaluate"):
            portion = matrices[part]
            eligible = np.isfinite(portion.y[:, col])
            start = perf_counter()
            predictions = model.predict(portion.x[eligible])
            inference = perf_counter() - start
            forecasts = {"xgboost": predictions,
                         **_baseline_predictions(matrices, part, col, eligible)}
            if tuple(forecasts) != MODEL_NAMES:
                raise ValueError("Forecast model ordering changed")
            common_mask = eligible & portion.common
            common_forecasts = {name: values[portion.common[eligible]]
                                for name, values in forecasts.items()}
            record["splits"][part] = {
                "eligible": _scores(matrices, part, col, eligible, forecasts),
                "common": _scores(matrices, part, col, common_mask, common_forecasts),
                "inference_seconds": inference,
                "predictions_file": _prediction_file(folder / f"{part}_predictions.csv.gz",
                                                    matrices, part, col, eligible, forecasts)}
        result["horizons"][str(horizon)] = record
        _write_json_atomic(output_dir / "results.json", result)
    return result


def verify_original_one_hour(output_dir=MODEL_DIR):
    """Reload frozen V2-A on its original eligible rows, then audit new +60m."""
    import numpy as np
    from .xgb_v2 import predict_saved_variant
    from .xgb_v2_diagnostics import V2_DIR

    matrices, manifest = load_matrices()
    original = json.loads((V2_DIR / "v2_results.json").read_text(encoding="utf-8"))
    old_meta = json.loads((V2_DIR / "V2-A" / "metadata.json").read_text(encoding="utf-8"))
    new_meta = json.loads((output_dir / "h060" / "metadata.json").read_text(encoding="utf-8"))
    if old_meta["feature_files"] != manifest["feature_files"]:
        raise ValueError("Original V2-A was trained on different feature files")
    same_model_hash = old_meta["model_sha256"] == new_meta["model_sha256"]
    comparison = {}
    col = HORIZONS.index(60)
    for part in ("tune", "evaluate"):
        portion = matrices[part]
        eligible = np.isfinite(portion.y[:, col])
        old_prediction = predict_saved_variant(
            V2_DIR / "V2-A" / "model.json", V2_DIR / "V2-A" / "metadata.json",
            portion.x[eligible], FEATURE_FIELDS)
        old_mae = numeric_scores(portion.y[eligible, col], old_prediction)["mae"]
        expected = original["splits"][part]["models"]["V2-A"]["mae"]
        if len(old_prediction) != original["splits"][part]["row_count"] or abs(old_mae - expected) > 1e-8:
            raise ValueError("Original V2-A score could not be reproduced")
        common_mask = eligible & portion.common
        old_common = predict_saved_variant(
            V2_DIR / "V2-A" / "model.json", V2_DIR / "V2-A" / "metadata.json",
            portion.x[common_mask], FEATURE_FIELDS)
        comparison[part] = {"original_rows": int(eligible.sum()),
                            "original_v2_a_mae": old_mae,
                            "common_rows": int(common_mask.sum()),
                            "frozen_v2_a_common_mae": numeric_scores(
                                portion.y[common_mask, col], old_common)["mae"]}
    result_path = output_dir / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["original_one_hour_verification"] = {
        "splits": comparison, "same_model_sha256": same_model_hash,
        "original_model_sha256": old_meta["model_sha256"],
        "new_model_sha256": new_meta["model_sha256"]}
    _write_json_atomic(result_path, result)
    return comparison


def add_baseline_runtimes(output_dir=MODEL_DIR):
    """Time deterministic baseline fitting and prepared-row batch inference."""
    import numpy as np
    matrices, _ = load_matrices()
    result_path = output_dir / "results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    train = matrices["train"]
    for col, h in enumerate(HORIZONS):
        train_mask = np.isfinite(train.y[:, col])
        train_labels = train.y[train_mask, col]
        start = perf_counter()
        constant = median(train_labels.tolist())
        constant_fit = perf_counter() - start
        start = perf_counter()
        groups = [[] for _ in range(168)]
        for row, value in zip((train.rows[i] for i in np.flatnonzero(train_mask)), train_labels):
            local = (_utc(row["decision_start_utc"]) + timedelta(minutes=h)).astimezone(sydney_zone())
            groups[local.weekday() * 24 + local.hour].append(float(value))
        medians = [median(group) for group in groups]
        how_fit = perf_counter() - start
        runtimes = {"persistence": {"fit": 0.0}, "daily_seasonal": {"fit": 0.0},
                    "train_median": {"fit": constant_fit},
                    "hour_of_week_median": {"fit": how_fit}}
        for part in ("tune", "evaluate"):
            portion = matrices[part]
            mask = np.isfinite(portion.y[:, col])
            count = int(mask.sum())
            start = perf_counter(); _ = portion.x[mask, LATEST_INDEX].astype(float)
            runtimes["persistence"][f"{part}_batch_inference"] = perf_counter() - start
            start = perf_counter(); _ = portion.seasonal[mask, col].copy()
            runtimes["daily_seasonal"][f"{part}_batch_inference"] = perf_counter() - start
            start = perf_counter(); _ = np.full(count, constant)
            runtimes["train_median"][f"{part}_batch_inference"] = perf_counter() - start
            start = perf_counter()
            predicted = []
            for i in np.flatnonzero(mask):
                local = (_utc(portion.rows[i]["decision_start_utc"])
                         + timedelta(minutes=h)).astimezone(sydney_zone())
                predicted.append(medians[local.weekday() * 24 + local.hour])
            runtimes["hour_of_week_median"][f"{part}_batch_inference"] = perf_counter() - start
        result["horizons"][str(h)]["baseline_runtimes_seconds"] = runtimes
    _write_json_atomic(result_path, result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "train", "verify", "baseline-times"))
    args = parser.parse_args()
    if args.stage == "build":
        print(json.dumps(build_targets()["eligible_counts"], indent=2))
    elif args.stage == "train":
        run_experiment()
    elif args.stage == "verify":
        print(json.dumps(verify_original_one_hour(), indent=2))
    else:
        add_baseline_runtimes()
