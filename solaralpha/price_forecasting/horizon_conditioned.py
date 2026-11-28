"""Offline shared XGBoost price model for ten *trained* forecast horizons only."""

import csv
import gc
from datetime import timedelta
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
import os
from pathlib import Path
import platform
from statistics import mean
from time import perf_counter

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .multi_horizon import (HORIZONS, MODEL_DIR as DIRECT_DIR, TARGET_DIR,
                            _scores, load_matrices)
from .ml_dataset import _utc
from .xgb_v1 import PARAMETERS


OUTPUT_DIR = ROOT / "data" / "models" / "price_horizon_conditioned_v1"
PLOT_DIR = ROOT / "plots" / "forecasting" / "horizon_conditioned_v1"
FEATURE_ORDER = (*FEATURE_FIELDS, "forecast_horizon_minutes")
FORECAST_NAMES = ("persistence", "daily_seasonal", "train_median",
                  "hour_of_week_median", "direct", "shared", "hybrid")
HYBRID_CANDIDATES = ("persistence", "direct", "shared")
SELECTED_METRIC = "mean_horizon_mae"


def memory_estimate(common_counts):
    """One allocated float32 matrix per fit split, with no evaluation stack."""
    return {part: {"examples": int(count) * len(HORIZONS),
                   "matrix_mib": count * len(HORIZONS) * len(FEATURE_ORDER) * 4 / 2**20,
                   "label_mib": count * len(HORIZONS) * 8 / 2**20}
            for part, count in common_counts.items()}


def mean_horizon_mae(predictions, labels):
    """Equal weight for ten horizon blocks, regardless of row count elsewhere."""
    import numpy as np
    p = np.asarray(predictions).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    if p.shape != y.shape or p.size == 0 or p.size % len(HORIZONS):
        raise ValueError("Balanced metric requires ten equal, aligned horizon blocks")
    return float(np.abs(p - y).reshape(len(HORIZONS), -1).mean(axis=1).mean())


def xgb_balanced_metric(predictions, dmatrix):
    return SELECTED_METRIC, mean_horizon_mae(predictions, dmatrix.get_label())


def stack_common(matrix):
    """Allocate once, fill horizon-major blocks from frozen features and labels."""
    import numpy as np
    indices = np.flatnonzero(matrix.common)
    n = len(indices)
    if not n or not np.isfinite(matrix.y[indices]).all():
        raise ValueError("Common origin set is incomplete")
    x = np.empty((n * len(HORIZONS), len(FEATURE_ORDER)), dtype=np.float32)
    y = np.empty(n * len(HORIZONS), dtype=np.float64)
    for j, h in enumerate(HORIZONS):
        block = slice(j * n, (j + 1) * n)
        x[block, :-1] = matrix.x[indices]
        x[block, -1] = h
        y[block] = matrix.y[indices, j]
    return x, y


def append_horizon(features, horizon_minutes):
    import numpy as np
    if horizon_minutes not in HORIZONS or features.ndim != 2 or features.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("Only trained horizons and the ordered 41-feature matrix are supported")
    x = np.empty((len(features), len(FEATURE_ORDER)), dtype=np.float32)
    x[:, :-1] = features
    x[:, -1] = horizon_minutes
    return x


def _peak_working_set_mib():
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = wintypes.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    return counters.PeakWorkingSetSize / 2**20 if get_info(current(), ctypes.byref(counters), counters.cb) else None


def _source_fingerprints():
    direct = DIRECT_DIR / "results.json"
    leader = ROOT / "data" / "models" / "forecasting_leaderboard.json"
    return {"direct_results_sha256": sha256(direct.read_bytes()).hexdigest(),
            "one_hour_leaderboard_sha256": sha256(leader.read_bytes()).hexdigest()}


def _params():
    # xgb.train uses num_boost_round outside params; all tree/loss settings match V2-A.
    return {k: v for k, v in {**PARAMETERS, "objective": "reg:absoluteerror"}.items()
            if k not in ("n_estimators", "early_stopping_rounds")}


def write_design(output_dir=OUTPUT_DIR):
    manifest = json.loads((TARGET_DIR / "manifest.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    design = {"schema_version": 1, "horizons_minutes": list(HORIZONS),
              "feature_order": list(FEATURE_ORDER), "feature_count": len(FEATURE_ORDER),
              "source_target_manifest_sha256": sha256((TARGET_DIR / "manifest.json").read_bytes()).hexdigest(),
              "source_fingerprints": _source_fingerprints(),
              "common_counts": manifest["common_counts"],
              "memory_estimate": memory_estimate(manifest["common_counts"]),
              "training_rows": "ten horizon-major blocks of common training origins",
              "tuning_rows": "ten horizon-major blocks of common January-June 2025 tuning origins",
              "training_objective": "reg:absoluteerror",
              "selection_metric": "arithmetic mean of ten per-horizon tuning MAEs on common origins",
              "num_boost_round": PARAMETERS["n_estimators"],
              "early_stopping_rounds": PARAMETERS["early_stopping_rounds"],
              "parameters": _params(),
              "fixed_semantics": "Frozen price_target_aligned_previous_day/week inputs remain +60-minute aligned, even at other horizons.",
              "held_out": "December 2025-February 2026 untouched"}
    _write_json_atomic(output_dir / "design.json", design)
    return design


def train_shared(output_dir=OUTPUT_DIR):
    """Train a single direct L1 tree ensemble; stop on balanced tuning MAE."""
    import xgboost as xgb
    design = write_design(output_dir)
    matrices, manifest = load_matrices()
    if manifest["common_counts"] != design["common_counts"]:
        raise ValueError("Target counts changed after design")
    gc.collect()
    print("Stacking common train/tune rows", flush=True)
    x_train, y_train = stack_common(matrices["train"])
    x_tune, y_tune = stack_common(matrices["tune"])
    after_stack_peak = _peak_working_set_mib()
    print(f"Stacked {len(y_train):,} training and {len(y_tune):,} tuning examples", flush=True)
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, feature_names=list(FEATURE_ORDER))
    dtune = xgb.QuantileDMatrix(x_tune, label=y_tune, feature_names=list(FEATURE_ORDER), ref=dtrain)
    del x_train, x_tune, y_train, y_tune
    gc.collect()
    after_quantile_peak = _peak_working_set_mib()
    curves = {}
    start = perf_counter()
    booster = xgb.train(_params(), dtrain, num_boost_round=PARAMETERS["n_estimators"],
                        evals=[(dtrain, "train"), (dtune, "tune")],
                        custom_metric=xgb_balanced_metric,
                        early_stopping_rounds=PARAMETERS["early_stopping_rounds"],
                        evals_result=curves, verbose_eval=False)
    fit_seconds = perf_counter() - start
    after_fit_peak = _peak_working_set_mib()
    model_path = output_dir / "model.json"
    booster.save_model(model_path)
    metadata = {"feature_order": list(FEATURE_ORDER), "horizons_minutes": list(HORIZONS),
                "parameters": _params(), "num_boost_round": PARAMETERS["n_estimators"],
                "early_stopping_rounds": PARAMETERS["early_stopping_rounds"],
                "selection_metric": SELECTED_METRIC,
                "best_iteration_zero_based": booster.best_iteration,
                "best_tune_mean_mae": float(booster.best_score),
                "fitted_rounds": len(curves["tune"][SELECTED_METRIC]),
                "fit_seconds": fit_seconds, "model_bytes": model_path.stat().st_size,
                "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
                "process_peak_working_set_mib": after_fit_peak,
                "process_peak_after_stack_mib": after_stack_peak,
                "process_peak_after_quantile_mib": after_quantile_peak,
                "source_fingerprints": design["source_fingerprints"],
                "target_manifest_sha256": design["source_target_manifest_sha256"],
                "python": platform.python_version(), "xgboost": xgb.__version__,
                "learning_curves": curves}
    _write_json_atomic(output_dir / "metadata.json", metadata)
    print(f"Best iteration {booster.best_iteration}, balanced tune MAE {booster.best_score}, fit {fit_seconds:.1f}s", flush=True)
    return metadata


def load_saved(model_path=OUTPUT_DIR / "model.json", metadata_path=OUTPUT_DIR / "metadata.json"):
    import xgboost as xgb
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if tuple(metadata["feature_order"]) != FEATURE_ORDER or tuple(metadata["horizons_minutes"]) != HORIZONS:
        raise ValueError("Shared model feature/horizon schema mismatch")
    if sha256(Path(model_path).read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Shared model checksum mismatch")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster, metadata


def predict_saved(features, feature_names, horizon_minutes, *, booster=None, metadata=None):
    import xgboost as xgb
    if booster is None or metadata is None:
        booster, metadata = load_saved()
    if tuple(feature_names) != FEATURE_FIELDS or tuple(metadata["feature_order"]) != FEATURE_ORDER:
        raise ValueError("Shared model feature order differs")
    x = append_horizon(features, horizon_minutes)
    d = xgb.DMatrix(x, feature_names=list(FEATURE_ORDER))
    return booster.predict(d, iteration_range=(0, metadata["best_iteration_zero_based"] + 1))


def _direct_predictions(direct_result, matrix, part, col, h):
    """Read frozen direct predictions on exactly the same eligible origin rows."""
    import numpy as np
    record = direct_result["horizons"][str(h)]["splits"][part]
    file_info = record["predictions_file"]
    path = ROOT / file_info["path"]
    if sha256(path.read_bytes()).hexdigest() != file_info["sha256"]:
        raise ValueError("Frozen direct prediction checksum mismatch")
    eligible = np.isfinite(matrix.y[:, col])
    indices = np.flatnonzero(eligible)
    forecasts = {name: np.empty(len(indices), dtype=np.float64)
                 for name in ("xgboost", "persistence", "daily_seasonal",
                              "train_median", "hour_of_week_median")}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        j = -1
        for j, (i, row) in enumerate(zip(indices, rows)):
            origin = matrix.rows[i]["decision_start_utc"]
            target = (_utc(origin) + timedelta(minutes=h)).isoformat().replace("+00:00", "Z")
            if (row["decision_start_utc"] != origin or
                    row["target_interval_start_utc"] != target or
                    abs(float(row["actual_aud_per_kwh"]) - matrix.y[i, col]) > 1e-10):
                raise ValueError("Direct prediction origin, target or price mismatch")
            for name in forecasts:
                forecasts[name][j] = float(row[f"{name}_aud_per_kwh"])
        if j + 1 != len(indices) or next(rows, None) is not None:
            raise ValueError("Direct prediction count mismatch")
    return eligible, forecasts


def _combined_forecasts(direct_forecasts, shared, choice):
    import numpy as np
    result = {"persistence": direct_forecasts["persistence"],
              "daily_seasonal": direct_forecasts["daily_seasonal"],
              "train_median": direct_forecasts["train_median"],
              "hour_of_week_median": direct_forecasts["hour_of_week_median"],
              "direct": direct_forecasts["xgboost"], "shared": np.asarray(shared)}
    result["hybrid"] = result[choice]
    if tuple(result) != FORECAST_NAMES:
        raise ValueError("Comparison forecast order changed")
    return result


def _common_forecasts(forecasts, eligible, common):
    within = common[eligible]
    return {name: values[within] for name, values in forecasts.items()}


def _write_predictions(path, matrix, col, eligible, forecasts, h):
    from .multi_horizon import _writer
    temporary = path.with_suffix(path.suffix + ".tmp")
    binary, compressed, stream = _writer(temporary)
    try:
        fields = ("decision_start_utc", "target_interval_start_utc",
                  "actual_aud_per_kwh", *(f"{name}_aud_per_kwh" for name in FORECAST_NAMES))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for j, i in enumerate(index for index, keep in enumerate(eligible) if keep):
            origin = matrix.rows[i]["decision_start_utc"]
            target = (_utc(origin) + timedelta(minutes=h)).isoformat().replace("+00:00", "Z")
            writer.writerow({"decision_start_utc": origin,
                             "target_interval_start_utc": target,
                             "actual_aud_per_kwh": matrix.y[i, col],
                             **{f"{name}_aud_per_kwh": value[j]
                                for name, value in forecasts.items()}})
    finally:
        stream.close(); compressed.close(); binary.close()
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def _gain_groups(booster, names):
    """Relative total split gain; descriptive, not causal importance."""
    scores = booster.get_score(importance_type="total_gain")
    values = {name: float(scores.get(name, scores.get(f"f{i}", 0.0)))
              for i, name in enumerate(names)}
    total = sum(values.values())
    groups = {"price_history": [], "regional_demand": [], "calendar": [], "horizon": []}
    for name in names:
        group = ("horizon" if name == "forecast_horizon_minutes" else
                 "regional_demand" if name.startswith("demand_") else
                 "price_history" if name.startswith("price_") else "calendar")
        groups[group].append(name)
    return {"total_gain_by_feature": values,
            "fraction_by_group": {group: sum(values[name] for name in members) / total
                                  if total else 0.0 for group, members in groups.items()},
            "top_ten": sorted(values, key=values.get, reverse=True)[:10],
            "horizon_gain_rank": (sorted(values, key=values.get, reverse=True).index("forecast_horizon_minutes") + 1
                                  if "forecast_horizon_minutes" in names else None)}


def _all_importances(shared):
    import xgboost as xgb
    direct = {}
    for h in HORIZONS:
        booster = xgb.Booster()
        booster.load_model(DIRECT_DIR / f"h{h:03d}" / "model.json")
        direct[str(h)] = _gain_groups(booster, FEATURE_FIELDS)
    return {"shared": _gain_groups(shared, FEATURE_ORDER), "direct": direct,
            "importance_type": "total_gain; fraction of total gain in that model"}


def _single_decision_latency(shared, metadata, matrix, samples=100):
    """Warm model, then time all ten predictions from one prepared feature row."""
    import numpy as np
    import xgboost as xgb
    indices = np.flatnonzero(matrix.common)
    positions = np.linspace(0, len(indices) - 1, samples, dtype=int)
    direct_models = []
    for h in HORIZONS:
        model = xgb.XGBRegressor()
        model.load_model(DIRECT_DIR / f"h{h:03d}" / "model.json")
        direct_models.append(model)
    def direct_predict(row):
        return [float(model.predict(row)[0]) for model in direct_models]
    def shared_predict(row):
        x = np.empty((len(HORIZONS), len(FEATURE_ORDER)), dtype=np.float32)
        x[:, :-1] = row[0]
        x[:, -1] = HORIZONS
        d = xgb.DMatrix(x, feature_names=list(FEATURE_ORDER))
        return shared.predict(d, iteration_range=(0, metadata["best_iteration_zero_based"] + 1))
    warm = matrix.x[indices[positions[0]]:indices[positions[0]] + 1]
    direct_predict(warm); shared_predict(warm)
    times = {"direct_ten_models": [], "shared_one_model": []}
    for pos in positions:
        row = matrix.x[indices[pos]:indices[pos] + 1]
        start = perf_counter(); direct_predict(row)
        times["direct_ten_models"].append(perf_counter() - start)
        start = perf_counter(); shared_predict(row)
        times["shared_one_model"].append(perf_counter() - start)
    return {name: {"median_ms": float(np.median(values) * 1000),
                   "p95_ms": float(np.percentile(values, 95) * 1000),
                   "samples": len(values)} for name, values in times.items()}


def evaluate_shared(output_dir=OUTPUT_DIR):
    """Freeze a tuning-only hybrid before any reused evaluation predictions."""
    import numpy as np
    booster, metadata = load_saved(output_dir / "model.json", output_dir / "metadata.json")
    design = json.loads((output_dir / "design.json").read_text(encoding="utf-8"))
    if _source_fingerprints() != design["source_fingerprints"]:
        raise ValueError("Frozen direct results or one-hour leaderboard changed")
    matrices, manifest = load_matrices()
    if manifest["common_counts"] != design["common_counts"]:
        raise ValueError("Origin counts changed after training")
    direct_result = json.loads((DIRECT_DIR / "results.json").read_text(encoding="utf-8"))
    result = {"schema_version": 1, "experiment": "shared horizon-conditioned XGBoost V2-A",
              "horizons_minutes": list(HORIZONS), "feature_order": list(FEATURE_ORDER),
              "train_common_examples": design["memory_estimate"]["train"]["examples"],
              "tune_common_examples": design["memory_estimate"]["tune"]["examples"],
              "target_manifest": manifest, "model_metadata": metadata,
              "source_fingerprints": design["source_fingerprints"],
              "evaluation_status": "July-November 2025 reused development evaluation",
              "held_out_status": "December 2025-February 2026 untouched",
              "horizons": {}, "mean_common_mae": {}}
    # All selection inputs come from common January-June tuning origins.
    tune = matrices["tune"]
    selection = {}
    tune_cache = {}
    for col, h in enumerate(HORIZONS):
        eligible, direct = _direct_predictions(direct_result, tune, "tune", col, h)
        start = perf_counter()
        shared = predict_saved(tune.x[eligible], FEATURE_FIELDS, h,
                               booster=booster, metadata=metadata)
        inference = perf_counter() - start
        within = tune.common[eligible]
        actual = tune.y[tune.common, col]
        direct_tune_mae = float(np.mean(np.abs(direct["xgboost"][within] - actual)))
        old_mae = direct_result["horizons"][str(h)]["splits"]["tune"]["common"]["models"]["xgboost"]["mae"]
        if abs(direct_tune_mae - old_mae) > 1e-9:
            raise ValueError("Direct tuning comparison changed")
        candidate_mae = {"persistence": float(np.mean(np.abs(direct["persistence"][within] - actual))),
                         "direct": direct_tune_mae,
                         "shared": float(np.mean(np.abs(shared[within] - actual)))}
        choice = min(HYBRID_CANDIDATES, key=lambda name: candidate_mae[name])
        selection[str(h)] = {"chosen": choice, "common_tune_mae": candidate_mae,
                             "common_tune_count": int(tune.common.sum())}
        tune_cache[h] = (eligible, direct, shared, inference)
    selection_record = {"selection_rule": "lowest common-origin January-June 2025 tuning MAE among persistence, direct, shared; ties prefer persistence then direct then shared",
                        "frozen_before_reused_evaluation": True, "horizons": selection,
                        "source_fingerprints": design["source_fingerprints"]}
    _write_json_atomic(output_dir / "hybrid_selection.json", selection_record)
    result["hybrid_selection"] = selection_record
    for part in ("tune", "evaluate"):
        portion = matrices[part]
        for col, h in enumerate(HORIZONS):
            if part == "tune":
                eligible, direct, shared, inference = tune_cache[h]
            else:
                eligible, direct = _direct_predictions(direct_result, portion, part, col, h)
                start = perf_counter()
                shared = predict_saved(portion.x[eligible], FEATURE_FIELDS, h,
                                       booster=booster, metadata=metadata)
                inference = perf_counter() - start
            forecasts = _combined_forecasts(direct, shared, selection[str(h)]["chosen"])
            common_forecasts = _common_forecasts(forecasts, eligible, portion.common)
            record = result["horizons"].setdefault(str(h), {"splits": {}})
            record["splits"][part] = {
                "eligible": _scores(matrices, part, col, eligible, forecasts),
                "common": _scores(matrices, part, col, eligible & portion.common, common_forecasts),
                "shared_batch_inference_seconds": inference,
                "prediction_file": _write_predictions(
                    output_dir / f"h{h:03d}_{part}_predictions.csv.gz",
                    portion, col, eligible, forecasts, h)}
            print(f"Scored {part} +{h}m", flush=True)
    for part in ("tune", "evaluate"):
        result["mean_common_mae"][part] = {
            name: float(mean(result["horizons"][str(h)]["splits"][part]["common"]["models"][name]["mae"]
                             for h in HORIZONS)) for name in FORECAST_NAMES}
    if abs(result["mean_common_mae"]["tune"]["shared"] - metadata["best_tune_mean_mae"]) > 1e-5:
        raise ValueError("Selected balanced tuning metric does not match saved inference")
    result["importance"] = _all_importances(booster)
    result["model_size_bytes"] = {"shared": metadata["model_bytes"],
                                  "direct_ten_total": sum((DIRECT_DIR / f"h{h:03d}" / "model.json").stat().st_size
                                                          for h in HORIZONS)}
    result["single_decision_inference"] = _single_decision_latency(
        booster, metadata, matrices["evaluate"])
    result["process_peak_working_set_after_evaluation_mib"] = _peak_working_set_mib()
    if _source_fingerprints() != design["source_fingerprints"]:
        raise ValueError("Frozen comparison artifact changed during evaluation")
    _write_json_atomic(output_dir / "results.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "train", "evaluate"))
    args = parser.parse_args()
    if args.stage == "design":
        print(json.dumps(write_design(), indent=2))
    elif args.stage == "train":
        metadata = train_shared()
        print(json.dumps({key: metadata[key] for key in (
            "best_iteration_zero_based", "best_tune_mean_mae", "fit_seconds",
            "process_peak_working_set_mib")}, indent=2))
    else:
        result = evaluate_shared()
        print(json.dumps(result["mean_common_mae"], indent=2))
