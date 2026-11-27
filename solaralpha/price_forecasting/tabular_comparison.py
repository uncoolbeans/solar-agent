"""Offline, matched one-hour price comparison; never reads the held-out summer split."""

import csv
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
import platform
from time import perf_counter

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .ml_dataset import load_slices
from .ml_metrics import full_scores, grouped_scores, movement_diagnostics, numeric_scores
from .xgb_v2 import _conditional_scores, _subperiod_scores
from .xgb_v2_diagnostics import V2_DIR


OUTPUT_DIR = ROOT / "data" / "models" / "price_tabular_comparison"
LEADERBOARD_PATH = ROOT / "data" / "models" / "forecasting_leaderboard.json"
NAMES = ("ridge", "lasso", "random_forest", "lightgbm", "catboost")
ORIGINAL_NAMES = ("V1", "V2-A", "V2-B", "V2-C", "V2-D", "persistence",
                  "daily_seasonal", "train_median", "hour_of_week_median")
ALL_NAMES = ORIGINAL_NAMES + NAMES
CONFIG = {
    "ridge": {"alpha_candidates": [0.1, 1.0, 10.0], "objective": "squared_error_l2"},
    "lasso": {"alpha_candidates": [0.0001, 0.001, 0.01],
              "objective": "squared_error_l1_penalty", "max_iter": 5000, "tol": 1e-4},
    "random_forest": {"n_estimators": 50, "max_depth": 12,
                      "min_samples_leaf": 20, "max_features": 0.7,
                      "criterion": "squared_error", "n_jobs": 1, "random_state": 42},
    "lightgbm": {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 31,
                 "min_child_samples": 30, "objective": "regression_l1", "metric": "l1",
                 "n_jobs": 4, "random_state": 42, "verbosity": -1},
    "catboost": {"iterations": 800, "depth": 6, "learning_rate": 0.05,
                 "loss_function": "MAE", "eval_metric": "MAE", "l2_leaf_reg": 3.0,
                 "thread_count": 4, "random_seed": 42, "verbose": False,
                 "allow_writing_files": False},
}


def assert_schema(matrix, names):
    if tuple(names) != FEATURE_FIELDS or matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("Tabular feature order or matrix width differs from training")


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_original_predictions(split, portion):
    """Align every frozen V1/V2/baseline forecast by origin, target and actual price."""
    import numpy as np
    path = V2_DIR / f"{split}_comparison.csv.gz"
    values = {name: np.empty(len(portion.y), dtype=float) for name in ORIGINAL_NAMES}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {f"{name}_aud_per_kwh" for name in ORIGINAL_NAMES}
        if not expected.issubset(reader.fieldnames):
            raise ValueError("Frozen V2 comparison columns changed")
        count = 0
        for count, (record, row, actual) in enumerate(
                zip(reader, portion.rows, portion.y, strict=True), 1):
            if (record["decision_start_utc"] != row["decision_start_utc"] or
                    record["target_interval_start_utc"] != row["target_interval_start_utc"] or
                    float(record["actual_aud_per_kwh"]) != actual):
                raise ValueError("Frozen forecast rows or target values differ")
            for name in ORIGINAL_NAMES:
                values[name][count - 1] = float(record[f"{name}_aud_per_kwh"])
        if count != len(portion.y):
            raise ValueError("Frozen forecast row count differs")
    return values


def _fit_linear(name, dataset):
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    spec = CONFIG[name]
    candidates = []
    elapsed = 0.0
    for alpha in spec["alpha_candidates"]:
        estimator = Ridge(alpha=alpha) if name == "ridge" else Lasso(
            alpha=alpha, max_iter=spec["max_iter"], tol=spec["tol"], selection="cyclic")
        model = make_pipeline(StandardScaler(), estimator)
        start = perf_counter()
        model.fit(dataset.train.x, dataset.train.y)
        elapsed += perf_counter() - start
        score = numeric_scores(dataset.tune.y, model.predict(dataset.tune.x))["mae"]
        candidates.append((score, alpha, model))
    score, alpha, selected = min(candidates, key=lambda item: (item[0], item[1]))
    scaler = selected.named_steps["standardscaler"]
    if not np.allclose(scaler.mean_, np.mean(dataset.train.x, axis=0, dtype=float),
                       rtol=1e-5, atol=1e-4):
        raise ValueError("Scaler was not fitted solely to training rows")
    return selected, elapsed, {"selected_alpha": alpha, "tuning_mae": score,
                               "candidate_tuning_mae": {str(a): s for s, a, _ in candidates},
                               "scaler_mean": scaler.mean_.tolist(),
                               "scaler_scale": scaler.scale_.tolist()}


def _fit_tree(name, dataset):
    start = perf_counter()
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**CONFIG[name])
        model.fit(dataset.train.x, dataset.train.y)
        detail = {"selection": "one fixed configuration; tuning MAE reported"}
    elif name == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMRegressor(**CONFIG[name])
        model.fit(dataset.train.x, dataset.train.y,
                  eval_X=dataset.tune.x, eval_y=dataset.tune.y,
                  eval_metric="l1", callbacks=[lgb.early_stopping(50, verbose=False)])
        detail = {"best_iteration": int(model.best_iteration_),
                  "fitted_rounds": len(model.evals_result_["valid_0"]["l1"]),
                  "tuning_curve": model.evals_result_["valid_0"]["l1"]}
    elif name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor(**CONFIG[name])
        model.fit(dataset.train.x, dataset.train.y,
                  eval_set=(dataset.tune.x, dataset.tune.y),
                  early_stopping_rounds=50, use_best_model=True)
        detail = {"best_iteration_zero_based": int(model.best_iteration_),
                  "fitted_rounds": int(model.tree_count_),
                  "tuning_curve": model.evals_result_["validation"]["MAE"]}
    else:
        raise ValueError(name)
    elapsed = perf_counter() - start
    detail["tuning_mae"] = numeric_scores(dataset.tune.y,
                                           model.predict(dataset.tune.x))["mae"]
    return model, elapsed, detail


def _save_model(name, model, path):
    if name in ("ridge", "lasso", "random_forest"):
        import joblib
        path = path / "model.joblib"
        joblib.dump(model, path, compress=3)
    elif name == "lightgbm":
        path = path / "model.txt"
        model.booster_.save_model(str(path), num_iteration=model.best_iteration_)
    elif name == "catboost":
        path = path / "model.cbm"
        model.save_model(str(path))
    return path


def predict_saved(name, model_path, metadata_path, matrix, feature_names):
    assert_schema(matrix, feature_names)
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if meta["name"] != name or tuple(meta["feature_order"]) != FEATURE_FIELDS:
        raise ValueError("Saved model schema or identity mismatch")
    if _hash(model_path) != meta["model_sha256"]:
        raise ValueError("Saved model checksum mismatch")
    if name in ("ridge", "lasso", "random_forest"):
        import joblib
        model = joblib.load(model_path)
    elif name == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_file=str(model_path))
    elif name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(model_path))
    else:
        raise ValueError(name)
    return model.predict(matrix)


def _timed_inference(model, matrix):
    import numpy as np
    # Warm both paths, then measure batch throughput and repeated one-row latency.
    model.predict(matrix[:1])
    start = perf_counter()
    batch = model.predict(matrix)
    batch_seconds = perf_counter() - start
    one = matrix[0:1]
    samples = []
    for _ in range(50):
        start = perf_counter()
        model.predict(one)
        samples.append(perf_counter() - start)
    return np.asarray(batch), {"batch_seconds": batch_seconds,
                               "batch_microseconds_per_row": 1e6 * batch_seconds / len(matrix),
                               "single_observation_median_microseconds":
                                   1e6 * float(np.median(samples)),
                               "single_observation_repeats": len(samples)}


def benchmark_frozen(dataset):
    """Time preserved XGBoost artifacts and deterministic baselines without refitting."""
    import numpy as np
    import xgboost as xgb
    from .xgb_v1 import baseline_predictions
    from .xgb_v2 import LATEST_INDEX

    class AbsoluteXgb:
        def __init__(self, path, residual):
            self.model = xgb.XGBRegressor()
            self.model.load_model(path)
            self.residual = residual

        def predict(self, matrix):
            raw = self.model.predict(matrix)
            return raw + matrix[:, LATEST_INDEX].astype(float) if self.residual else raw

    frozen = read_original_predictions("evaluate", dataset.evaluate)
    result = {}
    for name in ORIGINAL_NAMES[:5]:
        path = (ROOT / "data/models/price_xgb_v1/model.json" if name == "V1" else
                V2_DIR / name / "model.json")
        model = AbsoluteXgb(path, name in ("V2-C", "V2-D"))
        predicted, timing = _timed_inference(model, dataset.evaluate.x)
        if not np.allclose(predicted[:20], frozen[name][:20], rtol=1e-5, atol=1e-6):
            raise ValueError(f"Frozen {name} saved model differs from matched forecasts")
        meta_path = path.with_name("metadata.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        result[name] = {"training_seconds": meta.get("train_seconds", meta.get("training_seconds")),
                        "inference": {"evaluate": timing}, "model_bytes": path.stat().st_size,
                        "model_sha256": _hash(path), "artifact": path.relative_to(ROOT).as_posix()}

    # Rule-only baselines have no saved model. Their costs are measured from
    # already prepared row metadata, which is comparable only as model logic.
    baseline_names = ORIGINAL_NAMES[5:]
    def make_baselines():
        return baseline_predictions(dataset, dataset.evaluate)
    make_baselines()
    start = perf_counter()
    values = make_baselines()
    batch_seconds = perf_counter() - start
    one_row = dataset.evaluate.rows[0]
    for name in baseline_names:
        samples = []
        for _ in range(50):
            start = perf_counter()
            if name == "persistence":
                value = one_row["persistence"]
            elif name == "daily_seasonal":
                value = one_row["daily_seasonal"]
            elif name == "train_median":
                value = dataset.constant_median
            else:
                value = dataset.hour_of_week_medians[one_row["target_hour_of_week"]]
            samples.append(perf_counter() - start)
        if not np.allclose(values[name], frozen[name], rtol=0, atol=1e-12):
            raise ValueError(f"Frozen {name} baseline differs")
        result[name] = {"training_seconds": None, "inference": {"evaluate": {
            "batch_seconds": batch_seconds,
            "batch_microseconds_per_row": 1e6 * batch_seconds / len(dataset.evaluate.y),
            "single_observation_median_microseconds": 1e6 * float(np.median(samples)),
            "single_observation_repeats": len(samples)}},
            "model_bytes": None, "model_sha256": None, "artifact": None,
            "inference_note": "Batch timing makes all four baseline arrays together"}
    return result


def _write_predictions(path, portion, forecasts):
    fields = ("decision_start_utc", "target_interval_start_utc", "origin_local_date",
              "actual_aud_per_kwh", *(f"{name}_aud_per_kwh" for name in ALL_NAMES))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i, row in enumerate(portion.rows):
                    writer.writerow({"decision_start_utc": row["decision_start_utc"],
                                     "target_interval_start_utc": row["target_interval_start_utc"],
                                     "origin_local_date": row["origin_local_date"],
                                     "actual_aud_per_kwh": portion.y[i],
                                     **{f"{name}_aud_per_kwh": values[i]
                                        for name, values in forecasts.items()}})
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _hash(path),
            "bytes": path.stat().st_size}


def build_leaderboard(result):
    """A versioned, extensible row per model; absent future results are explicit nulls."""
    v2 = json.loads((V2_DIR / "v2_results.json").read_text(encoding="utf-8"))
    if v2["feature_files"] != result["feature_files"]:
        raise ValueError("XGBoost and tabular data fingerprints differ")
    xgb_loss = {"V1": "squared_error", "V2-A": "absolute_error",
                "V2-B": "pseudo_huber", "V2-C": "squared_error",
                "V2-D": "absolute_error"}
    baseline_loss = "none_deterministic_rule"
    entries = []
    for name in ALL_NAMES:
        if name in NAMES:
            detail = result["models"][name]
            family = name
            config = detail["configuration"]
            loss = (CONFIG[name].get("objective") or
                    ("squared_error_split_averaging" if name == "random_forest" else "MAE"))
            runtime_source = {"training_seconds": detail["training_seconds"],
                              "inference": detail["inference"]}
            artifact = detail["artifact"]
            model_bytes = detail["model_bytes"]
        elif name.startswith("V"):
            family = "xgboost"
            spec = v2["variants"].get(name)
            config = spec["parameters"] if spec else json.loads(
                (ROOT / "data/models/price_xgb_v1/metadata.json").read_text(encoding="utf-8"))["parameters"]
            loss = xgb_loss[name]
            runtime_source = result["frozen_runtime"][name]
            artifact = (f"data/models/price_xgb_v2/{name}/model.json" if spec else
                        "data/models/price_xgb_v1/model.json")
            model_bytes = (ROOT / artifact).stat().st_size
        else:
            family = "baseline"
            config = {"rule": name}
            loss = baseline_loss
            runtime_source = result["frozen_runtime"][name]
            artifact = None
            model_bytes = None
        timing = runtime_source["inference"]["evaluate"]
        runtime = {"training_seconds": runtime_source["training_seconds"],
                   "evaluation_batch_seconds": timing["batch_seconds"],
                   "evaluation_batch_microseconds_per_row":
                       timing["batch_microseconds_per_row"],
                   "evaluation_single_median_microseconds":
                       timing["single_observation_median_microseconds"],
                   "single_observation_repeats": timing["single_observation_repeats"],
                   "scope": "model logic only; prepared features/metadata; no startup"}
        if family == "baseline":
            runtime["note"] = runtime_source["inference_note"]
        entries.append({"name": name, "family": family, "target_type":
                        "residual_vs_latest_completed_price" if name in ("V2-C", "V2-D")
                        else "absolute_wholesale_price", "loss_function": loss,
                        "configuration": config, "tuning": result["splits"]["tune"]["models"][name],
                        "reused_development_evaluation":
                            result["splits"]["evaluate"]["models"][name],
                        "runtime": runtime, "model_artifact": artifact,
                        "model_bytes": model_bytes, "multi_horizon": None,
                        "downstream_mpc_financial_evaluation": None})
    return {"schema_version": 1, "forecast_horizon_minutes": 60,
            "target": "NSW1 wholesale AUD/kWh for interval starting decision+60min",
            "feature_order": list(FEATURE_FIELDS), "feature_files": result["feature_files"],
            "split_counts": result["counts"],
            "splits": {"train": "2023-01-01..2024-12-31",
                       "tune": "2025-01-01..2025-06-30, development week excluded",
                       "reused_development_evaluation":
                           "2025-07-01..2025-11-30, not independent test",
                       "held_out_status": "2025-12-01..2026-02-28 untouched"},
            "model_selection": "Jan-Jun 2025 MAE only", "entries": entries}


def run(output_dir=OUTPUT_DIR, leaderboard_path=LEADERBOARD_PATH):
    import numpy as np
    import sklearn
    import lightgbm
    import catboost

    dataset = load_slices()
    if dataset.counts["train"] != 208800 or dataset.counts["tune"] != 50100 or dataset.counts["evaluate"] != 44040:
        raise ValueError("Unexpected chronological split counts")
    for portion in (dataset.train, dataset.tune, dataset.evaluate):
        assert_schema(portion.x, FEATURE_FIELDS)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {"tune": {}, "evaluate": {}}
    details = {}
    for name in NAMES:
        print(f"Fitting {name}...", flush=True)
        model, train_seconds, fit_detail = (
            _fit_linear(name, dataset) if name in ("ridge", "lasso")
            else _fit_tree(name, dataset))
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        model_path = _save_model(name, model, path)
        meta = {"name": name, "feature_order": list(FEATURE_FIELDS),
                "feature_files": dataset.feature_report["files"],
                "configuration": CONFIG[name], "fit_detail": fit_detail,
                "python": platform.python_version(),
                "versions": {"numpy": np.__version__, "scikit_learn": sklearn.__version__,
                             "lightgbm": lightgbm.__version__, "catboost": catboost.__version__},
                "model_sha256": _hash(model_path), "training_seconds": train_seconds}
        _write_json_atomic(path / "metadata.json", meta)
        tune_pred, tune_timing = _timed_inference(model, dataset.tune.x)
        saved = predict_saved(name, model_path, path / "metadata.json",
                              dataset.tune.x[:12], FEATURE_FIELDS)
        if not np.allclose(saved, tune_pred[:12], rtol=1e-5, atol=1e-7):
            raise ValueError(f"Reloaded {name} predictions differ")
        predictions["tune"][name] = tune_pred
        details[name] = {"configuration": CONFIG[name], "fit": fit_detail,
                         "training_seconds": train_seconds,
                         "artifact": model_path.relative_to(ROOT).as_posix(),
                         "model_sha256": _hash(model_path),
                         "model_bytes": model_path.stat().st_size,
                         "inference": {"tune": tune_timing},
                         "reloaded_prediction_max_abs_difference":
                             float(np.max(np.abs(saved - tune_pred[:12])))}
        # July-November is inspected only after all training and tuning decisions.
        print(f"  tuning MAE {numeric_scores(dataset.tune.y, tune_pred)['mae']:.5f}", flush=True)
        predictions["evaluate"][name], details[name]["inference"]["evaluate"] = (
            _timed_inference(model, dataset.evaluate.x))
        if name in ("ridge", "lasso"):
            coefficients = model.named_steps[name].coef_
            importance = {field: float(value) for field, value in zip(FEATURE_FIELDS, coefficients)}
        elif name == "lightgbm":
            importance = {field: float(value) for field, value in zip(
                FEATURE_FIELDS, model.booster_.feature_importance(importance_type="gain"))}
        elif name == "catboost":
            importance = {field: float(value) for field, value in zip(
                FEATURE_FIELDS, model.feature_importances_)}
        else:
            importance = {field: float(value) for field, value in zip(
                FEATURE_FIELDS, model.feature_importances_)}
        details[name]["feature_importance"] = importance
    result = {"experiment": "Alternative tabular one-hour price forecasting",
              "status": "July-November reused development evaluation, not independent test",
              "held_out_status": "December 2025-February 2026 untouched",
              "selection_rule": "training fixed configurations and small linear alpha search by Jan-Jun MAE only",
              "feature_files": dataset.feature_report["files"],
              "counts": dataset.counts, "models": details,
              "frozen_runtime": benchmark_frozen(dataset), "splits": {}}
    for split in ("tune", "evaluate"):
        portion = getattr(dataset, split)
        forecasts = {**read_original_predictions(split, portion), **predictions[split]}
        if tuple(forecasts) != ALL_NAMES:
            raise ValueError("Matched comparison model order changed")
        result["splits"][split] = {
            "row_count": len(portion.y),
            "models": {name: full_scores(portion.y, values)
                       for name, values in forecasts.items()},
            "actual_price_ranges": _conditional_scores(portion.y, forecasts),
            "monthly": grouped_scores(portion.y, forecasts, portion.rows, "target_local_month"),
            "hour_of_day": grouped_scores(portion.y, forecasts, portion.rows, "target_local_hour"),
            "subperiods": _subperiod_scores(portion, forecasts),
            "movement": movement_diagnostics(portion.y, forecasts, portion.rows),
            "predictions_file": _write_predictions(
                output_dir / f"{split}_comparison.csv.gz", portion, forecasts)}
    _write_json_atomic(output_dir / "results.json", result)
    _write_json_atomic(leaderboard_path, build_leaderboard(result))
    return result


if __name__ == "__main__":
    report = run()
    print(json.dumps({name: {"tune_mae": report["splits"]["tune"]["models"][name]["mae"],
                            "evaluation_mae": report["splits"]["evaluate"]["models"][name]["mae"]}
                      for name in ALL_NAMES}, indent=2))
