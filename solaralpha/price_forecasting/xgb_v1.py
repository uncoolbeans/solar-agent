"""Single fixed XGBRegressor experiment; no model is connected to trading agents."""

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
from .ml_metrics import full_scores, grouped_scores, movement_diagnostics


MODEL_DIR = ROOT / "data" / "models" / "price_xgb_v1"
PARAMETERS = {
    "n_estimators": 1000, "max_depth": 5, "learning_rate": 0.03,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 10, "reg_alpha": 0.1, "reg_lambda": 1.0,
    "objective": "reg:squarederror", "eval_metric": "mae",
    "early_stopping_rounds": 50, "tree_method": "hist", "random_state": 42,
    "n_jobs": 4,
}
PREDICTION_FIELDS = ("decision_start_utc", "target_interval_start_utc",
                     "origin_local_date", "actual_aud_per_kwh",
                     "xgboost_aud_per_kwh", "persistence_aud_per_kwh",
                     "daily_seasonal_aud_per_kwh", "train_median_aud_per_kwh",
                     "hour_of_week_median_aud_per_kwh")


def baseline_predictions(dataset, portion):
    import numpy as np
    return {
        "persistence": np.asarray([r["persistence"] for r in portion.rows]),
        "daily_seasonal": np.asarray([r["daily_seasonal"] for r in portion.rows]),
        "train_median": np.full(len(portion.y), dataset.constant_median),
        "hour_of_week_median": np.asarray([
            dataset.hour_of_week_medians[r["target_hour_of_week"]]
            for r in portion.rows]),
    }


def assert_feature_order(names):
    if tuple(names) != FEATURE_FIELDS:
        raise ValueError("Inference feature order differs from the saved V1 schema")


def predict_saved(model_path, metadata_path, matrix, names):
    import xgboost as xgb
    assert_feature_order(names)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if tuple(metadata["feature_order"]) != FEATURE_FIELDS:
        raise ValueError("Saved feature ordering does not match this code")
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("Inference matrix has wrong feature width")
    if sha256(Path(model_path).read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Saved model checksum mismatch")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model.predict(matrix)


def _write_predictions(path, portion, forecasts):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
                writer.writeheader()
                for i, row in enumerate(portion.rows):
                    writer.writerow({
                        "decision_start_utc": row["decision_start_utc"],
                        "target_interval_start_utc": row["target_interval_start_utc"],
                        "origin_local_date": row["origin_local_date"],
                        "actual_aud_per_kwh": portion.y[i],
                        **{f"{name}_aud_per_kwh": values[i]
                           for name, values in forecasts.items()},
                    })
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def train_and_evaluate(output_dir=MODEL_DIR):
    import numpy as np
    import scipy
    import sklearn
    import xgboost as xgb

    dataset = load_slices()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = xgb.XGBRegressor(**PARAMETERS)
    start = perf_counter()
    model.fit(dataset.train.x, dataset.train.y,
              eval_set=[(dataset.train.x, dataset.train.y),
                        (dataset.tune.x, dataset.tune.y)], verbose=False)
    training_seconds = perf_counter() - start
    model_path = output_dir / "model.json"
    model.save_model(model_path)
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "experiment": "XGBoost V1 one-hour NSW1 wholesale price",
        "python": platform.python_version(),
        "versions": {"xgboost": xgb.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "parameters": PARAMETERS, "feature_order": list(FEATURE_FIELDS),
        "target": "price of five-minute interval starting at decision time +60 min",
        "partitions": {"train": "2023-01-01..2024-12-31",
                       "tune_early_stop": "2025-01-01..2025-06-30",
                       "evaluate_once": "2025-07-01..2025-11-30",
                       "held_out_untouched": "2025-12-01..2026-02-28"},
        "counts": dataset.counts,
        "feature_files": dataset.feature_report["files"],
        "feature_build_report_sha256": sha256(
            (ROOT / "data/features/build_report.json").read_bytes()).hexdigest(),
        "training_only_constant_median": dataset.constant_median,
        "training_only_hour_of_week_medians": list(dataset.hour_of_week_medians),
        "best_iteration_zero_based": int(model.best_iteration),
        "best_score_tuning_mae": float(model.best_score),
        "actual_boosting_rounds": len(model.evals_result()["validation_1"]["mae"]),
        "training_seconds": training_seconds,
        "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
    }
    _write_json_atomic(metadata_path, metadata)

    results = {"model_sha256": metadata["model_sha256"],
               "best_iteration_zero_based": metadata["best_iteration_zero_based"],
               "training_seconds": training_seconds,
               "splits": {}, "gain_feature_importance": {},
               "learning_curve": model.evals_result()}
    # Early stopping leaves later trees in the saved booster. Importance must
    # describe the trees used by predict(), through best_iteration inclusive.
    selected_booster = model.get_booster()[:model.best_iteration + 1]
    raw_gain = selected_booster.get_score(importance_type="gain")
    results["gain_feature_importance"] = {
        name: float(raw_gain.get(f"f{i}", 0.0))
        for i, name in enumerate(FEATURE_FIELDS)}
    for part_name in ("tune", "evaluate"):
        portion = getattr(dataset, part_name)
        start = perf_counter()
        xgb_prediction = model.predict(portion.x)
        inference_seconds = perf_counter() - start
        forecasts = {"xgboost": xgb_prediction,
                     **baseline_predictions(dataset, portion)}
        result = {
            "row_count": len(portion.y),
            "inference_seconds": inference_seconds,
            "inference_microseconds_per_row": 1e6 * inference_seconds / len(portion.y),
            "models": {name: full_scores(portion.y, values)
                       for name, values in forecasts.items()},
            "monthly": grouped_scores(portion.y, forecasts, portion.rows,
                                      "target_local_month"),
            "hour_of_day": grouped_scores(portion.y, forecasts, portion.rows,
                                          "target_local_hour"),
            "movement": movement_diagnostics(portion.y, forecasts, portion.rows),
        }
        result["predictions_file"] = _write_predictions(
            output_dir / f"{part_name}_predictions.csv.gz", portion, forecasts)
        results["splits"][part_name] = result
    _write_json_atomic(output_dir / "results.json", results)
    return results


if __name__ == "__main__":
    report = train_and_evaluate()
    print(json.dumps({"best_iteration": report["best_iteration_zero_based"],
                      "training_seconds": report["training_seconds"],
                      "evaluation": report["splits"]["evaluate"]["models"]},
                     indent=2))
