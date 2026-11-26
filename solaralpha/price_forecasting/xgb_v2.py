"""Four fixed loss/target XGBoost comparisons; no trading-agent integration."""

import csv
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .ml_dataset import load_slices
from .ml_metrics import full_scores, grouped_scores, movement_diagnostics, numeric_scores
from .xgb_v1 import MODEL_DIR as V1_DIR, PARAMETERS as V1_PARAMETERS, baseline_predictions
from .xgb_v2_diagnostics import PRICE_RANGES, V2_DIR, _stored_predictions


VARIANTS = {
    "V2-A": {"objective": "reg:absoluteerror", "residual": False},
    "V2-B": {"objective": "reg:pseudohubererror", "residual": False,
             "huber_slope": 1.0},
    "V2-C": {"objective": "reg:squarederror", "residual": True},
    "V2-D": {"objective": "reg:absoluteerror", "residual": True},
}
MODEL_NAMES = ("V1", *VARIANTS, "persistence", "daily_seasonal",
               "train_median", "hour_of_week_median")
LATEST_INDEX = FEATURE_FIELDS.index("price_lag_5m_aud_per_kwh")


def residual_target(actual_price, latest_completed_price):
    """All numbers are AUD/kWh and the latest input is already completed."""
    return actual_price - latest_completed_price


def reconstruct_price(predicted_residual, latest_completed_price):
    return latest_completed_price + predicted_residual


def latest_from_features(portion):
    # Use the same float32 feature value for residual labels and reconstruction.
    return portion.x[:, LATEST_INDEX].astype("float64")


def labels_for_variant(portion, *, residual):
    return residual_target(portion.y, latest_from_features(portion)) if residual else portion.y


def prediction_to_price(raw_prediction, portion, *, residual):
    return reconstruct_price(raw_prediction, latest_from_features(portion)) if residual else raw_prediction


def _conditional_scores(actual, forecasts):
    import numpy as np
    a = np.asarray(actual)
    result = {}
    for name, low, high in PRICE_RANGES:
        mask = np.ones(len(a), dtype=bool)
        if low is not None:
            mask &= a >= low
        if high is not None:
            mask &= a < high
        result[name] = {model: numeric_scores(a[mask], np.asarray(values)[mask])
                        for model, values in forecasts.items()} if mask.any() else {}
    return result


def _subperiod_scores(portion, forecasts):
    import numpy as np
    months = [row["target_local_month"] for row in portion.rows]
    ranges = {"2025-01_to_03": ("2025-01", "2025-04"),
              "2025-04_to_06": ("2025-04", "2025-07"),
              "2025-07_to_09": ("2025-07", "2025-10"),
              "2025-10_to_11": ("2025-10", "2025-12")}
    result = {}
    for label, (start, end) in ranges.items():
        indices = [i for i, month in enumerate(months) if start <= month < end]
        if indices:
            result[label] = {name: numeric_scores(portion.y[indices],
                                                  np.asarray(values)[indices])
                             for name, values in forecasts.items()}
    return result


def _write_prediction_file(path, portion, forecasts):
    import gzip
    from io import TextIOWrapper
    fields = ("decision_start_utc", "target_interval_start_utc",
              "origin_local_date", "actual_aud_per_kwh",
              *(f"{name}_aud_per_kwh" for name in MODEL_NAMES))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
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


def predict_saved_variant(model_path, metadata_path, matrix, feature_names):
    """Load an offline model and reconstruct absolute prices if it predicts residuals."""
    import xgboost as xgb
    if tuple(feature_names) != FEATURE_FIELDS or matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("V2 feature order or width mismatch")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if tuple(metadata["feature_order"]) != FEATURE_FIELDS:
        raise ValueError("V2 saved feature schema mismatch")
    if sha256(Path(model_path).read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("V2 model checksum mismatch")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    raw = model.predict(matrix)
    return reconstruct_price(raw, matrix[:, LATEST_INDEX].astype("float64")) \
        if metadata["residual_target"] else raw


def train_v2(output_dir=V2_DIR):
    import numpy as np
    import xgboost as xgb

    dataset = load_slices()
    v1_meta = json.loads((V1_DIR / "metadata.json").read_text(encoding="utf-8"))
    if v1_meta["feature_files"] != dataset.feature_report["files"]:
        raise ValueError("V1 and V2 data fingerprints differ")
    v1_sha_before = sha256((V1_DIR / "model.json").read_bytes()).hexdigest()
    if v1_sha_before != v1_meta["model_sha256"]:
        raise ValueError("Frozen V1 model checksum mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = {part: {} for part in ("tune", "evaluate")}
    details = {}
    trained = {}
    for name, spec in VARIANTS.items():
        params = {**V1_PARAMETERS, "objective": spec["objective"]}
        if "huber_slope" in spec:
            params["huber_slope"] = spec["huber_slope"]
        model = xgb.XGBRegressor(**params)
        y_train = labels_for_variant(dataset.train, residual=spec["residual"])
        y_tune = labels_for_variant(dataset.tune, residual=spec["residual"])
        start = perf_counter()
        model.fit(dataset.train.x, y_train,
                  eval_set=[(dataset.train.x, y_train),
                            (dataset.tune.x, y_tune)], verbose=False)
        train_seconds = perf_counter() - start
        path = output_dir / name
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.json"
        model.save_model(model_path)
        metadata = {
            "variant": name, "objective": spec["objective"],
            "residual_target": spec["residual"],
            "target_formula": ("future_price - latest_completed_price"
                               if spec["residual"] else "future_price"),
            "reconstruction": ("latest_completed_price + predicted_residual"
                               if spec["residual"] else "direct_predicted_price"),
            "eval_metric": "MAE on labels, algebraically equal to absolute-price MAE",
            "parameters": params, "feature_order": list(FEATURE_FIELDS),
            "feature_files": dataset.feature_report["files"],
            "python": platform.python_version(), "xgboost": xgb.__version__,
            "best_iteration_zero_based": int(model.best_iteration),
            "best_tune_mae": float(model.best_score),
            "fitted_rounds": len(model.evals_result()["validation_1"]["mae"]),
            "train_seconds": train_seconds,
            "model_sha256": sha256(model_path.read_bytes()).hexdigest(),
        }
        _write_json_atomic(path / "metadata.json", metadata)
        trained[name] = model
        details[name] = {**metadata, "learning_curve": model.evals_result(),
                         "inference_seconds": {}}
        # Train and select every fixed variant using tuning data before
        # examining any July-November prediction.
        start = perf_counter()
        raw = model.predict(dataset.tune.x)
        details[name]["inference_seconds"]["tune"] = perf_counter() - start
        final = prediction_to_price(raw, dataset.tune, residual=spec["residual"])
        if spec["residual"]:
            residual_mae = np.mean(np.abs(y_tune - raw))
            price_mae = np.mean(np.abs(dataset.tune.y - final))
            if not np.isclose(residual_mae, price_mae, atol=1e-8, rtol=1e-6):
                raise ValueError("Residual early-stopping MAE is not absolute-price MAE")
        predictions["tune"][name] = final

    selected = min(VARIANTS, key=lambda name: numeric_scores(
        dataset.tune.y, predictions["tune"][name])["mae"])
    for name, spec in VARIANTS.items():
        start = perf_counter()
        raw = trained[name].predict(dataset.evaluate.x)
        details[name]["inference_seconds"]["evaluate"] = perf_counter() - start
        predictions["evaluate"][name] = prediction_to_price(
            raw, dataset.evaluate, residual=spec["residual"])

    result = {
        "experiment": "Four fixed XGBoost V2 loss/target variants",
        "selection_rule": "Lowest Jan-Jun 2025 tuning MAE among V2 variants; July-Nov used only afterward",
        "selected_by_tuning_mae": selected,
        "july_november_status": "Reused development evaluation, not fresh independent test",
        "held_out_status": "Dec 2025-Feb 2026 untouched",
        "counts": dataset.counts,
        "v1_model_sha256": v1_sha_before,
        "feature_files": dataset.feature_report["files"],
        "variants": details, "splits": {},
    }
    for part in ("tune", "evaluate"):
        portion = getattr(dataset, part)
        forecasts = {"V1": _stored_predictions(part, portion),
                     **predictions[part], **baseline_predictions(dataset, portion)}
        if tuple(forecasts) != MODEL_NAMES:
            raise ValueError("Comparison model ordering mismatch")
        metrics = {
            "row_count": len(portion.y),
            "models": {name: full_scores(portion.y, values)
                       for name, values in forecasts.items()},
            "actual_price_ranges": _conditional_scores(portion.y, forecasts),
            "monthly": grouped_scores(portion.y, forecasts, portion.rows,
                                      "target_local_month"),
            "hour_of_day": grouped_scores(portion.y, forecasts, portion.rows,
                                          "target_local_hour"),
            "subperiods": _subperiod_scores(portion, forecasts),
            "movement": movement_diagnostics(portion.y, forecasts, portion.rows),
        }
        metrics["predictions_file"] = _write_prediction_file(
            output_dir / f"{part}_comparison.csv.gz", portion, forecasts)
        result["splits"][part] = metrics
    if sha256((V1_DIR / "model.json").read_bytes()).hexdigest() != v1_sha_before:
        raise ValueError("V1 model changed during V2 experiment")
    _write_json_atomic(output_dir / "v2_results.json", result)
    return result


if __name__ == "__main__":
    report = train_v2()
    print(json.dumps({"selected": report["selected_by_tuning_mae"],
                      "variants": {name: {
                          "best_iteration": detail["best_iteration_zero_based"],
                          "training_seconds": detail["train_seconds"]}
                          for name, detail in report["variants"].items()},
                      "tune_mae": {name: score["mae"] for name, score
                                   in report["splits"]["tune"]["models"].items()},
                      "reused_evaluation_mae": {name: score["mae"] for name, score
                          in report["splits"]["evaluate"]["models"].items()}}, indent=2))
