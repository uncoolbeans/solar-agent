"""Read-only V1 diagnosis on training, tuning and reused development evaluation."""

import csv
from datetime import datetime, timedelta
import gzip
from hashlib import sha256
import json

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .ml_dataset import load_slices
from .xgb_v1 import MODEL_DIR as V1_DIR, predict_saved


V2_DIR = ROOT / "data" / "models" / "price_xgb_v2"
PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)
PRICE_RANGES = (
    ("negative", None, 0.0), ("zero_to_10c", 0.0, 0.10),
    ("10c_to_30c", 0.10, 0.30), ("30c_to_1dollar", 0.30, 1.0),
    ("spike_at_least_1dollar", 1.0, None),
)


def distribution(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    quantiles = np.percentile(values, PERCENTILES)
    return {"count": len(values), "minimum": float(np.min(values)),
            "maximum": float(np.max(values)), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "std_population": float(np.std(values)),
            "quantiles": {f"p{p}": float(v) for p, v in zip(PERCENTILES, quantiles)}}


def range_bias(actual, prediction):
    import numpy as np
    a, p = np.asarray(actual), np.asarray(prediction)
    result = {}
    for name, low, high in PRICE_RANGES:
        mask = np.ones(len(a), dtype=bool)
        if low is not None:
            mask &= a >= low
        if high is not None:
            mask &= a < high
        n = int(mask.sum())
        result[name] = {"count": n, "bias_prediction_minus_actual":
                        float(np.mean(p[mask] - a[mask])) if n else None,
                        "mae": float(np.mean(np.abs(p[mask] - a[mask]))) if n else None,
                        "mean_prediction": float(np.mean(p[mask])) if n else None}
    return result


def latest_relationship(actual, prediction, latest):
    import numpy as np
    a, p, l = map(np.asarray, (actual, prediction, latest))
    mask = l < 0
    return {
        "pearson_latest_vs_prediction": float(np.corrcoef(l, p)[0, 1]),
        "linear_slope_prediction_on_latest": float(np.cov(l, p, bias=True)[0, 1] / np.var(l)),
        "mean_prediction_minus_latest": float(np.mean(p - l)),
        "latest_negative_count": int(mask.sum()),
        "predicted_negative_when_latest_negative": int(np.sum(mask & (p < 0))),
        "mean_prediction_when_latest_negative": float(np.mean(p[mask])) if mask.any() else None,
        "mean_latest_when_latest_negative": float(np.mean(l[mask])) if mask.any() else None,
        "predicted_negative_count": int(np.sum(p < 0)),
    }


def market_conditions(portion, latest, demand):
    import numpy as np
    y = np.asarray(portion.y)
    dates = None
    if portion.rows:
        dates = [datetime.fromisoformat(r["target_interval_start_utc"].replace("Z", "+00:00"))
                 for r in portion.rows]
    changes = np.abs(np.diff(y))
    if dates is not None:
        consecutive = np.array([b - a == timedelta(minutes=5)
                                for a, b in zip(dates, dates[1:])])
        changes = changes[consecutive]
    return {"count": len(y), "negative_count": int(np.sum(y < 0)),
            "negative_frequency": float(np.mean(y < 0)),
            "spike_count": int(np.sum(y >= 1)),
            "spike_frequency": float(np.mean(y >= 1)),
            "actual_price_std": float(np.std(y)),
            "mean_abs_consecutive_5m_price_change": float(np.mean(changes)),
            "latest_demand_mw_mean": float(np.mean(demand)),
            "latest_demand_mw_std": float(np.std(demand)),
            "actual_price": distribution(y)}


def squared_error_influence(actual, prediction, reference):
    """How much rare target ranges contribute to squared error, not causality."""
    import numpy as np
    a, p = np.asarray(actual), np.asarray(prediction)
    ref = np.full(len(a), reference)
    result = {}
    for label, error in (("v1_selected_model", (a - p) ** 2),
                         ("training_median_reference", (a - ref) ** 2)):
        total = float(np.sum(error))
        result[label] = {name: {
            "row_count": int(mask.sum()), "row_fraction": float(np.mean(mask)),
            "squared_error_fraction": float(np.sum(error[mask]) / total) if total else None}
            for name, mask in (("negative", a < 0), ("spike_at_least_1dollar", a >= 1),
                               ("extreme_abs_at_least_1dollar", np.abs(a) >= 1))}
    return result


def _stored_predictions(split, portion):
    import numpy as np
    path = V1_DIR / f"{split}_predictions.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != len(portion.y):
        raise ValueError("V1 prediction row count mismatch")
    if any(row["decision_start_utc"] != meta["decision_start_utc"]
           or float(row["actual_aud_per_kwh"]) != target
           for row, meta, target in zip(rows, portion.rows, portion.y)):
        raise ValueError("V1 predictions do not align to frozen features")
    return np.asarray([float(row["xgboost_aud_per_kwh"]) for row in rows])


def run_diagnostics(output_dir=V2_DIR):
    import numpy as np
    import xgboost as xgb

    dataset = load_slices()
    v1_metadata = json.loads((V1_DIR / "metadata.json").read_text(encoding="utf-8"))
    if v1_metadata["feature_files"] != dataset.feature_report["files"]:
        raise ValueError("V1 and V2 feature fingerprints differ")
    v1_results = json.loads((V1_DIR / "results.json").read_text(encoding="utf-8"))
    model = xgb.XGBRegressor()
    model.load_model(V1_DIR / "model.json")
    best = v1_metadata["best_iteration_zero_based"]
    booster = model.get_booster()[:best + 1]
    total_gain = booster.get_score(importance_type="total_gain")
    split_count = booster.get_score(importance_type="weight")
    gain_by_feature = {feature: float(total_gain.get(f"f{i}", 0.0))
                       for i, feature in enumerate(FEATURE_FIELDS)}
    gain_sum = sum(gain_by_feature.values())
    top = sorted(gain_by_feature, key=gain_by_feature.get, reverse=True)
    report = {
        "v1_model_sha256": v1_metadata["model_sha256"],
        "v1_results_sha256": sha256((V1_DIR / "results.json").read_bytes()).hexdigest(),
        "reused_development_evaluation": True,
        "training_only_median": dataset.constant_median,
        "price_range_semantics": [name for name, *_ in PRICE_RANGES],
        "splits": {},
        "feature_gain": {
            "total_gain_top_15": [
                {"name": name, "total_gain": gain_by_feature[name],
                 "fraction_of_all_total_gain": gain_by_feature[name] / gain_sum,
                 "split_count": int(split_count.get(f"f{FEATURE_FIELDS.index(name)}", 0))}
                for name in top[:15]],
            "top_5_total_gain_fraction": sum(gain_by_feature[n] for n in top[:5]) / gain_sum,
            "local_fold_total_gain": gain_by_feature["local_fold"],
            "active_feature_count": sum(value > 0 for value in gain_by_feature.values()),
        },
        "learning_curve": {},
    }
    curve = v1_results["learning_curve"]
    report["learning_curve"] = {
        "best_iteration_zero_based": best,
        "training_mae_at_best": curve["validation_0"]["mae"][best],
        "tuning_mae_at_best": curve["validation_1"]["mae"][best],
        "training_mae_final_fitted_round": curve["validation_0"]["mae"][-1],
        "tuning_mae_final_fitted_round": curve["validation_1"]["mae"][-1],
        "fitted_rounds": len(curve["validation_1"]["mae"]),
    }
    for split in ("train", "tune", "evaluate"):
        portion = getattr(dataset, split)
        if split == "train":
            pred = predict_saved(V1_DIR / "model.json", V1_DIR / "metadata.json",
                                 portion.x, FEATURE_FIELDS)
        else:
            pred = _stored_predictions(split, portion)
            saved = predict_saved(V1_DIR / "model.json", V1_DIR / "metadata.json",
                                  portion.x[:20], FEATURE_FIELDS)
            if not np.allclose(pred[:20], saved, rtol=1e-7, atol=1e-8):
                raise ValueError("Stored V1 predictions differ from saved-model inference")
        latest = portion.x[:, FEATURE_FIELDS.index("price_lag_5m_aud_per_kwh")]
        demand = portion.x[:, FEATURE_FIELDS.index("demand_lag_5m_mw")]
        report["splits"][split] = {
            "market": market_conditions(portion, latest, demand),
            "prediction": distribution(pred),
            "range_bias": range_bias(portion.y, pred),
            "latest_price_relation": latest_relationship(portion.y, pred, latest),
            "squared_error_influence": squared_error_influence(
                portion.y, pred, dataset.constant_median),
        }
    report["evaluation_monthly_v1"] = {
        key: value["xgboost"] for key, value in v1_results["splits"]["evaluate"]["monthly"].items()}
    report["evaluation_hourly_v1"] = {
        key: value["xgboost"] for key, value in v1_results["splits"]["evaluate"]["hour_of_day"].items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "v1_diagnostics.json", report)
    return report


if __name__ == "__main__":
    result = run_diagnostics()
    print(json.dumps({"splits": {k: v["market"] for k, v in result["splits"].items()},
                      "evaluation_prediction": result["splits"]["evaluate"]["prediction"],
                      "feature_gain": result["feature_gain"],
                      "learning_curve": result["learning_curve"]}, indent=2))
