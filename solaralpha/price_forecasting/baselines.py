"""One-hour wholesale price baselines, scored only on eligible validation origins."""

import csv
import gzip
from hashlib import sha256
from io import TextIOWrapper
from math import sqrt
from statistics import median

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_DIR, TARGET_FIELD, read_feature_rows


PREDICTION_FIELDS = ("decision_start_utc", "target_interval_start_utc",
                     "origin_local_date", "actual_aud_per_kwh",
                     "persistence_aud_per_kwh", "daily_seasonal_aud_per_kwh",
                     "train_median_aud_per_kwh")
SPIKE_THRESHOLD_AUD_PER_KWH = 1.0


def training_target_median(rows):
    """Fit on training targets alone, never validation or development-week targets."""
    values = []
    for row in rows:
        if row["origin_partition"] != "train" or row["target_partition"] != "train":
            raise ValueError("Median fit contains non-training target")
        values.append(float(row[TARGET_FIELD]))
    if not values:
        raise ValueError("Empty training targets")
    return median(values), len(values)


def predict_row(row, fitted_median):
    return {
        "persistence_aud_per_kwh": float(row["price_lag_5m_aud_per_kwh"]),
        "daily_seasonal_aud_per_kwh": float(
            row["price_target_aligned_previous_day_aud_per_kwh"]),
        "train_median_aud_per_kwh": fitted_median,
    }


def _scores(actual, predicted):
    errors = [p - a for a, p in zip(actual, predicted)]
    absolute = [abs(e) for e in errors]
    n = len(errors)
    if not n:
        raise ValueError("No validation predictions")
    return {"count": n, "mae_aud_per_kwh": sum(absolute) / n,
            "rmse_aud_per_kwh": sqrt(sum(e * e for e in errors) / n),
            "median_absolute_error_aud_per_kwh": median(absolute),
            "bias_aud_per_kwh": sum(errors) / n}


def _event_scores(actual, predicted, predicate):
    truth = [predicate(v) for v in actual]
    forecast = [predicate(v) for v in predicted]
    true_positive = sum(t and p for t, p in zip(truth, forecast))
    actual_count = sum(truth)
    predicted_count = sum(forecast)
    return {
        "actual_count": actual_count,
        "predicted_count": predicted_count,
        "true_positive": true_positive,
        "false_positive": predicted_count - true_positive,
        "recall": true_positive / actual_count if actual_count else None,
        "precision": true_positive / predicted_count if predicted_count else None,
        "conditional_mae_aud_per_kwh": (
            sum(abs(p - a) for a, p, t in zip(actual, predicted, truth) if t)
            / actual_count if actual_count else None),
    }


def evaluate_baselines(feature_dir=FEATURE_DIR):
    fitted_median, train_count = training_target_median(read_feature_rows(
        feature_dir / "train_one_hour_price.csv.gz"))
    actual = []
    predictions = {field: [] for field in PREDICTION_FIELDS[-3:]}
    output_path = feature_dir / "validation_baseline_predictions.csv.gz"
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    with temp_path.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=PREDICTION_FIELDS)
                writer.writeheader()
                for row in read_feature_rows(feature_dir / "validation_one_hour_price.csv.gz"):
                    if row["include_in_validation_score"] != "1":
                        continue
                    if row["origin_partition"] != "validation" or row["target_partition"] != "validation":
                        raise ValueError("Invalid validation row")
                    a = float(row[TARGET_FIELD])
                    p = predict_row(row, fitted_median)
                    writer.writerow({"decision_start_utc": row["decision_start_utc"],
                                     "target_interval_start_utc": row["target_interval_start_utc"],
                                     "origin_local_date": row["origin_local_date"],
                                     "actual_aud_per_kwh": a, **p})
                    actual.append(a)
                    for field, value in p.items():
                        predictions[field].append(value)
                    count += 1
    temp_path.replace(output_path)
    report = {
        "target": "NSW1 wholesale AUD/kWh at interval start t+60 minutes",
        "fit_train_targets": train_count,
        "fitted_training_median_aud_per_kwh": fitted_median,
        "scored_validation_origins": count,
        "negative_event_definition": "actual or forecast price < 0 AUD/kWh",
        "spike_event_definition": "actual or forecast price >= 1.00 AUD/kWh (fixed before validation scoring)",
        "models": {},
        "predictions_file": {"path": output_path.relative_to(ROOT).as_posix(),
                             "sha256": sha256(output_path.read_bytes()).hexdigest(),
                             "bytes": output_path.stat().st_size},
    }
    for field, values in predictions.items():
        report["models"][field] = {
            **_scores(actual, values),
            "negative_price": _event_scores(actual, values, lambda x: x < 0),
            "price_spike": _event_scores(
                actual, values, lambda x: x >= SPIKE_THRESHOLD_AUD_PER_KWH),
        }
    _write_json_atomic(feature_dir / "baseline_metrics.json", report)
    return report


def read_prediction_rows(path=FEATURE_DIR / "validation_baseline_predictions.csv.gz"):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)
