"""Shared numerical and event diagnostics for matched forecast rows."""

from datetime import datetime, timedelta

from .baselines import SPIKE_THRESHOLD_AUD_PER_KWH


def numeric_scores(actual, forecast):
    import numpy as np
    a, p = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    if a.shape != p.shape or not len(a):
        raise ValueError("Prediction and target shapes differ or are empty")
    error = p - a
    return {"count": len(a), "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "median_absolute_error": float(np.median(np.abs(error))),
            "mean_bias": float(np.mean(error))}


def event_scores(actual, forecast, threshold, negative=False):
    import numpy as np
    a, p = np.asarray(actual), np.asarray(forecast)
    truth = a < threshold if negative else a >= threshold
    predicted = p < threshold if negative else p >= threshold
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    actual_count = int(np.sum(truth))
    predicted_count = int(np.sum(predicted))
    precision = tp / predicted_count if predicted_count else None
    recall = tp / actual_count if actual_count else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else 0.0)
    return {"actual_count": actual_count, "predicted_count": predicted_count,
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "conditional_mae": float(np.mean(np.abs(p[truth] - a[truth])))
                if actual_count else None}


def full_scores(actual, forecast):
    return {**numeric_scores(actual, forecast),
            "negative_price": event_scores(actual, forecast, 0.0, negative=True),
            "spike": event_scores(actual, forecast, SPIKE_THRESHOLD_AUD_PER_KWH)}


def grouped_scores(actual, forecasts, rows, key):
    import numpy as np
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row[key]), []).append(i)
    return {name: {model: numeric_scores(np.asarray(actual)[indices],
                                         np.asarray(values)[indices])
                   for model, values in forecasts.items()}
            for name, indices in sorted(groups.items())}


def movement_diagnostics(actual, forecasts, rows):
    """Check one-hour changes and fresh events not already visible at decision time."""
    import numpy as np
    a = np.asarray(actual)
    latest = np.asarray([row["persistence"] for row in rows])
    target_times = [datetime.fromisoformat(
        row["target_interval_start_utc"].replace("Z", "+00:00")) for row in rows]
    actual_by_target = dict(zip(target_times, a))
    material = np.abs(a - latest) >= 0.05
    result = {"material_change_threshold_aud_per_kwh": 0.05,
              "material_change_count": int(np.sum(material)), "models": {}}
    for label, mask in {
        "material_upward_move": (a - latest) >= 0.05,
        "material_downward_move": (a - latest) <= -0.05,
    }.items():
        result[label] = {"count": int(np.sum(mask)), "models": {}}
        for name, values in forecasts.items():
            p = np.asarray(values)
            direction = (p - latest) > 0 if label == "material_upward_move" else (p - latest) < 0
            result[label]["models"][name] = {
                "direction_correct_count": int(np.sum(mask & direction)),
                "mae": float(np.mean(np.abs(p[mask] - a[mask]))) if np.any(mask) else None}
    for label, mask in {
        "negative_onset": (a < 0) & (latest >= 0),
        "spike_onset": (a >= SPIKE_THRESHOLD_AUD_PER_KWH)
                       & (latest < SPIKE_THRESHOLD_AUD_PER_KWH),
    }.items():
        result[label] = {"count": int(np.sum(mask)), "models": {}}
        for name, values in forecasts.items():
            p = np.asarray(values)
            detected = p < 0 if label == "negative_onset" else p >= SPIKE_THRESHOLD_AUD_PER_KWH
            result[label]["models"][name] = {
                "predicted_event_count": int(np.sum(mask & detected)),
                "mae": float(np.mean(np.abs(p[mask] - a[mask]))) if np.any(mask) else None}
    for name, values in forecasts.items():
        p = np.asarray(values)
        false_spikes = np.flatnonzero((p >= SPIKE_THRESHOLD_AUD_PER_KWH)
                                         & (a < SPIKE_THRESHOLD_AUD_PER_KWH))
        following_known = full_history = 0
        for index in false_spikes:
            target = target_times[index]
            # At decision t = target-60m, only intervals beginning at or
            # before t-5m are completed. Inspect [t-60m, t-5m].
            recent = [actual_by_target.get(target - timedelta(minutes=offset))
                      for offset in range(65, 125, 5)]
            if all(value is not None for value in recent):
                full_history += 1
                following_known += any(value >= SPIKE_THRESHOLD_AUD_PER_KWH
                                       for value in recent)
        result["models"][name] = {
            "material_change_direction_accuracy": (
                float(np.mean(np.sign(p[material] - latest[material])
                              == np.sign(a[material] - latest[material])))
                if np.any(material) else None),
            "material_change_mae": (float(np.mean(np.abs(p[material] - a[material])))
                                    if np.any(material) else None),
            "false_spike_predictions_with_full_known_hour": full_history,
            "false_spike_predictions_after_known_hour_spike": following_known,
        }
    return result
