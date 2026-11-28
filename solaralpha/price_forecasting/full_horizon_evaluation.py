"""Matched 144-offset diagnostics; all future prices remain offline labels."""

import argparse
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from solaralpha.historical_pipeline import ROOT, STEP, _write_json_atomic
from .features import FEATURE_FIELDS, load_training_validation_points
from .full_horizon_dataset import HORIZONS, TARGET_DIR, load_targets
from .full_horizon_direct import OUTPUT_DIR, PARAMS, model_paths, verified_model
from .full_horizon_tcn import (_loader as tcn_loader, load_saved as load_tcn,
                               load_sequence_targets)
from .ml_dataset import _utc
from .ml_metrics import full_scores, grouped_scores, numeric_scores
from .multi_horizon import HORIZONS as SPARSE_HORIZONS, LATEST_INDEX, _event_onset
from .sequence_experiment import _peak_working_set_mib, _setup
from .xgb_v2 import _conditional_scores


NAMES = ("persistence", "daily_seasonal", "train_median", "hour_of_week_median",
         "direct", "hybrid_tuned", "hybrid_fixed", "tcn")


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _calendar(data):
    points = load_training_validation_points()
    if any(points[i].time_utc - points[i-1].time_utc != STEP for i in range(1, len(points))):
        raise ValueError("Historical calendar grid is not contiguous")
    hour = np.asarray([p.local_time.weekday() * 24 + p.local_time.hour for p in points], dtype=np.uint8)
    month = np.asarray([p.local_time.year * 100 + p.local_time.month for p in points], dtype=np.int32)
    indices = {}
    for part, matrix in data.matrices.items():
        indices[part] = np.fromiter((int((_utc(row["decision_start_utc"]) - points[0].time_utc) / STEP)
                                      for row in matrix.rows), dtype=np.int32, count=len(matrix.rows))
        if np.any(np.diff(indices[part]) <= 0):
            raise ValueError("Dense evaluation origins are not chronological")
    return indices, hour, month


def _baselines(data, part, col, indices, point_hour):
    train_y = np.asarray(data.targets["train"][:, col])
    eligible = np.isfinite(train_y)
    train_hour = point_hour[indices["train"] + col]
    medians = np.empty(168, dtype=np.float64)
    for bin_ in range(168):
        values = train_y[eligible & (train_hour == bin_)]
        if not len(values):
            raise ValueError("Training-only seasonal median bin is empty")
        medians[bin_] = float(np.median(values))
    mask = data.common[part]
    matrix = data.matrices[part]
    return {"persistence": matrix.x[mask, LATEST_INDEX].astype(np.float64),
            "daily_seasonal": np.asarray(data.seasonal[part][mask, col]),
            "train_median": np.full(int(mask.sum()), float(np.median(train_y[eligible]))),
            "hour_of_week_median": medians[point_hour[indices[part][mask] + col]]}


def _model_fingerprint(output_dir):
    hashes = [json.loads(model_paths(h, output_dir)[1].read_text(encoding="utf-8"))["model_sha256"]
              for h in HORIZONS]
    return sha256("\n".join(hashes).encode()).hexdigest()


def _direct_predictions(data, part, *, output_dir=OUTPUT_DIR):
    path = output_dir / f"direct_{part}_common.npy"
    cache_meta = output_dir / f"direct_{part}_common.json"
    target_sha = _sha(TARGET_DIR / "manifest.json")
    model_fingerprint = _model_fingerprint(output_dir)
    if path.exists() and cache_meta.exists():
        cache = json.loads(cache_meta.read_text(encoding="utf-8"))
        arr = np.load(path, mmap_mode="r")
        if (arr.shape == (int(data.common[part].sum()), len(HORIZONS)) and
                cache["target_manifest_sha256"] == target_sha and
                cache["direct_model_fingerprint"] == model_fingerprint and
                cache["array_sha256"] == _sha(path)):
            return arr
        raise ValueError("Existing direct prediction cache has a different identity")
    matrix = data.matrices[part].x[data.common[part]]
    tmp = path.with_suffix(".npy.tmp")
    predictions = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                            shape=(len(matrix), len(HORIZONS)))
    for col, h in enumerate(HORIZONS):
        model, _ = verified_model(h, data, output_dir)
        predictions[:, col] = model.predict(matrix)
        print(f"Direct {part} +{h}m predicted", flush=True)
    predictions.flush()
    del predictions
    tmp.replace(path)
    _write_json_atomic(cache_meta, {"target_manifest_sha256": target_sha,
                                   "direct_model_fingerprint": model_fingerprint,
                                   "array_sha256": _sha(path),
                                   "shape": [len(matrix), len(HORIZONS)]})
    return np.load(path, mmap_mode="r")


def freeze_hybrid(data=None, output_dir=OUTPUT_DIR):
    """Write horizon choices before loading or reporting July-November predictions."""
    data = data or load_targets()
    direct = _direct_predictions(data, "tune", output_dir=output_dir)
    scores, choice = {}, []
    baseline = data.matrices["tune"].x[data.common["tune"], LATEST_INDEX].astype(np.float64)
    for col, h in enumerate(HORIZONS):
        actual = np.asarray(data.targets["tune"][data.common["tune"], col])
        persistence_mae = float(np.mean(np.abs(baseline - actual)))
        direct_mae = float(np.mean(np.abs(direct[:, col] - actual)))
        selected = "persistence" if persistence_mae <= direct_mae else "direct"
        scores[str(h)] = {"persistence": persistence_mae, "direct": direct_mae,
                          "selected": selected}
        choice.append(selected)
    selection = {"schema_version": 1, "selection_period": "January-June 2025 common origins",
                 "rule": "lower common-origin tuning MAE per horizon; persistence wins exact ties",
                 "horizons_minutes": list(HORIZONS), "choices": choice,
                 "fixed_choices": ["persistence" if h in (0, 5) else "direct" for h in HORIZONS],
                 "tune_scores": scores, "target_manifest_sha256": _sha(TARGET_DIR / "manifest.json"),
                 "direct_tune_sha256": _sha(output_dir / "direct_tune_common.npy"),
                 "frozen_before_reused_evaluation": True}
    _write_json_atomic(output_dir / "selection.json", selection)
    return selection


def _tcn_predictions(data, part, output_dir=OUTPUT_DIR):
    _setup()
    sparse, labels = load_sequence_targets(data)
    model, meta = load_tcn(output_dir / "tcn")
    if meta["target_manifest_sha256"] != _sha(TARGET_DIR / "manifest.json"):
        raise ValueError("Dense TCN target fingerprint differs")
    start = perf_counter()
    result = []
    with torch.inference_mode():
        for x, _ in tcn_loader(sparse, labels, part):
            result.append(model(x).numpy())
    array = np.concatenate(result).astype(np.float32)
    if array.shape != (int(data.common[part].sum()), len(HORIZONS)):
        raise ValueError("TCN common-origin prediction shape differs")
    return array, perf_counter() - start, sparse, model


def _per_horizon_scores(actual, forecasts, latest, months):
    for name, values in forecasts.items():
        if values.shape != actual.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} dense price predictions")
    rows = [{"month": str(value)} for value in months]
    return {"count": len(actual),
            "models": {name: {**full_scores(actual, values),
                              "negative_onset": _event_onset(actual, values, latest, True),
                              "spike_onset": _event_onset(actual, values, latest, False)}
                       for name, values in forecasts.items()},
            "conditional_price_ranges": _conditional_scores(actual, forecasts),
            "monthly": grouped_scores(actual, forecasts, rows, "month")}


def _save_comparison(path, origins, actual, forecasts):
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, origin_point_indices=origins.astype(np.int64),
                            actual=actual.astype(np.float32),
                            **{name: np.asarray(values, dtype=np.float32) for name, values in forecasts.items()})
    tmp.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path),
            "bytes": path.stat().st_size}


def _trajectory_summary(actual, predicted):
    diff_a, diff_p = np.diff(actual, axis=1), np.diff(predicted, axis=1)
    def negative_runs(values):
        mask = values < 0
        running = np.zeros(len(values), dtype=np.int16)
        longest = np.zeros(len(values), dtype=np.int16)
        for col in range(mask.shape[1]):
            running = np.where(mask[:, col], running + 1, 0)
            longest = np.maximum(longest, running)
        episodes = mask[:, 0].astype(np.int16) + np.sum(mask[:, 1:] & ~mask[:, :-1], axis=1)
        return longest * 5, episodes
    longest_actual, episodes_actual = negative_runs(actual)
    longest_predicted, episodes_predicted = negative_runs(predicted)
    return {
        "trajectories": len(actual),
        "mean_actual_range_aud_per_kwh": float(np.mean(np.ptp(actual, axis=1))),
        "mean_predicted_range_aud_per_kwh": float(np.mean(np.ptp(predicted, axis=1))),
        "median_actual_range_aud_per_kwh": float(np.median(np.ptp(actual, axis=1))),
        "median_predicted_range_aud_per_kwh": float(np.median(np.ptp(predicted, axis=1))),
        "mean_absolute_range_error": float(np.mean(np.abs(np.ptp(predicted, axis=1) - np.ptp(actual, axis=1)))),
        "peak_magnitude_mae": float(np.mean(np.abs(predicted.max(axis=1) - actual.max(axis=1)))),
        "trough_magnitude_mae": float(np.mean(np.abs(predicted.min(axis=1) - actual.min(axis=1)))),
        "peak_timing_mae_minutes": float(np.mean(np.abs(predicted.argmax(axis=1) - actual.argmax(axis=1))) * 5),
        "trough_timing_mae_minutes": float(np.mean(np.abs(predicted.argmin(axis=1) - actual.argmin(axis=1))) * 5),
        "five_minute_change_mae": float(np.mean(np.abs(diff_p - diff_a))),
        "mean_abs_five_minute_change_actual": float(np.mean(np.abs(diff_a))),
        "mean_abs_five_minute_change_predicted": float(np.mean(np.abs(diff_p))),
        "p95_abs_five_minute_change_actual": float(np.percentile(np.abs(diff_a), 95)),
        "p95_abs_five_minute_change_predicted": float(np.percentile(np.abs(diff_p), 95)),
        "predicted_jumps_above_10c_fraction": float(np.mean(np.abs(diff_p) > 0.1)),
        "actual_jumps_above_10c_fraction": float(np.mean(np.abs(diff_a) > 0.1)),
        "negative_duration_mae_minutes": float(np.mean(np.abs((predicted < 0).sum(axis=1) -
                                                              (actual < 0).sum(axis=1))) * 5),
        "longest_negative_run_actual_mean_minutes": float(np.mean(longest_actual)),
        "longest_negative_run_predicted_mean_minutes": float(np.mean(longest_predicted)),
        "longest_negative_run_mae_minutes": float(np.mean(np.abs(longest_predicted - longest_actual))),
        "negative_episode_count_actual_mean": float(np.mean(episodes_actual)),
        "negative_episode_count_predicted_mean": float(np.mean(episodes_predicted)),
    }


def trajectory_diagnostics(path):
    """Full 12-hour and paired-midnight/noon development-day diagnostics."""
    with np.load(path) as source:
        actual = np.asarray(source["actual"])
        origins = np.asarray(source["origin_point_indices"])
        names = ("direct", "tcn", "hybrid_tuned", "hybrid_fixed", "persistence")
        forecasts = {name: np.asarray(source[name]) for name in names}
    points = load_training_validation_points()
    pairs = {}
    for i, index in enumerate(origins):
        local = points[int(index)].local_time
        if local.minute == 0 and local.hour in (0, 12):
            pairs.setdefault(local.date().isoformat(), {})[local.hour] = i
    dates = sorted(date for date, v in pairs.items() if 0 in v and 12 in v and
                   points[int(origins[v[12]])].time_utc -
                   points[int(origins[v[0]])].time_utc == timedelta(hours=12))
    daily_actual = np.stack([np.concatenate((actual[pairs[d][0]], actual[pairs[d][12]])) for d in dates])
    daily = {name: _trajectory_summary(
        daily_actual,
        np.stack([np.concatenate((forecast[pairs[d][0]], forecast[pairs[d][12]])) for d in dates]))
        for name, forecast in forecasts.items()}
    return {"twelve_hour_all_common_origins": {
                name: _trajectory_summary(actual, forecast) for name, forecast in forecasts.items()},
            "paired_midnight_noon_local_days": {"count": len(dates), "dates": dates,
                                                   "models": daily,
                                                   "definition": "two 144-point forecasts issued at Sydney-local midnight and noon, covering one 24-hour local day when both origins are eligible; DST-length transition days excluded"}}


def _latency(predict, repeats):
    predict()
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        result = predict()
        if np.asarray(result).shape != (len(HORIZONS),):
            raise ValueError("Incomplete single-decision trajectory")
        elapsed.append((perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(elapsed)), "p95_ms": float(np.percentile(elapsed, 95)),
            "repeats": repeats, "scope": "warm, preassembled 41-feature or scaled 72x41 input; excludes data construction and model loading"}


def resource_summary(data, sparse, tcn_model, output_dir=OUTPUT_DIR):
    _setup()
    models, metadata = [], []
    for h in HORIZONS:
        model, meta = verified_model(h, data, output_dir)
        models.append(model)
        metadata.append(meta)
    row = data.matrices["evaluate"].x[data.common["evaluate"]][0:1]
    selected = json.loads((output_dir / "selection.json").read_text(encoding="utf-8"))
    def direct_predict():
        return np.asarray([model.predict(row)[0] for model in models])
    def tuned_predict():
        return np.asarray([row[0, LATEST_INDEX] if choice == "persistence" else models[i].predict(row)[0]
                           for i, choice in enumerate(selected["choices"])])
    def fixed_predict():
        return np.asarray([row[0, LATEST_INDEX] if choice == "persistence" else models[i].predict(row)[0]
                           for i, choice in enumerate(selected["fixed_choices"])])
    end = int(sparse.indices["evaluate"][0])
    window = torch.from_numpy(sparse.sequence.x[end-71:end+1][None])
    def tcn_predict():
        with torch.inference_mode():
            return tcn_model(window).numpy()[0]
    tcn_meta = json.loads((output_dir / "tcn/metadata.json").read_text(encoding="utf-8"))
    return {"direct_models": {"reused": len(SPARSE_HORIZONS),
                               "new": len(HORIZONS) - len(SPARSE_HORIZONS),
                               "selected_last_budget_round": sum(m["best_iteration_zero_based"] == 999 for m in metadata),
                               "sum_fit_seconds": float(sum(m["train_seconds"] for m in metadata)),
                               "new_fit_seconds": float(sum(m["train_seconds"] for h, m in zip(HORIZONS, metadata)
                                                            if h not in SPARSE_HORIZONS)),
                               "sum_model_bytes": int(sum(model_paths(h, output_dir)[0].stat().st_size for h in HORIZONS))},
            "tcn": {"trainable_parameters": tcn_meta["trainable_parameters"],
                    "best_epoch_one_based": tcn_meta["best_epoch_one_based"],
                    "fit_seconds": tcn_meta["training_wall_seconds"],
                    "model_bytes": tcn_meta["model_bytes"],
                    "training_peak_working_set_mib": tcn_meta["process_peak_working_set_mib"]},
            "single_decision_latency": {"direct": _latency(direct_predict, 20),
                                        "hybrid_tuned": _latency(tuned_predict, 20),
                                        "hybrid_fixed": _latency(fixed_predict, 20),
                                        "tcn": _latency(tcn_predict, 50)}}


def evaluate(data=None, output_dir=OUTPUT_DIR):
    data = data or load_targets()
    selection_path = output_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (tuple(selection["horizons_minutes"]) != HORIZONS or
            selection["target_manifest_sha256"] != _sha(TARGET_DIR / "manifest.json") or
            selection["direct_tune_sha256"] != _sha(output_dir / "direct_tune_common.npy")):
        raise ValueError("Frozen hybrid selection does not match dense targets")
    indices, point_hour, point_month = _calendar(data)
    direct = {"tune": _direct_predictions(data, "tune", output_dir=output_dir)}
    tune_tcn, tune_seconds, sparse, tcn_model = _tcn_predictions(data, "tune", output_dir)
    tcn, tcn_batch = {"tune": tune_tcn}, {"tune": tune_seconds}
    result = {"schema_version": 1, "horizons_minutes": list(HORIZONS),
              "counts": {part: int(data.common[part].sum()) for part in data.common},
              "selection": selection, "selection_file_sha256": _sha(selection_path),
              "target_manifest_sha256": _sha(TARGET_DIR / "manifest.json"),
              "source_hashes": {
                  "sparse_multi_output_results": _sha(ROOT / "data/models/price_multi_output_neural_v1/results.json"),
                  "sparse_direct_results": _sha(ROOT / "data/models/price_multi_horizon_v1/results.json"),
                  "one_hour_leaderboard": _sha(ROOT / "data/models/forecasting_leaderboard.json"),
                  "sparse_multi_horizon_leaderboard": _sha(ROOT / "data/models/multi_horizon_leaderboard.json")},
              "evaluation_status": "July-November 2025 reused development evaluation",
              "held_out": "December 2025-February 2026 untouched",
              "splits": {}, "mean_mae": {}, "tcn_batch_inference_seconds": tcn_batch}
    for part in ("tune", "evaluate"):
        if part == "evaluate":
            # The overall provisional choice and both hybrid rules are on disk first.
            direct[part] = _direct_predictions(data, part, output_dir=output_dir)
            tcn[part], tcn_batch[part], sparse, tcn_model = _tcn_predictions(data, part, output_dir)
        mask = data.common[part]
        actual = np.asarray(data.targets[part][mask], dtype=np.float64)
        origins = indices[part][mask]
        forecasts = {name: np.empty_like(actual, dtype=np.float32) for name in NAMES}
        record = {"horizons": {}}
        for col, h in enumerate(HORIZONS):
            base = _baselines(data, part, col, indices, point_hour)
            values = {**base, "direct": np.asarray(direct[part][:, col], dtype=np.float64),
                      "tcn": tcn[part][:, col].astype(np.float64)}
            choice = selection["choices"][col]
            fixed = selection["fixed_choices"][col]
            values["hybrid_tuned"] = values[choice]
            values["hybrid_fixed"] = values[fixed]
            values = {name: np.asarray(values[name]) for name in NAMES}
            for name in NAMES:
                forecasts[name][:, col] = values[name]
            months = point_month[origins + col]
            record["horizons"][str(h)] = _per_horizon_scores(actual[:, col], values,
                                                             values["persistence"], months)
            print(f"Scored {part} +{h}m", flush=True)
        record["predictions_file"] = _save_comparison(output_dir / f"{part}_comparison.npz",
                                                       origins, actual, forecasts)
        result["splits"][part] = record
        result["mean_mae"][part] = {name: float(mean(record["horizons"][str(h)]["models"][name]["mae"]
                                                  for h in HORIZONS)) for name in NAMES}
        if part == "tune":
            provisional = {"selected_by_equal_horizon_tune_mae": min(NAMES, key=result["mean_mae"]["tune"].get),
                           "tune_mean_mae": result["mean_mae"]["tune"],
                           "written_before_reused_evaluation": True}
            _write_json_atomic(output_dir / "model_selection.json", provisional)
            result["provisional_selection"] = provisional
    result["trajectory_diagnostics"] = trajectory_diagnostics(output_dir / "evaluate_comparison.npz")
    sparse_result = json.loads((ROOT / "data/models/price_multi_output_neural_v1/results.json").read_text(encoding="utf-8"))
    matched = {}
    for h in SPARSE_HORIZONS:
        old_models = sparse_result["splits"]["evaluate"]["horizons"][str(h)]["models"]
        new_models = result["splits"]["evaluate"]["horizons"][str(h)]["models"]
        if abs(old_models["direct"]["mae"] - new_models["direct"]["mae"]) > 1e-7:
            raise ValueError("Dense direct score does not reproduce sparse matched origins")
        matched[str(h)] = {"sparse_direct_mae": old_models["direct"]["mae"],
                           "dense_direct_mae": new_models["direct"]["mae"],
                           "sparse_tcn_mae": old_models["multi_tcn"]["mae"],
                           "dense_tcn_mae": new_models["tcn"]["mae"],
                           "sparse_tcn_negative_f1": old_models["multi_tcn"]["negative_price"]["f1"],
                           "dense_tcn_negative_f1": new_models["tcn"]["negative_price"]["f1"]}
    result["sparse_matched_comparison"] = matched
    result["resources"] = resource_summary(data, sparse, tcn_model, output_dir)
    result["process_peak_working_set_mib"] = _peak_working_set_mib()
    _write_json_atomic(output_dir / "results.json", result)
    _write_json_atomic(ROOT / "data/models/full_horizon_leaderboard.json", {
        "schema_version": 1, "task": "144 ordered five-minute interval-start NSW1 price forecasts",
        "common_evaluation_origins": result["counts"]["evaluate"],
        "selected_by_tuning": result["provisional_selection"]["selected_by_equal_horizon_tune_mae"],
        "entries": [{"name": name, "tune_mean_mae": result["mean_mae"]["tune"][name],
                     "reused_evaluation_mean_mae": result["mean_mae"]["evaluate"][name]}
                    for name in NAMES],
        "prior_leaderboard_sha256": result["source_hashes"]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    args = parser.parse_args()
    if args.stage == "freeze":
        s = freeze_hybrid()
        print(f"Frozen {s['choices'].count('persistence')} persistence and {s['choices'].count('direct')} direct choices")
    else:
        print(json.dumps(evaluate()["mean_mae"], indent=2))


if __name__ == "__main__":
    main()
