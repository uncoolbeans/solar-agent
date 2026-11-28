"""Matched offline diagnostics for ten-output neural prices and frozen references."""

import csv
from datetime import timedelta
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .horizon_conditioned import OUTPUT_DIR as SHARED_DIR
from .ml_dataset import _utc
from .multi_horizon import HORIZONS, LATEST_INDEX, Matrix, _scores
from .multi_output_experiment import (NAMES, OUTPUT_DIR, _predict, _sha,
                                      make_multi_output_model, predict_saved)
from .sequence_experiment import OUTPUT_DIR as SINGLE_DIR, _peak_working_set_mib, _setup


NEW_NAMES = tuple(f"multi_{name}" for name in NAMES)
BASE_NAMES = ("persistence", "daily_seasonal", "train_median",
              "hour_of_week_median", "direct", "shared", "hybrid")
ALL_NAMES = (*BASE_NAMES, *NEW_NAMES)
SINGLE_NAMES = tuple(f"single_{name}" for name in NAMES)


def _source_hashes():
    paths = {"shared_results": SHARED_DIR / "results.json",
             "single_results": SINGLE_DIR / "results.json",
             "one_hour_leaderboard": ROOT / "data/models/forecasting_leaderboard.json",
             "direct_results": ROOT / "data/models/price_multi_horizon_v1/results.json"}
    return {name: _sha(path) for name, path in paths.items()}


def _read_references(data, part, col, source):
    """Match all seven older methods to exactly these sequence origins."""
    h = HORIZONS[col]
    record = source["horizons"][str(h)]["splits"][part]["prediction_file"]
    path = ROOT / record["path"]
    if _sha(path) != record["sha256"]:
        raise ValueError("Frozen shared/direct prediction checksum changed")
    origins = data.origins[part]
    positions = {origin: i for i, origin in enumerate(origins)}
    if len(positions) != len(origins):
        raise ValueError("Duplicate common sequence origin")
    forecasts = {name: np.empty(len(origins), dtype=np.float64) for name in BASE_NAMES}
    seen = np.zeros(len(origins), dtype=bool)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            j = positions.get(row["decision_start_utc"])
            if j is None:
                continue
            target = (_utc(origins[j]) + timedelta(minutes=h)).isoformat().replace("+00:00", "Z")
            if (seen[j] or row["target_interval_start_utc"] != target or
                    abs(float(row["actual_aud_per_kwh"]) - data.targets[part][j, col]) > 1e-10):
                raise ValueError("Reference target/origin mismatch")
            seen[j] = True
            for name in BASE_NAMES:
                forecasts[name][j] = float(row[f"{name}_aud_per_kwh"])
    if not seen.all():
        raise ValueError("Common sequence origin missing frozen reference forecast")
    return forecasts


def _single_output_60(data, part):
    """Old one-hour LSTM/GRU/TCN on exactly the new common origin subset."""
    path = SINGLE_DIR / f"{part}_comparison.csv.gz"
    old_result = json.loads((SINGLE_DIR / "results.json").read_text(encoding="utf-8"))
    if _sha(path) != old_result["splits"][part]["predictions_file"]["sha256"]:
        raise ValueError("Frozen one-hour neural predictions changed")
    positions = {origin: i for i, origin in enumerate(data.origins[part])}
    values = {name: np.empty(len(positions), dtype=np.float64) for name in SINGLE_NAMES}
    seen = np.zeros(len(positions), dtype=bool)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            j = positions.get(row["decision_start_utc"])
            if j is None:
                continue
            target = (_utc(data.origins[part][j]) + timedelta(minutes=60)).isoformat().replace("+00:00", "Z")
            if (seen[j] or row["target_interval_start_utc"] != target or
                    abs(float(row["actual_aud_per_kwh"]) - data.targets[part][j, HORIZONS.index(60)]) > 1e-10):
                raise ValueError("Old neural +60 target mismatch")
            seen[j] = True
            for name in NAMES:
                values[f"single_{name}"][j] = float(row[f"{name}_aud_per_kwh"])
    if not seen.all():
        raise ValueError("Old neural forecast missing matched origin")
    return values


def _write_predictions(path, data, part, col, forecasts):
    h = HORIZONS[col]
    fields = ("decision_start_utc", "target_interval_start_utc", "actual_aud_per_kwh",
              *(f"{name}_aud_per_kwh" for name in forecasts))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as binary:
        with gzip.GzipFile(fileobj=binary, mode="wb", filename="", mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for i, origin in enumerate(data.origins[part]):
                    target = (_utc(origin) + timedelta(minutes=h)).isoformat().replace("+00:00", "Z")
                    writer.writerow({"decision_start_utc": origin,
                                     "target_interval_start_utc": target,
                                     "actual_aud_per_kwh": data.targets[part][i, col],
                                     **{f"{name}_aud_per_kwh": values[i]
                                        for name, values in forecasts.items()}})
    temporary.replace(path)
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha(path), "bytes": path.stat().st_size}


def _latency(model, window, repeats=50):
    model.eval()
    with torch.inference_mode():
        model(window)
        values = []
        for _ in range(repeats):
            start = perf_counter()
            output = model(window)
            values.append(perf_counter() - start)
    if output.shape != (1, len(HORIZONS)):
        raise ValueError("Single decision did not return ten prices")
    return {"median_ms": float(np.median(values) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "repeats": repeats,
            "scope": "CPU model forward on one preassembled scaled 72x41 window; excludes input construction and model loading"}


def evaluate(data, output_dir=OUTPUT_DIR):
    _setup()
    design = json.loads((output_dir / "design.json").read_text(encoding="utf-8"))
    if design["feature_files"] != data.sequence.feature_files or design["target_manifest_sha256"] != _sha(
            ROOT / "data/features/multi_horizon_v1/manifest.json"):
        raise ValueError("Neural design and frozen inputs differ")
    source_hashes = _source_hashes()
    source = json.loads((SHARED_DIR / "results.json").read_text(encoding="utf-8"))
    meta = {}
    models = {}
    timing = {}
    for name in NAMES:
        folder = output_dir / name
        m = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        checkpoint = folder / "model.pt"
        if (m["model_sha256"] != _sha(checkpoint) or
                m["target_manifest_sha256"] != design["target_manifest_sha256"] or
                m["feature_files"] != data.sequence.feature_files):
            raise ValueError("Neural model or training data fingerprint differs")
        model = make_multi_output_model(name)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        models[name] = model
        meta[name] = m
        end = int(data.indices["tune"][0])
        raw = (data.sequence.x[end - 71:end + 1] * data.sequence.scaler_scale
               + data.sequence.scaler_mean).astype(np.float32)
        reloaded = predict_saved(name, checkpoint, folder / "metadata.json",
                                 raw, FEATURE_FIELDS, [71])
        with torch.inference_mode():
            expected = model(torch.from_numpy(data.sequence.x[end - 71:end + 1][None])).numpy()
        if not np.allclose(reloaded, expected, atol=1e-5, rtol=1e-5):
            raise ValueError("Reloaded ten-output model inference differs")
        one = torch.from_numpy(data.sequence.x[end - 71:end + 1][None])
        timing[name] = _latency(model, one)
    # Tune predictions and provisional selection are completed before evaluation.
    predictions = {"tune": {}, "evaluate": {}}
    batch_times = {"tune": {}, "evaluate": {}}
    for name, model in models.items():
        start = perf_counter()
        predictions["tune"][f"multi_{name}"] = _predict(model, data, "tune")
        batch_times["tune"][name] = perf_counter() - start
    tune_mae = {name: float(np.mean(np.abs(values - data.targets["tune"]), axis=0).mean())
                for name, values in predictions["tune"].items()}
    selected_neural = min(NEW_NAMES, key=lambda name: tune_mae[name])
    base_errors = {name: [] for name in BASE_NAMES}
    for col, _ in enumerate(HORIZONS):
        reference = _read_references(data, "tune", col, source)
        for name in BASE_NAMES:
            base_errors[name].append(float(np.mean(np.abs(reference[name] - data.targets["tune"][:, col]))))
    all_tune_mae = {name: float(mean(values)) for name, values in base_errors.items()}
    all_tune_mae.update(tune_mae)
    selected_all = min(ALL_NAMES, key=lambda name: all_tune_mae[name])
    selection = {"selected_neural": selected_neural,
                 "selected_all_approaches": selected_all,
                 "selection_rule": "lowest equal-horizon mean January-June common-origin tuning MAE",
                 "tune_mean_mae": all_tune_mae, "frozen_before_reused_evaluation": True}
    _write_json_atomic(output_dir / "selection.json", selection)
    for name, model in models.items():
        start = perf_counter()
        predictions["evaluate"][f"multi_{name}"] = _predict(model, data, "evaluate")
        batch_times["evaluate"][name] = perf_counter() - start
    result = {"schema_version": 1, "experiment": "ten-output LSTM/GRU/TCN on 72-row historical sequences",
              "horizons_minutes": list(HORIZONS),
              "counts": {part: len(data.indices[part]) for part in data.indices},
              "sequence_exclusions": data.exclusions,
              "evaluation_status": "July-November 2025 reused development evaluation",
              "held_out_status": "December 2025-February 2026 untouched",
              "source_hashes": source_hashes, "design": design,
              "models": {name: {**meta[name], "single_decision_inference": timing[name],
                                "batch_inference_seconds": {part: batch_times[part][name]
                                                            for part in batch_times}}
                         for name in NAMES},
              "selection": selection, "splits": {}, "mean_mae": {}}
    for part in ("tune", "evaluate"):
        latest = _read_references(data, part, HORIZONS.index(60), source)["persistence"]
        x_for_scores = np.zeros((len(latest), len(FEATURE_FIELDS)), dtype=np.float32)
        x_for_scores[:, LATEST_INDEX] = latest
        matrix = Matrix(x_for_scores, data.targets[part], np.zeros_like(data.targets[part]),
                        np.ones(len(latest), dtype=bool),
                        [{"decision_start_utc": origin} for origin in data.origins[part]])
        matrices = {part: matrix}
        old_neural = _single_output_60(data, part)
        result["splits"][part] = {"row_count": len(latest), "horizons": {},
                                   "matched_single_output_60": {}}
        for col, h in enumerate(HORIZONS):
            forecasts = {**_read_references(data, part, col, source),
                         **{name: predictions[part][name][:, col] for name in NEW_NAMES}}
            if tuple(forecasts) != ALL_NAMES:
                raise ValueError("Comparison model ordering changed")
            mask = np.ones(len(latest), dtype=bool)
            metrics = _scores(matrices, part, col, mask, forecasts)
            metrics["prediction_file"] = _write_predictions(
                output_dir / f"h{h:03d}_{part}_predictions.csv.gz", data, part, col, forecasts)
            result["splits"][part]["horizons"][str(h)] = metrics
            print(f"Scored {part} +{h}m", flush=True)
            if h == 60:
                all_sixty = {**forecasts, **old_neural}
                result["splits"][part]["matched_single_output_60"] = _scores(
                    matrices, part, col, mask, all_sixty)["models"]
        result["mean_mae"][part] = {name: float(mean(result["splits"][part]["horizons"][str(h)]["models"][name]["mae"]
                                              for h in HORIZONS)) for name in ALL_NAMES}
    if _source_hashes() != source_hashes:
        raise ValueError("Prior forecasting artifacts changed during multi-output evaluation")
    result["process_peak_working_set_after_evaluation_mib"] = _peak_working_set_mib()
    _write_json_atomic(output_dir / "results.json", result)
    _write_json_atomic(ROOT / "data/models/multi_horizon_leaderboard.json", {
        "schema_version": 1, "task": "ten sparse five-minute interval-start NSW1 price horizons",
        "common_evaluation_origins": len(data.indices["evaluate"]),
        "selection": selection, "source_hashes": source_hashes,
        "entries": [{"name": name, "tune_mean_mae": result["mean_mae"]["tune"][name],
                     "reused_evaluation_mean_mae": result["mean_mae"]["evaluate"][name],
                     "model_artifact": (f"data/models/price_multi_output_neural_v1/{name[6:]}/model.pt"
                                        if name in NEW_NAMES else None)}
                    for name in ALL_NAMES],
        "one_hour_leaderboard_unchanged_sha256": source_hashes["one_hour_leaderboard"]})
    return result
