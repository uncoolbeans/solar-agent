"""Offline CPU training and matched evaluation of compact LSTM, GRU and TCN."""

import argparse
import csv
import gzip
from hashlib import sha256
from io import TextIOWrapper
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter, process_time

import numpy as np
import torch
from torch.utils.data import DataLoader

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .ml_dataset import Slice
from .ml_metrics import full_scores, grouped_scores, movement_diagnostics
from .sequence_dataset import WINDOW, load_sequence_data
from .sequence_models import MODEL_SPECS, make_model, parameter_count
from .tabular_comparison import (ALL_NAMES as TABULAR_NAMES, LEADERBOARD_PATH,
                                 OUTPUT_DIR as TABULAR_DIR,
                                 _conditional_scores, _subperiod_scores)


OUTPUT_DIR = ROOT / "data" / "models" / "price_sequence_comparison"
NEURAL_NAMES = ("lstm", "gru", "tcn")
ALL_NAMES = TABULAR_NAMES + NEURAL_NAMES
TRAIN_CONFIG = {"seed": 42, "batch_size": 512, "learning_rate": 0.001,
                "weight_decay": 0.0001, "max_epochs": 8, "patience": 2,
                "minimum_mae_improvement_aud_per_kwh": 0.0001,
                "gradient_clip_norm": 1.0, "torch_threads": 4,
                "data_loader_workers": 0, "loss": "MAE", "optimizer": "AdamW",
                "device": "cpu"}


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read_tabular_predictions(split, portion):
    """Carry all fourteen frozen columns across on identical timestamps and targets."""
    result = {name: np.empty(len(portion.y), dtype=np.float64) for name in TABULAR_NAMES}
    path = TABULAR_DIR / f"{split}_comparison.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        count = 0
        for count, (record, row, actual) in enumerate(
                zip(reader, portion.rows, portion.y, strict=True), 1):
            if (record["decision_start_utc"] != row["decision_start_utc"] or
                    record["target_interval_start_utc"] != row["target_interval_start_utc"] or
                    float(record["actual_aud_per_kwh"]) != actual):
                raise ValueError("Frozen 14-model predictions do not match sequence origins")
            for name in TABULAR_NAMES:
                result[name][count - 1] = float(record[f"{name}_aud_per_kwh"])
        if count != len(portion.y):
            raise ValueError("Frozen 14-model row count differs")
    return result


def _setup():
    torch.manual_seed(TRAIN_CONFIG["seed"])
    np.random.seed(TRAIN_CONFIG["seed"])
    torch.set_num_threads(TRAIN_CONFIG["torch_threads"])
    torch.use_deterministic_algorithms(True)


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
    result = get_info(current(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2**20 if result else None


def _loader(data, part, *, shuffle=False):
    generator = torch.Generator().manual_seed(TRAIN_CONFIG["seed"])
    return DataLoader(data.windows(part), batch_size=TRAIN_CONFIG["batch_size"],
                      shuffle=shuffle, generator=generator,
                      num_workers=TRAIN_CONFIG["data_loader_workers"])


def _predict(model, loader):
    model.eval()
    result = []
    with torch.inference_mode():
        for x, _ in loader:
            result.append(model(x).cpu().numpy())
    return np.concatenate(result).astype(np.float64)


def short_benchmark(data, batches=4):
    """Prespecified hardware check: four training steps, without selecting a model."""
    _setup()
    report = {}
    for name in NEURAL_NAMES:
        torch.manual_seed(TRAIN_CONFIG["seed"])
        model = make_model(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                                      weight_decay=TRAIN_CONFIG["weight_decay"])
        loader = _loader(data, "train", shuffle=True)
        start = perf_counter()
        for index, (x, y) in enumerate(loader):
            if index >= batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.abs(model(x) - y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           TRAIN_CONFIG["gradient_clip_norm"])
            optimizer.step()
        elapsed = perf_counter() - start
        report[name] = {"batches": batches, "batch_size": TRAIN_CONFIG["batch_size"],
                        "elapsed_seconds": elapsed,
                        "extrapolated_epoch_seconds": elapsed * len(loader) / batches,
                        "parameters": parameter_count(model)}
        print(f"{name} short benchmark: {elapsed:.2f}s for {batches} batches", flush=True)
    return report


def train_one(name, data, output_dir=OUTPUT_DIR):
    _setup()
    torch.manual_seed(TRAIN_CONFIG["seed"])
    model = make_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                                  weight_decay=TRAIN_CONFIG["weight_decay"])
    train_loader = _loader(data, "train", shuffle=True)
    tune_loader = _loader(data, "tune")
    path = output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "model.pt"
    curves = {"train_mae": [], "tune_mae": []}
    best_mae = float("inf")
    best_epoch = 0
    stale = 0
    wall_start, cpu_start = perf_counter(), process_time()
    for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
        model.train()
        absolute_sum = 0.0
        count = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.mean(torch.abs(prediction - y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           TRAIN_CONFIG["gradient_clip_norm"])
            optimizer.step()
            absolute_sum += float(torch.sum(torch.abs(prediction.detach() - y)))
            count += len(y)
        train_mae = absolute_sum / count
        tune_pred = _predict(model, tune_loader)
        tune_mae = float(np.mean(np.abs(tune_pred - data.targets["tune"])))
        curves["train_mae"].append(train_mae)
        curves["tune_mae"].append(tune_mae)
        print(f"{name} epoch {epoch}: train {train_mae:.5f}, tune {tune_mae:.5f}", flush=True)
        if best_mae - tune_mae >= TRAIN_CONFIG["minimum_mae_improvement_aud_per_kwh"]:
            best_mae, best_epoch, stale = tune_mae, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    wall_seconds = perf_counter() - wall_start
    cpu_seconds = process_time() - cpu_start
    metadata = {"name": name, "architecture": MODEL_SPECS[name],
                "training": TRAIN_CONFIG, "feature_order": list(FEATURE_FIELDS),
                "feature_files": data.feature_files,
                "bridge_source_sha256": data.bridge_source_sha256,
                "scaler_mean": data.scaler_mean.tolist(),
                "scaler_scale": data.scaler_scale.tolist(),
                "parameter_count": parameter_count(model),
                "train_count": len(data.indices["train"]),
                "tune_count": len(data.indices["tune"]),
                "sequence_warmup_exclusions": data.exclusions,
                "best_epoch_one_based": best_epoch, "best_tune_mae": best_mae,
                "epochs_fitted": len(curves["tune_mae"]), "learning_curves": curves,
                "training_wall_seconds": wall_seconds,
                "training_cpu_seconds": cpu_seconds,
                "effective_cpu_cores": cpu_seconds / wall_seconds,
                "process_peak_working_set_mib": _peak_working_set_mib(),
                "gpu_peak_allocated_mib": 0.0,
                "device": "cpu", "torch_version": torch.__version__,
                "python": platform.python_version(),
                "model_sha256": _hash(checkpoint),
                "model_bytes": checkpoint.stat().st_size}
    _write_json_atomic(path / "metadata.json", metadata)
    return metadata


def predict_saved(name, checkpoint, metadata_path, raw_matrix, feature_names,
                  end_indices, batch_size=512):
    """For offline inference, scale with saved training statistics and check schema/hash."""
    if tuple(feature_names) != FEATURE_FIELDS or raw_matrix.ndim != 2 or raw_matrix.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("Neural feature order or width differs from training")
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if (meta["name"] != name or tuple(meta["feature_order"]) != FEATURE_FIELDS or
            _hash(checkpoint) != meta["model_sha256"]):
        raise ValueError("Neural saved model identity, schema or checksum mismatch")
    mean = np.asarray(meta["scaler_mean"])
    scale = np.asarray(meta["scaler_scale"])
    scaled = ((raw_matrix - mean) / scale).astype(np.float32)
    model = make_model(name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(end_indices), batch_size):
            indices = end_indices[start:start + batch_size]
            windows = np.stack([scaled[int(end)-WINDOW+1:int(end)+1] for end in indices])
            result.append(model(torch.from_numpy(windows)).numpy())
    return np.concatenate(result).astype(np.float64)


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


def evaluate_all(data, output_dir=OUTPUT_DIR, leaderboard_path=LEADERBOARD_PATH):
    """Only after training finishes: inspect reused July-November and extend leaderboard."""
    _setup()
    base_path = output_dir / "base_14_model_leaderboard.json"
    if not base_path.exists():
        prior = json.loads(leaderboard_path.read_text(encoding="utf-8"))
        if len(prior["entries"]) != 14:
            raise ValueError("First evaluation needs the frozen 14-model leaderboard")
        shutil.copyfile(leaderboard_path, base_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    if len(base["entries"]) != 14 or base["feature_files"] != data.feature_files:
        raise ValueError("Frozen leaderboard or feature fingerprints changed")
    predictions = {"tune": {}, "evaluate": {}}
    timing = {}
    metadata = {}
    for name in NEURAL_NAMES:
        path = output_dir / name
        checkpoint = path / "model.pt"
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta["feature_files"] != data.feature_files:
            raise ValueError("Model trained on different feature files")
        metadata[name] = meta
        model = make_model(name)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        model.eval()
        timing[name] = {}
        for part in ("tune", "evaluate"):
            loader = _loader(data, part)
            model(torch.zeros((1, WINDOW, len(FEATURE_FIELDS))))
            start = perf_counter()
            prediction = _predict(model, loader)
            batch_seconds = perf_counter() - start
            one = torch.from_numpy(data.x[int(data.indices[part][0])-WINDOW+1:
                                          int(data.indices[part][0])+1][None, :, :])
            samples = []
            with torch.inference_mode():
                for _ in range(50):
                    start = perf_counter()
                    model(one)
                    samples.append(perf_counter() - start)
            predictions[part][name] = prediction
            timing[name][part] = {"batch_seconds": batch_seconds,
                                  "batch_microseconds_per_row": 1e6 * batch_seconds / len(prediction),
                                  "single_observation_median_microseconds":
                                      1e6 * float(np.median(samples)),
                                  "single_observation_repeats": len(samples)}
            # Reload using the separate public inference path and original feature units.
            raw = data.x[:int(data.indices[part][7])+1] * data.scaler_scale + data.scaler_mean
            saved = predict_saved(name, checkpoint, path / "metadata.json", raw,
                                  FEATURE_FIELDS, data.indices[part][:8])
            if not np.allclose(saved, prediction[:8], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Reloaded {name} inference differs")
        print(f"{name}: tune MAE {np.mean(np.abs(predictions['tune'][name]-data.targets['tune'])):.5f}",
              flush=True)
    result = {"experiment": "Fixed compact LSTM, GRU, TCN one-hour price comparison",
              "evaluation_status": "July-November 2025 reused development evaluation",
              "held_out_status": "December 2025-February 2026 untouched",
              "feature_files": data.feature_files,
              "bridge_source_sha256": data.bridge_source_sha256,
              "original_leaderboard_sha256": _hash(base_path),
              "sequence_length": WINDOW, "target_horizon_minutes": 60,
              "target_resolution_minutes": 5,
              "counts": {part: len(data.indices[part]) for part in ("train", "tune", "evaluate")},
              "exclusions": data.exclusions,
              "models": {name: {**metadata[name], "inference": timing[name]}
                         for name in NEURAL_NAMES}, "splits": {}}
    observation_path = output_dir / "resource_observation.json"
    if observation_path.exists():
        result["training_process_resource_observation"] = json.loads(
            observation_path.read_text(encoding="utf-8-sig"))
    for part in ("tune", "evaluate"):
        portion = Slice(None, data.targets[part], data.rows[part])
        forecasts = {**read_tabular_predictions(part, portion), **predictions[part]}
        if tuple(forecasts) != ALL_NAMES:
            raise ValueError("Matched 17-model forecast order changed")
        result["splits"][part] = {
            "row_count": len(portion.y),
            "models": {name: full_scores(portion.y, values)
                       for name, values in forecasts.items()},
            "actual_price_ranges": _conditional_scores(portion.y, forecasts),
            "monthly": grouped_scores(portion.y, forecasts, portion.rows, "target_local_month"),
            "hour_of_day": grouped_scores(portion.y, forecasts, portion.rows, "target_local_hour"),
            "subperiods": _subperiod_scores(portion, forecasts),
            "movement": movement_diagnostics(portion.y, forecasts, portion.rows),
            "predictions_file": _write_predictions(output_dir / f"{part}_comparison.csv.gz",
                                                     portion, forecasts)}
    _write_json_atomic(output_dir / "results.json", result)
    old_entries = base["entries"]
    if [entry["name"] for entry in old_entries] != list(TABULAR_NAMES):
        raise ValueError("Original leaderboard model order changed")
    new_entries = []
    for name in NEURAL_NAMES:
        detail = result["models"][name]
        rt = detail["inference"]["evaluate"]
        new_entries.append({"name": name, "family": name,
                            "target_type": "absolute_wholesale_price",
                            "loss_function": "MAE",
                            "configuration": {"architecture": MODEL_SPECS[name],
                                              "training": TRAIN_CONFIG,
                                              "sequence_length": WINDOW},
                            "tuning": result["splits"]["tune"]["models"][name],
                            "reused_development_evaluation":
                                result["splits"]["evaluate"]["models"][name],
                            "runtime": {"training_seconds": detail["training_wall_seconds"],
                                        "evaluation_batch_seconds": rt["batch_seconds"],
                                        "evaluation_batch_microseconds_per_row":
                                            rt["batch_microseconds_per_row"],
                                        "evaluation_single_median_microseconds":
                                            rt["single_observation_median_microseconds"],
                                        "single_observation_repeats": 50,
                                        "scope": ("CPU; batch includes lazy window assembly and "
                                                  "DataLoader collation; single uses preassembled window")},
                            "parameter_count": detail["parameter_count"],
                            "model_artifact": f"data/models/price_sequence_comparison/{name}/model.pt",
                            "model_bytes": detail["model_bytes"],
                            "multi_horizon": None,
                            "downstream_mpc_financial_evaluation": None})
    board = {**base, "schema_version": 2, "entries": old_entries + new_entries,
             "sequence_comparison": {"input_rows": WINDOW,
                                     "first_train_warmup_exclusions": data.exclusions["train"],
                                     "original_14_model_sha256": _hash(base_path)}}
    _write_json_atomic(leaderboard_path, board)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--model", choices=(*NEURAL_NAMES, "all"), default="all")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    data = load_sequence_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.benchmark:
        _write_json_atomic(OUTPUT_DIR / "short_benchmark.json", short_benchmark(data))
        return
    if not args.evaluate:
        for name in (NEURAL_NAMES if args.model == "all" else (args.model,)):
            train_one(name, data)
    if args.evaluate or args.model == "all":
        result = evaluate_all(data)
        print(json.dumps({name: {part: result["splits"][part]["models"][name]["mae"]
                                for part in ("tune", "evaluate")}
                          for name in NEURAL_NAMES}, indent=2))


if __name__ == "__main__":
    main()
