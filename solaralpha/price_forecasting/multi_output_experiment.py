"""Offline 72-row, ten-output LSTM/GRU/TCN price experiment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter, process_time

import numpy as np
import torch
from torch.utils.data import DataLoader

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .multi_horizon import HORIZONS, TARGET_DIR
from .multi_output_dataset import load_multi_output_data
from .multi_output_models import equal_horizon_mae, make_multi_output_model, model_specification
from .sequence_dataset import WINDOW
from .sequence_experiment import TRAIN_CONFIG, _peak_working_set_mib, _setup


OUTPUT_DIR = ROOT / "data" / "models" / "price_multi_output_neural_v1"
NAMES = ("lstm", "gru", "tcn")


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _loader(data, part, shuffle=False):
    generator = torch.Generator().manual_seed(TRAIN_CONFIG["seed"])
    return DataLoader(data.windows(part), batch_size=TRAIN_CONFIG["batch_size"],
                      shuffle=shuffle, generator=generator,
                      num_workers=TRAIN_CONFIG["data_loader_workers"])


def _predict(model, data, part):
    model.eval()
    result = []
    with torch.inference_mode():
        for x, _ in _loader(data, part):
            result.append(model(x).numpy())
    return np.concatenate(result).astype(np.float64)


def write_design(data=None, output_dir=OUTPUT_DIR):
    data = data or load_multi_output_data()
    output_dir.mkdir(parents=True, exist_ok=True)
    design = {"schema_version": 1, "architecture": model_specification(),
              "training": TRAIN_CONFIG, "input_shape": [WINDOW, len(FEATURE_FIELDS)],
              "output_shape": [len(HORIZONS)], "horizons_minutes": list(HORIZONS),
              "loss": "equal-weight mean absolute error across ten horizons and examples",
              "selection": "minimum mean of ten common January-June tuning horizon MAEs; 0.0001 minimum improvement, patience 2",
              "common_counts": {part: len(data.indices[part]) for part in data.indices},
              "sequence_exclusions": data.exclusions,
              "feature_files": data.sequence.feature_files,
              "target_manifest_sha256": _sha(TARGET_DIR / "manifest.json"),
              "bridge_source_sha256": data.sequence.bridge_source_sha256,
              "scaler_fit": "original 208,800 training feature rows only",
              "fixed_feature_semantics": "+60-aligned historical daily/weekly price input columns are unchanged",
              "held_out": "December 2025-February 2026 untouched"}
    _write_json_atomic(output_dir / "design.json", design)
    return design


def short_benchmark(data, batches=4):
    _setup()
    result = {}
    for name in NAMES:
        torch.manual_seed(TRAIN_CONFIG["seed"])
        model = make_multi_output_model(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                                      weight_decay=TRAIN_CONFIG["weight_decay"])
        loader = _loader(data, "train", shuffle=True)
        start = perf_counter()
        for i, (x, y) in enumerate(loader):
            if i >= batches:
                break
            optimizer.zero_grad(set_to_none=True)
            loss = equal_horizon_mae(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip_norm"])
            optimizer.step()
        elapsed = perf_counter() - start
        result[name] = {"batches": batches, "seconds": elapsed,
                        "estimated_epoch_seconds": elapsed * len(loader) / batches,
                        "parameters": model_specification()[name]["trainable_parameters"]}
        print(f"{name}: {elapsed:.2f}s for {batches} batches", flush=True)
    _write_json_atomic(OUTPUT_DIR / "short_benchmark.json", result)
    return result


def train_one(name, data, output_dir=OUTPUT_DIR):
    if name not in NAMES:
        raise ValueError(name)
    _setup()
    torch.manual_seed(TRAIN_CONFIG["seed"])
    model = make_multi_output_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                                  weight_decay=TRAIN_CONFIG["weight_decay"])
    train_loader = _loader(data, "train", shuffle=True)
    path = output_dir / name
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "model.pt"
    curves = {"train_mae": [], "tune_mae": [], "tune_mae_by_horizon": []}
    best, best_epoch, stale = float("inf"), 0, 0
    wall_start, cpu_start = perf_counter(), process_time()
    for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
        model.train()
        absolute_sum = 0.0
        count = 0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = equal_horizon_mae(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip_norm"])
            optimizer.step()
            absolute_sum += float(torch.abs(prediction.detach() - y).sum())
            count += len(y)
        train_mae = absolute_sum / (count * len(HORIZONS))
        forecast = _predict(model, data, "tune")
        by_horizon = np.mean(np.abs(forecast - data.targets["tune"]), axis=0)
        tune_mae = float(by_horizon.mean())
        curves["train_mae"].append(train_mae)
        curves["tune_mae"].append(tune_mae)
        curves["tune_mae_by_horizon"].append(by_horizon.tolist())
        print(f"{name} epoch {epoch}: train {train_mae:.5f}, tune {tune_mae:.5f}", flush=True)
        if best - tune_mae >= TRAIN_CONFIG["minimum_mae_improvement_aud_per_kwh"]:
            best, best_epoch, stale = tune_mae, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    wall_seconds = perf_counter() - wall_start
    cpu_seconds = process_time() - cpu_start
    meta = {"name": name, "architecture": model_specification()[name],
            "training": TRAIN_CONFIG, "feature_order": list(FEATURE_FIELDS),
            "horizons_minutes": list(HORIZONS), "sequence_length": WINDOW,
            "feature_files": data.sequence.feature_files,
            "target_manifest_sha256": _sha(TARGET_DIR / "manifest.json"),
            "bridge_source_sha256": data.sequence.bridge_source_sha256,
            "scaler_mean": data.sequence.scaler_mean.tolist(),
            "scaler_scale": data.sequence.scaler_scale.tolist(),
            "train_count": len(data.indices["train"]), "tune_count": len(data.indices["tune"]),
            "sequence_exclusions": data.exclusions,
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "best_epoch_one_based": best_epoch, "best_tune_mean_mae": best,
            "epochs_fitted": len(curves["tune_mae"]), "learning_curves": curves,
            "training_wall_seconds": wall_seconds, "training_cpu_seconds": cpu_seconds,
            "effective_cpu_cores": cpu_seconds / wall_seconds,
            "process_peak_working_set_mib": _peak_working_set_mib(),
            "gpu_peak_allocated_mib": 0.0, "device": "cpu",
            "torch_version": torch.__version__, "python": platform.python_version(),
            "model_sha256": _sha(checkpoint), "model_bytes": checkpoint.stat().st_size}
    _write_json_atomic(path / "metadata.json", meta)
    return meta


def predict_saved(name, checkpoint, metadata_path, raw_matrix, feature_names,
                  end_indices, batch_size=512):
    """Reload ten-output head and training-only scaler, checking model identity."""
    if tuple(feature_names) != FEATURE_FIELDS or raw_matrix.ndim != 2 or raw_matrix.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("Multi-output input order or width changed")
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    if (meta["name"] != name or tuple(meta["feature_order"]) != FEATURE_FIELDS or
            tuple(meta["horizons_minutes"]) != HORIZONS or _sha(checkpoint) != meta["model_sha256"]):
        raise ValueError("Saved multi-output identity/schema/checksum mismatch")
    scaled = ((raw_matrix - np.asarray(meta["scaler_mean"])) /
              np.asarray(meta["scaler_scale"])).astype(np.float32)
    model = make_multi_output_model(name)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(end_indices), batch_size):
            ends = end_indices[start:start + batch_size]
            windows = np.stack([scaled[int(end)-WINDOW+1:int(end)+1] for end in ends])
            result.append(model(torch.from_numpy(windows)).numpy())
    return np.concatenate(result).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "benchmark", "train", "evaluate"))
    parser.add_argument("--model", choices=NAMES)
    args = parser.parse_args()
    data = load_multi_output_data()
    if args.stage == "design":
        print(json.dumps(write_design(data), indent=2))
    elif args.stage == "benchmark":
        write_design(data)
        short_benchmark(data)
    elif args.stage == "train":
        write_design(data)
        if not args.model:
            parser.error("--model is required for train")
        meta = train_one(args.model, data)
        print(json.dumps({k: meta[k] for k in ("best_epoch_one_based", "best_tune_mean_mae", "training_wall_seconds")}, indent=2))
    else:
        from .multi_output_evaluation import evaluate
        result = evaluate(data)
        print(json.dumps(result["mean_mae"], indent=2))


if __name__ == "__main__":
    main()
