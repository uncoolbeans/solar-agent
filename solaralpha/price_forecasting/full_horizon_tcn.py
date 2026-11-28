"""Frozen compact causal TCN with one output per five-minute target start."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter, process_time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .full_horizon_dataset import HORIZONS, TARGET_DIR, load_targets
from .multi_output_dataset import load_multi_output_data
from .sequence_dataset import WINDOW, WindowDataset
from .sequence_experiment import TRAIN_CONFIG, _peak_working_set_mib, _setup
from .sequence_models import make_model


OUTPUT_DIR = ROOT / "data/models/price_full_horizon_v1/tcn"


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def make_model_144():
    model = make_model("tcn")
    model.output = nn.Linear(24, len(HORIZONS))
    if model.receptive_field < WINDOW:
        raise ValueError("TCN receptive field is too short")
    return model


def load_sequence_targets(full=None, sparse=None):
    full = full or load_targets()
    sparse = sparse or load_multi_output_data()
    labels = {}
    for part in ("train", "tune", "evaluate"):
        indices = np.flatnonzero(full.common[part])
        if part == "train":
            indices = indices[71:]
        origins = [full.matrices[part].rows[i]["decision_start_utc"] for i in indices]
        if origins != sparse.origins[part]:
            raise ValueError("Dense target origins do not match frozen 72-row windows")
        labels[part] = np.asarray(full.targets[part][indices], dtype=np.float32)
        if labels[part].shape != (len(sparse.indices[part]), len(HORIZONS)):
            raise ValueError("TCN output labels are not 144-wide")
    return sparse, labels


def write_design(full=None, sparse=None, output_dir=OUTPUT_DIR):
    full = full or load_targets()
    sparse, labels = load_sequence_targets(full, sparse)
    model = make_model_144()
    spec = {"schema_version": 1, "architecture": "existing seven-block causal residual TCN",
            "input_shape": [WINDOW, len(FEATURE_FIELDS)], "output_shape": [len(HORIZONS)],
            "receptive_field_rows": model.receptive_field,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "horizons_minutes": list(HORIZONS), "training": TRAIN_CONFIG,
            "loss": "mean absolute error across all common origins and all 144 horizons",
            "feature_order": list(FEATURE_FIELDS),
            "feature_files": full.manifest["feature_files"],
            "target_manifest_sha256": _sha(TARGET_DIR / "manifest.json"),
            "sequence_counts": {part: len(labels[part]) for part in labels},
            "sequence_warmup_exclusions": {"train": 71, "tune": 0, "evaluate": 0},
            "scaler_fit": "frozen original 208,800 training feature rows",
            "held_out": "December 2025-February 2026 untouched"}
    if spec["trainable_parameters"] != 13656:
        raise ValueError("TCN architecture changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "design.json", spec)
    return spec


def _loader(sparse, labels, part, shuffle=False):
    dataset = WindowDataset(sparse.sequence.x, sparse.indices[part], labels[part])
    return DataLoader(dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=shuffle,
                      generator=torch.Generator().manual_seed(TRAIN_CONFIG["seed"]),
                      num_workers=TRAIN_CONFIG["data_loader_workers"])


def _mae_by_horizon(model, loader):
    model.eval()
    absolute = np.zeros(len(HORIZONS), dtype=np.float64)
    count = 0
    with torch.inference_mode():
        for x, y in loader:
            absolute += torch.abs(model(x) - y).sum(dim=0).numpy()
            count += len(y)
    return absolute / count


def train(full=None, sparse=None, output_dir=OUTPUT_DIR):
    _setup()
    full = full or load_targets()
    sparse, labels = load_sequence_targets(full, sparse)
    design = write_design(full, sparse, output_dir)
    torch.manual_seed(TRAIN_CONFIG["seed"])
    model = make_model_144()
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                                  weight_decay=TRAIN_CONFIG["weight_decay"])
    curves = {"train_mae": [], "tune_mean_mae": [], "tune_by_horizon": []}
    checkpoint = output_dir / "model.pt"
    best, best_epoch, stale = float("inf"), 0, 0
    wall_start, cpu_start = perf_counter(), process_time()
    for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
        model.train()
        absolute, count = 0.0, 0
        for x, y in _loader(sparse, labels, "train", shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = torch.abs(prediction - y).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip_norm"])
            optimizer.step()
            absolute += float(torch.abs(prediction.detach() - y).sum())
            count += len(y)
        tune_by_horizon = _mae_by_horizon(model, _loader(sparse, labels, "tune"))
        tune_mean = float(tune_by_horizon.mean())
        curves["train_mae"].append(absolute / (count * len(HORIZONS)))
        curves["tune_mean_mae"].append(tune_mean)
        curves["tune_by_horizon"].append(tune_by_horizon.tolist())
        print(f"TCN epoch {epoch}: train {curves['train_mae'][-1]:.5f}, tune {tune_mean:.5f}", flush=True)
        if best - tune_mean >= TRAIN_CONFIG["minimum_mae_improvement_aud_per_kwh"]:
            best, best_epoch, stale = tune_mean, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= TRAIN_CONFIG["patience"]:
                break
    metadata = {**design, "best_epoch_one_based": best_epoch,
                "best_tune_mean_mae": best, "epochs_fitted": len(curves["train_mae"]),
                "learning_curves": curves,
                "training_wall_seconds": perf_counter() - wall_start,
                "training_cpu_seconds": process_time() - cpu_start,
                "process_peak_working_set_mib": _peak_working_set_mib(),
                "model_bytes": checkpoint.stat().st_size, "model_sha256": _sha(checkpoint),
                "device": "cpu", "python": platform.python_version(),
                "torch": torch.__version__,
                "scaler_mean": sparse.sequence.scaler_mean.tolist(),
                "scaler_scale": sparse.sequence.scaler_scale.tolist()}
    _write_json_atomic(output_dir / "metadata.json", metadata)
    return metadata


def load_saved(output_dir=OUTPUT_DIR):
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = output_dir / "model.pt"
    if (_sha(checkpoint) != metadata["model_sha256"] or
            tuple(metadata["feature_order"]) != FEATURE_FIELDS or
            tuple(metadata["horizons_minutes"]) != HORIZONS):
        raise ValueError("Saved dense TCN identity differs")
    model = make_model_144()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    return model, metadata


def predict_saved(raw_matrix, feature_order, end_indices, *, output_dir=OUTPUT_DIR):
    if tuple(feature_order) != FEATURE_FIELDS or raw_matrix.ndim != 2 or raw_matrix.shape[1] != len(FEATURE_FIELDS):
        raise ValueError("Dense TCN feature order or width differs")
    model, meta = load_saved(output_dir)
    scaled = ((raw_matrix - np.asarray(meta["scaler_mean"])) /
              np.asarray(meta["scaler_scale"])).astype(np.float32)
    predictions = []
    with torch.inference_mode():
        for end in end_indices:
            end = int(end)
            if end < WINDOW - 1 or end >= len(scaled):
                raise ValueError("Incomplete dense TCN input history")
            predictions.append(model(torch.from_numpy(scaled[end-WINDOW+1:end+1][None])).numpy()[0])
    return np.asarray(predictions, dtype=np.float64).reshape(-1, len(HORIZONS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("design", "train"))
    args = parser.parse_args()
    if args.stage == "design":
        print(json.dumps(write_design(), indent=2))
    else:
        metadata = train()
        print(json.dumps({k: metadata[k] for k in ("best_epoch_one_based", "best_tune_mean_mae", "training_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
