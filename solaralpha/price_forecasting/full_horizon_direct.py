"""Resumable direct absolute-error XGBoost models for all 144 offsets."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import xgboost as xgb

from solaralpha.historical_pipeline import ROOT, _write_json_atomic
from .features import FEATURE_FIELDS
from .full_horizon_dataset import HORIZONS, TARGET_DIR, load_targets
from .multi_horizon import HORIZONS as SPARSE_HORIZONS, MODEL_DIR as SPARSE_DIR
from .xgb_v1 import PARAMETERS


OUTPUT_DIR = ROOT / "data/models/price_full_horizon_v1"
DIRECT_DIR = OUTPUT_DIR / "direct"
PARAMS = {**PARAMETERS, "objective": "reg:absoluteerror"}


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def model_paths(horizon, output_dir=OUTPUT_DIR):
    if horizon not in HORIZONS:
        raise ValueError("Unknown dense horizon")
    if horizon in SPARSE_HORIZONS:
        folder = SPARSE_DIR / f"h{horizon:03d}"
    else:
        folder = output_dir / "direct" / f"h{horizon:03d}"
    return folder / "model.json", folder / "metadata.json"


def verified_model(horizon, data, output_dir=OUTPUT_DIR):
    path, metadata_path = model_paths(horizon, output_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (_sha(path) != metadata["model_sha256"] or
            tuple(metadata["feature_order"]) != FEATURE_FIELDS or
            metadata["parameters"] != PARAMS or
            metadata["feature_files"] != data.manifest["feature_files"] or
            metadata["horizon_minutes"] != horizon):
        raise ValueError(f"Direct +{horizon} model identity differs")
    col = HORIZONS.index(horizon)
    for part in ("train", "tune"):
        count = int(np.isfinite(data.targets[part][:, col]).sum())
        if horizon in SPARSE_HORIZONS:
            old = SPARSE_HORIZONS.index(horizon)
            if not np.array_equal(data.targets[part][:, col], data.matrices[part].y[:, old], equal_nan=True):
                raise ValueError("Frozen direct model labels differ")
        elif metadata[f"{part}_count"] != count or metadata["target_manifest_sha256"] != _sha(TARGET_DIR / "manifest.json"):
            raise ValueError("Direct model target fingerprint differs")
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model, metadata


def train_range(start=0, stop=144, data=None, output_dir=OUTPUT_DIR):
    if not (0 <= start <= stop <= len(HORIZONS)):
        raise ValueError("Invalid resumable batch")
    data = data or load_targets()
    output_dir.mkdir(parents=True, exist_ok=True)
    train, tune = data.matrices["train"], data.matrices["tune"]
    for col in range(start, stop):
        horizon = HORIZONS[col]
        path, metadata_path = model_paths(horizon, output_dir)
        if horizon in SPARSE_HORIZONS or metadata_path.exists():
            verified_model(horizon, data, output_dir)
            print(f"+{horizon}m verified existing", flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        train_mask = np.isfinite(data.targets["train"][:, col])
        tune_mask = np.isfinite(data.targets["tune"][:, col])
        model = xgb.XGBRegressor(**PARAMS)
        started = perf_counter()
        model.fit(train.x[train_mask], np.asarray(data.targets["train"][train_mask, col]),
                  eval_set=[(train.x[train_mask], np.asarray(data.targets["train"][train_mask, col])),
                            (tune.x[tune_mask], np.asarray(data.targets["tune"][tune_mask, col]))],
                  verbose=False)
        fit_seconds = perf_counter() - started
        model.save_model(path)
        metadata = {"schema_version": 1, "horizon_minutes": horizon,
                    "feature_order": list(FEATURE_FIELDS),
                    "feature_files": data.manifest["feature_files"],
                    "target_manifest_sha256": _sha(TARGET_DIR / "manifest.json"),
                    "parameters": PARAMS, "train_count": int(train_mask.sum()),
                    "tune_count": int(tune_mask.sum()),
                    "best_iteration_zero_based": int(model.best_iteration),
                    "best_tune_mae": float(model.best_score),
                    "fitted_rounds": len(model.evals_result()["validation_1"]["mae"]),
                    "train_seconds": fit_seconds, "model_bytes": path.stat().st_size,
                    "model_sha256": _sha(path), "python": platform.python_version(),
                    "xgboost": xgb.__version__}
        _write_json_atomic(metadata_path, metadata)
        print(f"+{horizon}m trained: {fit_seconds:.1f}s, best tuning MAE {model.best_score:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0, help="inclusive horizon index")
    parser.add_argument("--stop", type=int, default=144, help="exclusive horizon index")
    args = parser.parse_args()
    train_range(args.start, args.stop)


if __name__ == "__main__":
    main()
