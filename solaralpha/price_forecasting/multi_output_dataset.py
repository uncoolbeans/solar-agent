"""Lazy 72-row inputs paired with ten decision-time-safe price labels."""

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json

import numpy as np

from solaralpha.historical_pipeline import ROOT
from .multi_horizon import HORIZONS, TARGET_DIR, TARGET_HEADER
from .sequence_dataset import WINDOW, WindowDataset, load_sequence_data


@dataclass
class MultiOutputData:
    sequence: object
    indices: dict
    targets: dict
    origins: dict
    exclusions: dict
    target_manifest: dict

    def windows(self, part):
        return WindowDataset(self.sequence.x, self.indices[part], self.targets[part])


def _minute(value):
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.utcoffset() != timedelta(0):
        raise ValueError("Target origin timestamp is not UTC")
    return int(instant.timestamp() // 60)


def load_multi_output_data(sequence=None, target_dir=TARGET_DIR):
    """Align common ten-target origins to frozen sequence ends; no window copies."""
    seq = sequence or load_sequence_data()
    manifest_path = target_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["horizons_minutes"]) != HORIZONS or manifest["feature_files"] != seq.feature_files:
        raise ValueError("Multi-output targets differ from frozen sequence features")
    allowed = {part: {int(seq.minutes[index]): (int(index), j)
                      for j, index in enumerate(seq.indices[part])}
               for part in ("train", "tune", "evaluate")}
    output = {part: {"indices": [], "targets": [], "origins": []}
              for part in allowed}
    previous = {"train": None, "validation": None}
    for split in ("train", "validation"):
        path = target_dir / f"{split}_targets.csv.gz"
        if sha256(path.read_bytes()).hexdigest() != manifest["files"][split]["sha256"]:
            raise ValueError("Multi-horizon target checksum mismatch")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames) != TARGET_HEADER:
                raise ValueError("Multi-horizon target schema changed")
            for row in reader:
                minute = _minute(row["decision_start_utc"])
                if previous[split] is not None and minute <= previous[split]:
                    raise ValueError("Target origins are not strictly chronological")
                previous[split] = minute
                if row["common_eligible"] != "1":
                    continue
                part = row["period"]
                if part not in output or (split == "train") != (part == "train"):
                    raise ValueError("Common origin in the wrong chronological partition")
                found = allowed[part].get(minute)
                if found is None:
                    if part == "train" and minute < int(seq.minutes[seq.indices["train"][0]]):
                        continue  # first 71 sequence origins lack a complete 72-row context
                    raise ValueError("Common target origin lacks a valid sequence window")
                end, selected_position = found
                if end - WINDOW + 1 < 0 or seq.minutes[end] - seq.minutes[end - WINDOW + 1] != 355:
                    raise ValueError("Sequence window is incomplete")
                values = np.asarray([float(row[f"price_h{h}_aud_per_kwh"]) for h in HORIZONS], dtype=np.float64)
                if not np.isfinite(values).all() or values[HORIZONS.index(60)] != seq.targets[part][selected_position]:
                    raise ValueError("Ten targets are missing or +60 differs from frozen sequence")
                output[part]["indices"].append(end)
                output[part]["targets"].append(values)
                output[part]["origins"].append(row["decision_start_utc"])
    indices = {part: np.asarray(bucket["indices"], dtype=np.int32) for part, bucket in output.items()}
    targets = {part: np.asarray(bucket["targets"], dtype=np.float64) for part, bucket in output.items()}
    origins = {part: bucket["origins"] for part, bucket in output.items()}
    exclusions = {part: manifest["common_counts"][part] - len(indices[part]) for part in output}
    if exclusions != {"train": 71, "tune": 0, "evaluate": 0}:
        raise ValueError(f"Unexpected multi-output sequence exclusions: {exclusions}")
    for part in output:
        if targets[part].shape != (len(indices[part]), len(HORIZONS)) or np.any(np.diff(indices[part]) <= 0):
            raise ValueError("Target shape or sequence ordering changed")
    return MultiOutputData(seq, indices, targets, origins, exclusions, manifest)
