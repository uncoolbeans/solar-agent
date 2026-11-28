"""Chronology, ten-target alignment, shared-head shape and saved inference."""

import csv
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json
import unittest

import numpy as np
import torch

from solaralpha.historical_pipeline import ROOT, sydney_zone
from solaralpha.price_forecasting.features import FEATURE_FIELDS
from solaralpha.price_forecasting.multi_horizon import HORIZONS, TARGET_DIR
from solaralpha.price_forecasting.multi_output_dataset import load_multi_output_data
from solaralpha.price_forecasting.multi_output_experiment import NAMES, OUTPUT_DIR, predict_saved
from solaralpha.price_forecasting.multi_output_models import (
    equal_horizon_mae, make_multi_output_model, model_specification,
)
from solaralpha.price_forecasting.sequence_dataset import WINDOW, WindowDataset


class MultiOutputForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_multi_output_data()

    def test_common_counts_shapes_and_chronology(self):
        d = self.data
        self.assertEqual(d.exclusions, {"train": 71, "tune": 0, "evaluate": 0})
        self.assertEqual({part: len(d.indices[part]) for part in d.indices},
                         {"train": 208598, "tune": 49838, "evaluate": 43909})
        for part in d.indices:
            self.assertEqual(d.targets[part].shape, (len(d.indices[part]), 10))
            self.assertTrue(np.all(np.diff(d.indices[part]) > 0))
            self.assertEqual(d.windows(part)[0][0].shape, (72, 41))
            self.assertEqual(d.windows(part)[0][1].shape, (10,))
        self.assertLess(d.indices["train"][-1], d.indices["tune"][0])
        self.assertLess(d.indices["tune"][-1], d.indices["evaluate"][0])

    def test_target_order_and_exact_interval_start(self):
        d = self.data
        for part in ("train", "tune", "evaluate"):
            split = "train" if part == "train" else "validation"
            selected = [0, len(d.indices[part]) // 2, len(d.indices[part]) - 1]
            wanted = {d.origins[part][i]: i for i in selected}
            with gzip.open(TARGET_DIR / f"{split}_targets.csv.gz", "rt", encoding="utf-8", newline="") as f:
                found = {row["decision_start_utc"]: row for row in csv.DictReader(f)
                         if row["decision_start_utc"] in wanted}
            self.assertEqual(set(found), set(wanted))
            for origin, index in wanted.items():
                row = found[origin]
                self.assertEqual(row["common_eligible"], "1")
                for col, h in enumerate(HORIZONS):
                    self.assertAlmostEqual(d.targets[part][index, col], float(row[f"price_h{h}_aud_per_kwh"]))
                    target = datetime.fromisoformat(origin.replace("Z", "+00:00")) + timedelta(minutes=h)
                    self.assertEqual(target.utcoffset(), timedelta(0))
                self.assertEqual(d.sequence.minutes[d.indices[part][index]] -
                                 d.sequence.minutes[d.indices[part][index] - 71], 355)

    def test_partition_and_development_week_isolation(self):
        d = self.data
        zone = sydney_zone()
        for part, origins in d.origins.items():
            for origin in (origins[0], origins[len(origins) // 2], origins[-1]):
                local = datetime.fromisoformat(origin.replace("Z", "+00:00")).astimezone(zone).date()
                end = (datetime.fromisoformat(origin.replace("Z", "+00:00")) +
                       timedelta(minutes=715)).astimezone(zone).date()
                if part == "train":
                    self.assertLess(end.year, 2025)
                elif part == "tune":
                    self.assertLess(end, datetime(2025, 7, 1).date())
                    self.assertFalse(datetime(2025, 1, 13).date() <= local <= datetime(2025, 1, 19).date())
                    self.assertFalse(datetime(2025, 1, 13).date() <= end <= datetime(2025, 1, 19).date())
                else:
                    self.assertGreaterEqual(local, datetime(2025, 7, 1).date())
                    self.assertLessEqual(end, datetime(2025, 11, 30).date())

    def test_future_prices_and_rows_do_not_change_earlier_input(self):
        d = self.data
        end = int(d.indices["evaluate"][100])
        x = d.sequence.x.copy()
        y = d.targets["evaluate"].copy()
        before = WindowDataset(x, d.indices["evaluate"], y)[100][0].clone()
        x[end + 1] += 999
        y[100, 0] = -999  # unknown current settlement label
        y[100, -1] = 999  # distant target label
        after = WindowDataset(x, d.indices["evaluate"], y)[100][0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_equal_horizon_loss(self):
        actual = torch.zeros((2, 10))
        forecast = torch.zeros((2, 10))
        forecast[:, 0] = 10
        self.assertEqual(float(equal_horizon_mae(forecast, actual)), 1.0)
        with self.assertRaises(ValueError):
            equal_horizon_mae(forecast[:, :9], actual)

    def test_output_shapes_negative_head_and_tcn_receptive_field(self):
        expected = {"lstm": 9930, "gru": 10370, "tcn": 10306}
        spec = model_specification()
        x = torch.zeros((3, WINDOW, len(FEATURE_FIELDS)))
        for name in NAMES:
            model = make_multi_output_model(name)
            self.assertEqual(spec[name]["trainable_parameters"], expected[name])
            self.assertEqual(model(x).shape, (3, 10))
            with torch.no_grad():
                model.output.weight.zero_()
                model.output.bias.fill_(-0.25)
            torch.testing.assert_close(model(x), torch.full((3, 10), -0.25))
        self.assertGreaterEqual(make_multi_output_model("tcn").receptive_field, 72)

    def test_saved_models_and_frozen_artifacts(self):
        d = self.data
        design = json.loads((OUTPUT_DIR / "design.json").read_text(encoding="utf-8"))
        self.assertEqual(design["target_manifest_sha256"], sha256((TARGET_DIR / "manifest.json").read_bytes()).hexdigest())
        board = ROOT / "data/models/forecasting_leaderboard.json"
        self.assertEqual(len(json.loads(board.read_text(encoding="utf-8"))["entries"]), 17)
        result = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["source_hashes"]["one_hour_leaderboard"], sha256(board.read_bytes()).hexdigest())
        end = int(d.indices["evaluate"][0])
        raw = (d.sequence.x[end - 71:end + 1] * d.sequence.scaler_scale +
               d.sequence.scaler_mean).astype(np.float32)
        for name in NAMES:
            path = OUTPUT_DIR / name
            predicted = predict_saved(name, path / "model.pt", path / "metadata.json", raw,
                                      FEATURE_FIELDS, [71])
            self.assertEqual(predicted.shape, (1, 10))
            with gzip.open(OUTPUT_DIR / "h060_evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
                row = next(csv.DictReader(f))
            self.assertAlmostEqual(float(predicted[0, HORIZONS.index(60)]),
                                   float(row[f"multi_{name}_aud_per_kwh"]), places=5)
            with self.assertRaises(ValueError):
                predict_saved(name, path / "model.pt", path / "metadata.json", raw,
                              FEATURE_FIELDS[::-1], [71])


if __name__ == "__main__":
    unittest.main()
