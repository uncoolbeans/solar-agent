"""Chronology, information boundary, causality and saved PyTorch inference."""

from hashlib import sha256
import json
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import ROOT
from solaralpha.price_forecasting.features import FEATURE_FIELDS
from solaralpha.price_forecasting.sequence_dataset import WINDOW, WindowDataset, load_sequence_data
from solaralpha.price_forecasting.sequence_models import make_model, parameter_count
from solaralpha.price_forecasting.sequence_experiment import (
    NEURAL_NAMES, OUTPUT_DIR, predict_saved,
)


class SequenceForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_sequence_data()

    def test_shape_order_and_context_across_train_tune_boundary(self):
        import numpy as np
        d = self.data
        self.assertEqual(d.x.shape, (304992, 41))
        self.assertEqual(d.exclusions, {"train": 71, "tune": 0, "evaluate": 0})
        self.assertEqual({part: len(indices) for part, indices in d.indices.items()},
                         {"train": 208729, "tune": 50100, "evaluate": 44040})
        self.assertTrue(np.all(np.diff(d.minutes) == 5))
        first_tune = int(d.indices["tune"][0])
        self.assertLess(first_tune - 71, 208800)  # December train context is used.
        self.assertEqual(d.windows("tune")[0][0].shape, (WINDOW, 41))
        self.assertEqual(d.windows("evaluate")[-1][0].shape, (WINDOW, 41))

    def test_target_alignment_and_partition_isolation(self):
        from datetime import datetime, timedelta
        d = self.data
        self.assertLess(max(d.indices["train"]), min(d.indices["tune"]))
        self.assertLess(max(d.indices["tune"]), min(d.indices["evaluate"]))
        for part in ("tune", "evaluate"):
            for i in (0, len(d.indices[part]) // 2, len(d.indices[part]) - 1):
                end = int(d.indices[part][i])
                origin = datetime.fromisoformat(d.rows[part][i]["decision_start_utc"].replace("Z", "+00:00"))
                target = datetime.fromisoformat(d.rows[part][i]["target_interval_start_utc"].replace("Z", "+00:00"))
                self.assertEqual(target - origin, timedelta(minutes=60))
                self.assertEqual(int(origin.timestamp() // 60), d.minutes[end])
                self.assertEqual(d.minutes[end] - d.minutes[end - 71], 355)

    def test_training_only_scaler(self):
        import numpy as np
        from solaralpha.price_forecasting.ml_dataset import load_slices
        original = load_slices()
        np.testing.assert_allclose(self.data.scaler_mean,
                                   original.train.x.mean(axis=0, dtype=np.float64), atol=1e-10)
        np.testing.assert_allclose(self.data.scaler_scale,
                                   original.train.x.std(axis=0, dtype=np.float64), atol=1e-10)
        # Validation distribution differs; fitting it would change this value.
        self.assertGreater(float(np.max(np.abs(self.data.scaler_mean -
                                               original.evaluate.x.mean(axis=0)))), 0.01)

    def test_future_rows_and_targets_cannot_change_earlier_input(self):
        import numpy as np
        d = self.data
        end = int(d.indices["evaluate"][100])
        x = d.x.copy()
        labels = d.targets["evaluate"].copy()
        before = WindowDataset(x, d.indices["evaluate"], labels)[100][0].clone()
        x[end + 1] += 1000
        labels[101] += 1000
        after = WindowDataset(x, d.indices["evaluate"], labels)[100][0]
        np.testing.assert_array_equal(before.numpy(), after.numpy())

    def test_tcn_causality_and_receptive_field(self):
        import torch
        torch.manual_seed(3)
        model = make_model("tcn").eval()
        self.assertGreaterEqual(model.receptive_field, WINDOW)
        x = torch.randn(1, WINDOW, len(FEATURE_FIELDS))
        changed = x.clone()
        changed[:, 50:, :] += 100
        with torch.inference_mode():
            before = model.forward_features(x)
            after = model.forward_features(changed)
        torch.testing.assert_close(before[:, :, :50], after[:, :, :50], rtol=0, atol=0)

    def test_all_output_shapes_and_unrestricted_negative_head(self):
        import torch
        x = torch.zeros((3, WINDOW, len(FEATURE_FIELDS)))
        expected_parameters = {"lstm": 9633, "gru": 10001, "tcn": 10081}
        for name in NEURAL_NAMES:
            model = make_model(name)
            self.assertEqual(parameter_count(model), expected_parameters[name])
            self.assertEqual(model(x).shape, (3,))
            with torch.no_grad():
                model.output.weight.zero_()
                model.output.bias.fill_(-0.25)
            torch.testing.assert_close(model(x), torch.full((3,), -0.25))

    def test_saved_inference_hash_and_feature_order(self):
        import numpy as np
        import torch
        torch.manual_seed(42)
        raw = np.zeros((WINDOW + 1, len(FEATURE_FIELDS)), dtype=np.float32)
        checkpoint = ROOT / "tests" / "_sequence_test_model.pt"
        metadata = ROOT / "tests" / "_sequence_test_metadata.json"
        self.addCleanup(lambda: checkpoint.unlink(missing_ok=True))
        self.addCleanup(lambda: metadata.unlink(missing_ok=True))
        model = make_model("gru").eval()
        torch.save(model.state_dict(), checkpoint)
        metadata.write_text(json.dumps({"name": "gru", "feature_order": FEATURE_FIELDS,
                                        "scaler_mean": [0.0] * len(FEATURE_FIELDS),
                                        "scaler_scale": [1.0] * len(FEATURE_FIELDS),
                                        "model_sha256": sha256(checkpoint.read_bytes()).hexdigest()}),
                            encoding="utf-8")
        with torch.inference_mode():
            expected = model(torch.from_numpy(raw[:WINDOW][None, :, :])).numpy()
        np.testing.assert_allclose(predict_saved("gru", checkpoint, metadata, raw,
                                                FEATURE_FIELDS, [WINDOW - 1]), expected,
                                   rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            predict_saved("gru", checkpoint, metadata, raw, FEATURE_FIELDS[::-1],
                          [WINDOW - 1])

    @unittest.skipUnless((OUTPUT_DIR / "results.json").exists(), "Neural results absent")
    def test_saved_results_match_frozen_14_model_rows(self):
        import csv
        import gzip
        from solaralpha.price_forecasting.tabular_comparison import (
            ALL_NAMES as OLD_NAMES, OUTPUT_DIR as OLD_DIR, LEADERBOARD_PATH)
        report = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
        board = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(board["entries"]), 17)
        self.assertEqual([entry["name"] for entry in board["entries"][-3:]],
                         list(NEURAL_NAMES))
        for part, count in (("tune", 50100), ("evaluate", 44040)):
            with gzip.open(OLD_DIR / f"{part}_comparison.csv.gz", "rt", encoding="utf-8") as stream:
                old = list(csv.DictReader(stream))
            with gzip.open(OUTPUT_DIR / f"{part}_comparison.csv.gz", "rt", encoding="utf-8") as stream:
                new = list(csv.DictReader(stream))
            self.assertEqual(len(new), count)
            for before, after in zip(old, new):
                for field in ("decision_start_utc", "target_interval_start_utc",
                              "actual_aud_per_kwh", *(f"{n}_aud_per_kwh" for n in OLD_NAMES)):
                    self.assertEqual(before[field], after[field])
            self.assertEqual(report["splits"][part]["row_count"], count)


if __name__ == "__main__":
    unittest.main()
