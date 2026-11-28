"""Dense target timing, leakage, saved models and forecast contract checks."""

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import unittest

import numpy as np
import torch

from solaralpha.historical_pipeline import ROOT, sydney_zone
from solaralpha.price_forecasting.features import FEATURE_FIELDS
from solaralpha.price_forecasting.full_horizon_dataset import HORIZONS, TARGET_DIR, load_targets
from solaralpha.price_forecasting.full_horizon_direct import OUTPUT_DIR, model_paths, verified_model
from solaralpha.price_forecasting.full_horizon_interface import (
    ForecastRequest, FullHorizonForecaster, assemble_response, validate_request,
)
from solaralpha.price_forecasting.full_horizon_tcn import (
    load_saved as load_tcn, make_model_144, predict_saved as predict_tcn,
)
from solaralpha.price_forecasting.multi_horizon import HORIZONS as SPARSE_HORIZONS
from solaralpha.price_forecasting.multi_output_dataset import load_multi_output_data
from solaralpha.price_forecasting.features import load_training_validation_points
from solaralpha.price_forecasting.sequence_dataset import WindowDataset


class FullHorizonForecastingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_targets()

    def test_target_order_shape_common_origins_and_sparse_alignment(self):
        d = self.data
        self.assertEqual(HORIZONS, tuple(range(0, 720, 5)))
        self.assertEqual({p: int(d.common[p].sum()) for p in d.common},
                         {"train": 208669, "tune": 49838, "evaluate": 43909})
        for part in ("train", "tune", "evaluate"):
            self.assertEqual(d.targets[part].shape[1], 144)
            self.assertTrue(np.isfinite(d.targets[part][d.common[part]]).all())
            for h in SPARSE_HORIZONS:
                np.testing.assert_array_equal(d.targets[part][:, HORIZONS.index(h)],
                                              d.matrices[part].y[:, SPARSE_HORIZONS.index(h)])

    def test_exact_target_time_partition_and_development_exclusion(self):
        d = self.data
        points = load_training_validation_points()
        first = points[0].time_utc
        for part in ("train", "tune", "evaluate"):
            positions = np.flatnonzero(d.common[part])
            for i in (positions[0], positions[len(positions)//2], positions[-1]):
                start = datetime.fromisoformat(d.matrices[part].rows[i]["decision_start_utc"].replace("Z", "+00:00"))
                end = start + timedelta(minutes=715)
                self.assertEqual((end - start).total_seconds(), 715 * 60)
                point_index = int((start - first) / timedelta(minutes=5))
                for h in (0, 10, 355, 710, 715):
                    self.assertEqual(points[point_index + h//5].time_utc, start + timedelta(minutes=h))
                    self.assertEqual(d.targets[part][i, HORIZONS.index(h)],
                                     points[point_index + h//5].price_aud_per_kwh)
                local0, local1 = start.astimezone(sydney_zone()).date(), end.astimezone(sydney_zone()).date()
                if part == "train":
                    self.assertLess(local1.year, 2025)
                elif part == "tune":
                    self.assertLess(local1, datetime(2025, 7, 1).date())
                    self.assertFalse(datetime(2025, 1, 13).date() <= local0 <= datetime(2025, 1, 19).date())
                    self.assertFalse(datetime(2025, 1, 13).date() <= local1 <= datetime(2025, 1, 19).date())
                else:
                    self.assertGreaterEqual(local0, datetime(2025, 7, 1).date())
                    self.assertLessEqual(local1, datetime(2025, 11, 30).date())

    def test_future_realised_prices_cannot_change_prior_window(self):
        sparse = load_multi_output_data()
        end = int(sparse.indices["evaluate"][100])
        matrix = sparse.sequence.x.copy()
        labels = np.asarray(self.data.targets["evaluate"][self.data.common["evaluate"]], dtype=np.float32)
        before = WindowDataset(matrix, sparse.indices["evaluate"], labels)[100][0].clone()
        matrix[end+1] += 1000
        labels[100, 0] = -1000
        labels[100, -1] = 1000
        after = WindowDataset(matrix, sparse.indices["evaluate"], labels)[100][0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_partition_boundary_and_development_week_targets_are_absent(self):
        d = self.data
        def find(part, year, month, day, hour):
            for i, row in enumerate(d.matrices[part].rows):
                local = datetime.fromisoformat(row["decision_start_utc"].replace("Z", "+00:00")).astimezone(sydney_zone())
                if (local.year, local.month, local.day, local.hour, local.minute) == (year, month, day, hour, 0):
                    return i
            self.fail("Expected boundary origin missing")
        self.assertTrue(np.isnan(d.targets["train"][find("train", 2024, 12, 31, 20), -1]))
        self.assertTrue(np.isnan(d.targets["tune"][find("tune", 2025, 6, 30, 20), -1]))
        self.assertTrue(np.isnan(d.targets["evaluate"][find("evaluate", 2025, 11, 30, 20), -1]))
        self.assertTrue(np.isnan(d.targets["tune"][find("tune", 2025, 1, 15, 12)]).all())

    def test_tcn_shape_negative_prices_and_receptive_field(self):
        model = make_model_144()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 13656)
        self.assertGreaterEqual(model.receptive_field, 72)
        x = torch.zeros((2, 72, 41))
        self.assertEqual(model(x).shape, (2, 144))
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.fill_(-.25)
        torch.testing.assert_close(model(x), torch.full((2, 144), -.25))
        self.assertEqual(float(torch.abs(torch.cat((torch.full((2, 1), 144.), torch.zeros((2, 143))), dim=1)).mean()), 1.)

    def test_saved_model_inference_and_frozen_prior_models(self):
        data = self.data
        result = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
        self.assertEqual(result["counts"]["evaluate"], 43909)
        for h in SPARSE_HORIZONS:
            self.assertEqual(model_paths(h)[0].parent, ROOT / f"data/models/price_multi_horizon_v1/h{h:03d}")
        model, meta = verified_model(10, data)
        row = data.matrices["evaluate"].x[data.common["evaluate"]][0:1]
        with np.load(OUTPUT_DIR / "evaluate_comparison.npz") as comparison:
            self.assertAlmostEqual(float(model.predict(row)[0]), float(comparison["direct"][0, 2]), places=5)
            expected = comparison["tcn"][0]
        sparse = load_multi_output_data()
        end = int(sparse.indices["evaluate"][0])
        raw = (sparse.sequence.x[end-71:end+1] * sparse.sequence.scaler_scale +
               sparse.sequence.scaler_mean).astype(np.float32)
        prediction = predict_tcn(raw, FEATURE_FIELDS, [71])
        self.assertEqual(prediction.shape, (1, 144))
        np.testing.assert_allclose(prediction[0], expected, atol=2e-5, rtol=2e-5)
        times = tuple(datetime.fromtimestamp(int(value) * 60, timezone.utc)
                      for value in sparse.sequence.minutes[end-71:end+1])
        response = FullHorizonForecaster("tcn").forecast(
            ForecastRequest(times[-1], times, raw, FEATURE_FIELDS, times[-1]))
        np.testing.assert_allclose(response.predicted_wholesale_aud_per_kwh,
                                   expected, atol=2e-5, rtol=2e-5)

    def test_forecast_interface_timestamps_freshness_and_rejection(self):
        t = datetime(2025, 8, 1, 2, 0, tzinfo=timezone.utc)
        history = tuple(t - timedelta(minutes=5 * (71-i)) for i in range(72))
        request = ForecastRequest(t, history, np.zeros((72, 41), dtype=np.float32), FEATURE_FIELDS, t)
        response = assemble_response(request, "test", np.full(144, -.1))
        self.assertEqual(response.target_interval_starts_utc[0], t)
        self.assertEqual(response.target_interval_starts_utc[-1], t + timedelta(minutes=715))
        self.assertEqual(response.unit, "AUD/kWh")
        self.assertTrue(response.input_valid)
        stale = ForecastRequest(t, history, request.feature_history, FEATURE_FIELDS,
                                t - timedelta(minutes=10))
        self.assertFalse(assemble_response(stale, "test", np.zeros(144)).input_valid)
        with self.assertRaises(ValueError):
            assemble_response(request, "test", np.zeros(143))
        with self.assertRaises(ValueError):
            assemble_response(request, "test", np.full(144, np.nan))
        with self.assertRaises(ValueError):
            validate_request(ForecastRequest(t, history[::-1], request.feature_history,
                                             FEATURE_FIELDS, t))
        with self.assertRaises(ValueError):
            validate_request(ForecastRequest(t, history, request.feature_history,
                                             FEATURE_FIELDS[::-1], t))
        with self.assertRaises(ValueError):
            validate_request(ForecastRequest(t, history, request.feature_history,
                                             FEATURE_FIELDS, t + timedelta(minutes=5)))

    def test_selection_frozen_and_separate_leaderboards(self):
        selection_path = OUTPUT_DIR / "selection.json"
        result = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        self.assertEqual(result["selection_file_sha256"], sha256(selection_path.read_bytes()).hexdigest())
        self.assertTrue(selection["frozen_before_reused_evaluation"])
        self.assertEqual(len(selection["choices"]), 144)
        self.assertEqual(selection["fixed_choices"][:3], ["persistence", "persistence", "direct"])
        board = json.loads((ROOT / "data/models/forecasting_leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual(len(board["entries"]), 17)
        dense_board = json.loads((ROOT / "data/models/full_horizon_leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual(len(dense_board["entries"]), 8)
        self.assertEqual(dense_board["common_evaluation_origins"], 43909)


if __name__ == "__main__":
    unittest.main()
