"""Shared-horizon alignment, chronology, leakage and artifact safeguards."""

import csv
from datetime import datetime, timedelta
import gzip
from hashlib import sha256
import json
import unittest

import numpy as np

from solaralpha.historical_pipeline import ROOT, sydney_zone
from solaralpha.price_forecasting.features import FEATURE_DIR, FEATURE_FIELDS, read_feature_rows
from solaralpha.price_forecasting.horizon_conditioned import (
    FEATURE_ORDER, OUTPUT_DIR, append_horizon, mean_horizon_mae,
    memory_estimate, predict_saved, stack_common,
)
from solaralpha.price_forecasting.multi_horizon import HORIZONS, Matrix, TARGET_DIR


def tiny_matrix():
    x = np.arange(4 * len(FEATURE_FIELDS), dtype=np.float32).reshape(4, -1)
    y = np.arange(4 * len(HORIZONS), dtype=np.float64).reshape(4, -1)
    y[1, :] = np.nan  # excluded boundary/development origin
    rows = [{"decision_start_utc": f"2025-01-20T00:{i * 5:02d}:00Z"} for i in range(4)]
    return Matrix(x, y, np.zeros_like(y), np.asarray([True, False, True, True]), rows)


class HorizonConditionedTests(unittest.TestCase):
    def test_stacked_horizon_and_target_alignment(self):
        matrix = tiny_matrix()
        x, y = stack_common(matrix)
        self.assertEqual(x.shape, (30, 42))
        self.assertEqual(y.shape, (30,))
        for j, h in enumerate(HORIZONS):
            np.testing.assert_array_equal(x[j * 3:(j + 1) * 3, :-1], matrix.x[[0, 2, 3]])
            np.testing.assert_array_equal(x[j * 3:(j + 1) * 3, -1], h)
            np.testing.assert_array_equal(y[j * 3:(j + 1) * 3], matrix.y[[0, 2, 3], j])

    def test_future_labels_cannot_change_inputs(self):
        matrix = tiny_matrix()
        x, y = stack_common(matrix)
        matrix.y[0, 0] = -999.0  # current, unknown settlement changes
        matrix.y[2, 9] = 999.0  # distant future changes
        changed_x, changed_y = stack_common(matrix)
        np.testing.assert_array_equal(x, changed_x)
        self.assertFalse(np.array_equal(y, changed_y))

    def test_common_origin_exclusions_and_memory_estimate(self):
        matrix = tiny_matrix()
        self.assertEqual(len(stack_common(matrix)[1]), 3 * len(HORIZONS))
        matrix.common[1] = True
        with self.assertRaises(ValueError):
            stack_common(matrix)  # incomplete target cannot enter shared training
        estimate = memory_estimate({"train": 208669})["train"]
        self.assertEqual(estimate["examples"], 2086690)
        self.assertAlmostEqual(estimate["matrix_mib"], 334.3238067626953)

    def test_equal_horizon_early_stop_metric(self):
        actual = np.zeros(10 * 3)
        predicted = np.zeros_like(actual)
        predicted[:3] = 10
        self.assertEqual(mean_horizon_mae(predicted, actual), 1.0)
        with self.assertRaises(ValueError):
            mean_horizon_mae(predicted[:-1], actual[:-1])

    def test_real_partition_and_development_boundaries(self):
        with gzip.open(TARGET_DIR / "validation_targets.csv.gz", "rt", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        zone = sydney_zone()
        june_end = [row for row in rows if datetime.fromisoformat(row["decision_start_utc"].replace("Z", "+00:00")).astimezone(zone).date().isoformat() == "2025-06-30"]
        self.assertTrue(any(row["price_h715_aud_per_kwh"] == "" and row["common_eligible"] == "0"
                            for row in june_end))
        development = [row for row in rows if datetime.fromisoformat(row["decision_start_utc"].replace("Z", "+00:00")).astimezone(zone).date().isoformat() == "2025-01-15"]
        self.assertTrue(development)
        self.assertTrue(all(row["common_eligible"] == "0" and not row["price_h0_aud_per_kwh"]
                            for row in development))

    def test_saved_model_inference_and_trained_horizon_guard(self):
        with gzip.open(OUTPUT_DIR / "h715_evaluate_predictions.csv.gz", "rt", encoding="utf-8", newline="") as f:
            saved = next(csv.DictReader(f))
        row = next(row for row in read_feature_rows(FEATURE_DIR / "validation_one_hour_price.csv.gz")
                   if row["decision_start_utc"] == saved["decision_start_utc"])
        matrix = np.asarray([[float(row[field]) for field in FEATURE_FIELDS]], dtype=np.float32)
        predicted = predict_saved(matrix, FEATURE_FIELDS, 715)
        self.assertAlmostEqual(float(predicted[0]), float(saved["shared_aud_per_kwh"]), places=6)
        with self.assertRaises(ValueError):
            predict_saved(matrix, tuple(reversed(FEATURE_FIELDS)), 715)
        with self.assertRaises(ValueError):
            append_horizon(matrix, 25)  # untrained intermediate offset
        with self.assertRaises(ValueError):
            append_horizon(matrix[:, :-1], 715)

    def test_frozen_artifacts_and_tuning_only_hybrid(self):
        design = json.loads((OUTPUT_DIR / "design.json").read_text(encoding="utf-8"))
        selection = json.loads((OUTPUT_DIR / "hybrid_selection.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(design["feature_order"]), FEATURE_ORDER)
        self.assertEqual(tuple(design["horizons_minutes"]), HORIZONS)
        self.assertEqual(design["source_fingerprints"]["direct_results_sha256"],
                         sha256((ROOT / "data/models/price_multi_horizon_v1/results.json").read_bytes()).hexdigest())
        self.assertEqual(design["source_fingerprints"]["one_hour_leaderboard_sha256"],
                         sha256((ROOT / "data/models/forecasting_leaderboard.json").read_bytes()).hexdigest())
        self.assertTrue(selection["frozen_before_reused_evaluation"])
        for entry in selection["horizons"].values():
            self.assertEqual(entry["chosen"], min(("persistence", "direct", "shared"),
                             key=lambda name: entry["common_tune_mae"][name]))


if __name__ == "__main__":
    unittest.main()
