"""Target availability and frozen-artifact guards for the offline experiment."""

import csv
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import ROOT, sydney_zone
from solaralpha.price_forecasting.features import (FEATURE_DIR, FEATURE_FIELDS,
                                                    PricePoint, feature_values_at)
from solaralpha.price_forecasting.multi_horizon import (
    HORIZONS, MODEL_DIR, TARGET_DIR, eligible_period, predict_saved,
    target_for_horizon,
)


def points(count=2300):
    start = datetime(2025, 1, 20, tzinfo=sydney_zone()).astimezone(timezone.utc)
    return [PricePoint(start + i * timedelta(minutes=5),
                       (start + i * timedelta(minutes=5)).astimezone(sydney_zone()),
                       i / 1000, 8000 + i, "validation", False)
            for i in range(count)]


class MultiHorizonTests(unittest.TestCase):
    def test_exact_target_and_seasonal_alignment(self):
        p = points()
        index = 2050
        for h in HORIZONS:
            actual, seasonal = target_for_horizon(p, index, h)
            self.assertEqual(actual, p[index + h // 5].price_aud_per_kwh)
            self.assertEqual(seasonal, p[index + h // 5 - 288].price_aud_per_kwh)
            self.assertLess(p[index + h // 5 - 288].time_utc, p[index].time_utc)

    def test_zero_minute_settlement_is_never_an_input(self):
        p = points()
        index = 2050
        before = feature_values_at(p, index)
        old_target = target_for_horizon(p, index, 0)[0]
        p[index] = PricePoint(p[index].time_utc, p[index].local_time, -0.9,
                              99999, p[index].partition, False)
        self.assertEqual(before, feature_values_at(p, index))
        self.assertNotEqual(old_target, target_for_horizon(p, index, 0)[0])

    def test_partition_and_development_boundaries(self):
        p = points(400)
        i = 300
        self.assertEqual(eligible_period(p[i], p[i + 1]), "tune")
        p[i + 1] = PricePoint(p[i + 1].time_utc, p[i + 1].local_time, 1, 1,
                              "train", False)
        self.assertIsNone(target_for_horizon(p, i, 5))
        p[i + 1] = PricePoint(p[i + 1].time_utc, p[i + 1].local_time, 1, 1,
                              "validation", True)
        self.assertIsNone(target_for_horizon(p, i, 5))
        p[i] = PricePoint(p[i].time_utc, p[i].local_time, 1, 1,
                          "validation", True)
        self.assertIsNone(target_for_horizon(p, i, 0))

    def test_tuning_evaluation_boundary(self):
        zone = sydney_zone()
        a = datetime(2025, 6, 30, 23, 55, tzinfo=zone)
        b = a + timedelta(minutes=5)
        pa = PricePoint(a.astimezone(timezone.utc), a, 0, 0, "validation", False)
        pb = PricePoint(b.astimezone(timezone.utc), b, 0, 0, "validation", False)
        self.assertIsNone(eligible_period(pa, pb))

    def test_common_origin_requires_every_target(self):
        with gzip.open(TARGET_DIR / "validation_targets.csv.gz", "rt", newline="", encoding="utf-8") as f:
            rows = csv.DictReader(f)
            for row in rows:
                complete = all(row[f"price_h{h}_aud_per_kwh"] for h in HORIZONS)
                self.assertEqual(row["common_eligible"] == "1", complete)

    def test_original_one_hour_files_and_leaderboard_preserved(self):
        manifest = json.loads((TARGET_DIR / "manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            path = FEATURE_DIR / f"{split}_one_hour_price.csv.gz"
            self.assertEqual(sha256(path.read_bytes()).hexdigest(),
                             manifest["feature_files"][split]["sha256"])
        leaderboard = json.loads((ROOT / "data/models/forecasting_leaderboard.json").read_text(encoding="utf-8"))
        self.assertEqual(len(leaderboard["entries"]), 17)

    def test_saved_model_inference_and_horizon_guard(self):
        import numpy as np
        from solaralpha.price_forecasting.features import read_feature_rows
        folder = MODEL_DIR / "h060"
        with gzip.open(folder / "evaluate_predictions.csv.gz", "rt", newline="", encoding="utf-8") as f:
            saved = next(csv.DictReader(f))
        row = next(row for row in read_feature_rows(FEATURE_DIR / "validation_one_hour_price.csv.gz")
                   if row["decision_start_utc"] == saved["decision_start_utc"])
        x = np.asarray([[float(row[field]) for field in FEATURE_FIELDS]], dtype=np.float32)
        predictions = predict_saved(folder / "model.json", folder / "metadata.json",
                                    x, FEATURE_FIELDS, 60)
        self.assertEqual(predictions.shape, (1,))
        self.assertAlmostEqual(float(predictions[0]),
                               float(saved["xgboost_aud_per_kwh"]), places=6)
        old_meta = json.loads((ROOT / "data/models/price_xgb_v2/V2-A/metadata.json").read_text(encoding="utf-8"))
        new_meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(old_meta["model_sha256"], new_meta["model_sha256"])
        with self.assertRaises(ValueError):
            predict_saved(folder / "model.json", folder / "metadata.json", x,
                          FEATURE_FIELDS, 120)


if __name__ == "__main__":
    unittest.main()
