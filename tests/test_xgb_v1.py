"""XGBoost V1 schema, split, reproducibility and inference safeguards."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import sydney_zone
from solaralpha.price_forecasting.features import (
    FEATURE_FIELDS, HORIZON_STEPS, OUTPUT_FIELDS, PricePoint, feature_values_at,
)
from solaralpha.price_forecasting.ml_dataset import (
    assert_feature_header, period_for_row, target_hour_of_week,
)
from solaralpha.price_forecasting.xgb_v1 import (
    MODEL_DIR, assert_feature_order, predict_saved,
)


def row_at(local_clock, *, partition="validation", development=False):
    origin = local_clock.astimezone(timezone.utc)
    target = origin + timedelta(hours=1)
    return {
        "decision_start_utc": origin.isoformat().replace("+00:00", "Z"),
        "target_interval_start_utc": target.isoformat().replace("+00:00", "Z"),
        "origin_local_date": local_clock.date().isoformat(),
        "origin_partition": partition, "target_partition": partition,
        "is_development_origin": str(int(development)),
        "is_development_target": str(int(development)),
        "include_in_validation_score": "0" if development else "1",
    }


class XGBV1Tests(unittest.TestCase):
    def test_feature_order_and_csv_header_reject_swaps(self):
        assert_feature_order(FEATURE_FIELDS)
        with self.assertRaisesRegex(ValueError, "order"):
            assert_feature_order(FEATURE_FIELDS[::-1])
        path = Path(__file__).with_name("_xgb_header_test.csv.gz")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        with gzip.open(path, "wt", encoding="utf-8") as file:
            file.write(",".join(OUTPUT_FIELDS) + "\n")
        assert_feature_header(path)
        with gzip.open(path, "wt", encoding="utf-8") as file:
            file.write(",".join(OUTPUT_FIELDS[::-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "ordering"):
            assert_feature_header(path)

    def test_tuning_evaluation_and_development_partition_isolation(self):
        zone = sydney_zone()
        june = row_at(datetime(2025, 6, 30, 22, 55, tzinfo=zone))
        boundary = row_at(datetime(2025, 6, 30, 23, 0, tzinfo=zone))
        july = row_at(datetime(2025, 7, 1, 0, tzinfo=zone))
        self.assertEqual(period_for_row(june), "tune")
        self.assertIsNone(period_for_row(boundary))
        self.assertEqual(period_for_row(july), "evaluate")
        self.assertEqual(target_hour_of_week(july), 1 * 24 + 1)
        dev = row_at(datetime(2025, 1, 13, 10, tzinfo=zone), development=True)
        self.assertIsNone(period_for_row(dev))
        with self.assertRaisesRegex(ValueError, "Development"):
            period_for_row({**dev, "include_in_validation_score": "1"})
        train_crossing = row_at(datetime(2024, 12, 31, 23, tzinfo=zone),
                                partition="train")
        with self.assertRaisesRegex(ValueError, "crosses"):
            period_for_row(train_crossing)
        evaluation_crossing = row_at(datetime(2025, 11, 30, 23, tzinfo=zone))
        with self.assertRaises(ValueError):
            period_for_row(evaluation_crossing)
        held_out = row_at(datetime(2025, 12, 1, 10, tzinfo=zone),
                          partition="held_out_test")
        with self.assertRaisesRegex(ValueError, "held-out"):
            period_for_row(held_out)

    def test_future_realised_changes_cannot_enter_model_input(self):
        zone = sydney_zone()
        start = datetime(2024, 12, 20, tzinfo=timezone.utc)
        points = []
        for i in range(2300):
            instant = start + i * timedelta(minutes=5)
            local = instant.astimezone(zone)
            points.append(PricePoint(instant, local, i / 1000, 7000 + i,
                                     "train", False, local.fold))
        i = 2200
        original = feature_values_at(points, i)
        changed = list(points)
        changed[i + HORIZON_STEPS] = replace(
            changed[i + HORIZON_STEPS], price_aud_per_kwh=20.3,
            regional_demand_mw=10000)
        self.assertEqual(original, feature_values_at(changed, i))
        self.assertEqual(tuple(original), FEATURE_FIELDS)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional XGBoost not installed")
    def test_seeded_fit_and_saved_model_inference(self):
        import numpy as np
        from xgboost import XGBRegressor

        rng = np.random.default_rng(7)
        x = rng.normal(size=(160, len(FEATURE_FIELDS))).astype(np.float32)
        y = x[:, 0] * 0.3 - x[:, 1] * 0.2 + 0.05
        params = {"n_estimators": 24, "max_depth": 3, "learning_rate": 0.05,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "tree_method": "hist", "random_state": 42, "n_jobs": 1,
                  "objective": "reg:squarederror"}
        first, second = XGBRegressor(**params), XGBRegressor(**params)
        first.fit(x[:120], y[:120], verbose=False)
        second.fit(x[:120], y[:120], verbose=False)
        np.testing.assert_array_equal(first.predict(x[120:]), second.predict(x[120:]))
        model_path = Path(__file__).with_name("_xgb_saved_test.json")
        metadata_path = Path(__file__).with_name("_xgb_metadata_test.json")
        self.addCleanup(lambda: model_path.unlink(missing_ok=True))
        self.addCleanup(lambda: metadata_path.unlink(missing_ok=True))
        first.save_model(model_path)
        metadata_path.write_text(json.dumps({
            "feature_order": FEATURE_FIELDS,
            "model_sha256": sha256(model_path.read_bytes()).hexdigest()}),
                                 encoding="utf-8")
        loaded = predict_saved(model_path, metadata_path, x[120:], FEATURE_FIELDS)
        np.testing.assert_array_equal(first.predict(x[120:]), loaded)
        with self.assertRaisesRegex(ValueError, "order"):
            predict_saved(model_path, metadata_path, x[120:], FEATURE_FIELDS[::-1])

    @unittest.skipUnless((MODEL_DIR / "metadata.json").exists(), "Full V1 artifact absent")
    def test_saved_full_model_fingerprints_and_schema(self):
        from hashlib import sha256
        metadata = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(tuple(metadata["feature_order"]), FEATURE_FIELDS)
        self.assertEqual(metadata["counts"]["train"], 208800)
        self.assertEqual(metadata["counts"]["tune_evaluate_boundary"], 12)
        self.assertEqual(metadata["model_sha256"],
                         sha256((MODEL_DIR / "model.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
