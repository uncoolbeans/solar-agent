"""Forecast timing, leakage, split, and baseline checks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import sydney_zone
from solaralpha.price_forecasting.baselines import (
    SPIKE_THRESHOLD_AUD_PER_KWH, _event_scores, predict_row,
    training_target_median,
)
from solaralpha.price_forecasting.features import (
    FEATURE_FIELDS, HORIZON_STEPS, PricePoint, TARGET_FIELD,
    build_feature_dataset, feature_values_at, read_feature_rows, trailing_stats,
)


class PriceForecastingTests(unittest.TestCase):
    def points(self, count=3100):
        zone = sydney_zone()
        start = datetime(2024, 12, 23, 13, tzinfo=timezone.utc)  # Dec 24 local
        result = []
        for i in range(count):
            utc = start + i * timedelta(minutes=5)
            local = utc.astimezone(zone)
            partition = "train" if local.year == 2024 else "validation"
            result.append(PricePoint(utc, local, 0.05 + (i % 37) / 1000,
                                     7000 + i % 200, partition,
                                     local.date().isoformat() == "2025-01-01",
                                     local.fold))
        return result

    def test_trailing_window_ends_before_decision(self):
        stats = trailing_stats([1.0, 2.0, 3.0, 10.0], 3)
        self.assertIsNone(stats[2])
        self.assertEqual(stats[3][0], 2)
        self.assertEqual(stats[3][2:], (1, 3))
        self.assertEqual(stats[4][0], 5)
        self.assertEqual(stats[4][2:], (2, 10))

    def test_future_or_current_realised_price_and_demand_cannot_change_features(self):
        points = self.points()
        i = 2500
        original = feature_values_at(points, i)
        for changed_index in (i, i + 1, i + HORIZON_STEPS, i + 200):
            changed = list(points)
            changed[changed_index] = replace(
                changed[changed_index], price_aud_per_kwh=20.0,
                regional_demand_mw=10000.0)
            self.assertEqual(original, feature_values_at(changed, i))
        changed = list(points)
        changed[i - 1] = replace(changed[i - 1], price_aud_per_kwh=-1.0)
        self.assertNotEqual(original, feature_values_at(changed, i))
        self.assertEqual(set(original), set(FEATURE_FIELDS))

    def test_target_exactly_one_hour_and_no_train_validation_crossing(self):
        points = self.points()
        output = Path(__file__).parent / "_price_features_test"
        output.mkdir(exist_ok=True)
        self.addCleanup(output.rmdir)
        for name in ("train_one_hour_price.csv.gz", "validation_one_hour_price.csv.gz",
                     "build_report.json"):
            self.addCleanup(lambda p=output / name: p.unlink(missing_ok=True))
        report = build_feature_dataset(points, output)
        self.assertEqual(report["counts"]["cross_partition_targets_excluded"], 12)
        train = list(read_feature_rows(output / "train_one_hour_price.csv.gz"))
        validation = list(read_feature_rows(output / "validation_one_hour_price.csv.gz"))
        self.assertTrue(train)
        self.assertTrue(validation)
        for row in train + validation:
            origin = datetime.fromisoformat(row["decision_start_utc"].replace("Z", "+00:00"))
            target = datetime.fromisoformat(row["target_interval_start_utc"].replace("Z", "+00:00"))
            self.assertEqual(target - origin, timedelta(minutes=60))
            self.assertEqual(row["origin_partition"], row["target_partition"])
        self.assertTrue(any(r["include_in_validation_score"] == "1" for r in validation))
        self.assertTrue(any(r["include_in_validation_score"] == "0" for r in validation))
        self.assertTrue(all(r["is_development_origin"] == "0"
                            and r["is_development_target"] == "0"
                            for r in validation if r["include_in_validation_score"] == "1"))
        first = validation[0]
        target = datetime.fromisoformat(first["target_interval_start_utc"].replace("Z", "+00:00"))
        matching = next(p for p in points if p.time_utc == target)
        self.assertEqual(float(first[TARGET_FIELD]), matching.price_aud_per_kwh)

    def test_training_median_and_baseline_features(self):
        rows = [{"origin_partition": "train", "target_partition": "train",
                 TARGET_FIELD: value} for value in ("0.1", "0.3", "0.2")]
        self.assertEqual(training_target_median(rows), (0.2, 3))
        with self.assertRaises(ValueError):
            training_target_median(rows + [{**rows[0], "origin_partition": "validation"}])
        forecast = predict_row({"price_lag_5m_aud_per_kwh": "-0.1",
                                "price_target_aligned_previous_day_aud_per_kwh": "0.4"}, 0.2)
        self.assertEqual(tuple(forecast.values()), (-0.1, 0.4, 0.2))

    def test_price_event_thresholds_preserve_negative_and_spikes(self):
        self.assertEqual(SPIKE_THRESHOLD_AUD_PER_KWH, 1.0)
        result = _event_scores([-1.0, 0.2, 2.0], [-0.2, -0.1, 3.0],
                               lambda v: v < 0)
        self.assertEqual((result["actual_count"], result["true_positive"],
                          result["false_positive"]), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
