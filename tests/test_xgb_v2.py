"""Residual algebra, frozen features, objective behavior and V2 model loading."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import sydney_zone
from solaralpha.price_forecasting.features import (
    FEATURE_FIELDS, HORIZON_STEPS, PricePoint, feature_values_at,
)
from solaralpha.price_forecasting.ml_dataset import Slice
from solaralpha.price_forecasting.xgb_v2 import (
    LATEST_INDEX, V2_DIR, VARIANTS, labels_for_variant,
    prediction_to_price, predict_saved_variant, reconstruct_price,
    residual_target,
)
from solaralpha.price_forecasting.xgb_v1 import MODEL_DIR as V1_DIR


class XGBV2Tests(unittest.TestCase):
    def test_residual_target_reconstruction_and_mae_identity(self):
        import numpy as np
        actual = np.array([-1.0, -0.01, 0.1, 20.3])
        latest = np.array([0.1, -0.2, 0.05, 0.08])
        predicted_residual = np.array([-0.3, 0.1, 0.0, 2.0])
        target = residual_target(actual, latest)
        np.testing.assert_allclose(target, [-1.1, 0.19, 0.05, 20.22])
        predicted_price = reconstruct_price(predicted_residual, latest)
        np.testing.assert_allclose(actual - predicted_price,
                                   target - predicted_residual)
        self.assertAlmostEqual(float(np.mean(np.abs(actual - predicted_price))),
                               float(np.mean(np.abs(target - predicted_residual))))

    def test_residual_labels_use_latest_feature_and_future_only_for_label(self):
        import numpy as np
        x = np.zeros((2, len(FEATURE_FIELDS)), dtype=np.float32)
        x[:, LATEST_INDEX] = [-0.2, 0.3]
        portion = Slice(x, np.array([0.1, -0.1]), [])
        before = x.copy()
        np.testing.assert_allclose(labels_for_variant(portion, residual=True),
                                   [0.3, -0.4], atol=1e-8)
        np.testing.assert_allclose(prediction_to_price(
            np.array([0.3, -0.4]), portion, residual=True), portion.y, atol=1e-8)
        np.testing.assert_array_equal(x, before)
        altered = Slice(x, np.array([20.3, -1.0]), [])
        np.testing.assert_array_equal(altered.x, portion.x)
        self.assertNotEqual(labels_for_variant(altered, residual=True)[0],
                            labels_for_variant(portion, residual=True)[0])

    def test_future_realised_observations_leave_v2_features_unchanged(self):
        zone = sydney_zone()
        start = datetime(2024, 12, 20, tzinfo=timezone.utc)
        points = []
        for i in range(2300):
            instant = start + i * timedelta(minutes=5)
            local = instant.astimezone(zone)
            points.append(PricePoint(instant, local, 0.1, 7000 + i,
                                     "train", False, local.fold))
        i = 2200
        prior = feature_values_at(points, i)
        changed = list(points)
        changed[i + HORIZON_STEPS] = replace(changed[i + HORIZON_STEPS],
                                             price_aud_per_kwh=20.3,
                                             regional_demand_mw=9999)
        self.assertEqual(prior, feature_values_at(changed, i))

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional XGBoost absent")
    def test_installed_objectives_and_residual_early_stop_metric(self):
        import numpy as np
        from xgboost import XGBRegressor

        rng = np.random.default_rng(42)
        x = rng.normal(size=(120, len(FEATURE_FIELDS))).astype(np.float32)
        latest = x[:, LATEST_INDEX].astype(float)
        actual = latest + 0.1 * x[:, 0] - 0.03
        target = residual_target(actual, latest)
        for objective in ("reg:absoluteerror", "reg:pseudohubererror",
                          "reg:squarederror"):
            model = XGBRegressor(n_estimators=20, max_depth=2, learning_rate=0.1,
                                 objective=objective, eval_metric="mae",
                                 early_stopping_rounds=3, tree_method="hist",
                                 random_state=42, n_jobs=1)
            model.fit(x[:80], target[:80],
                      eval_set=[(x[:80], target[:80]), (x[80:], target[80:])],
                      verbose=False)
            raw = model.predict(x[80:])
            price_mae = float(np.mean(np.abs(actual[80:] -
                                             reconstruct_price(raw, latest[80:]))))
            self.assertAlmostEqual(price_mae, float(model.best_score), places=5)

    @unittest.skipUnless(importlib.util.find_spec("xgboost"), "Optional XGBoost absent")
    def test_saved_residual_model_inference_and_order_guard(self):
        import numpy as np
        from xgboost import XGBRegressor

        rng = np.random.default_rng(11)
        x = rng.normal(size=(80, len(FEATURE_FIELDS))).astype(np.float32)
        y = 0.2 * x[:, 0]
        model = XGBRegressor(n_estimators=10, max_depth=2,
                             objective="reg:absoluteerror", tree_method="hist",
                             random_state=42, n_jobs=1)
        model.fit(x, y, verbose=False)
        path = Path(__file__).with_name("_xgb_v2_model_test.json")
        metadata = Path(__file__).with_name("_xgb_v2_metadata_test.json")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.addCleanup(lambda: metadata.unlink(missing_ok=True))
        model.save_model(path)
        metadata.write_text(json.dumps({"feature_order": FEATURE_FIELDS,
                                       "model_sha256": sha256(path.read_bytes()).hexdigest(),
                                       "residual_target": True}), encoding="utf-8")
        expected = x[:, LATEST_INDEX].astype(float) + model.predict(x)
        np.testing.assert_allclose(predict_saved_variant(path, metadata, x,
                                                        FEATURE_FIELDS), expected)
        with self.assertRaisesRegex(ValueError, "order"):
            predict_saved_variant(path, metadata, x, FEATURE_FIELDS[::-1])

    @unittest.skipUnless((V2_DIR / "v2_results.json").exists(), "V2 artifacts absent")
    def test_artifacts_preserve_v1_and_partition_counts(self):
        results = json.loads((V2_DIR / "v2_results.json").read_text(encoding="utf-8"))
        v1 = json.loads((V1_DIR / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(results["v1_model_sha256"], v1["model_sha256"])
        self.assertEqual(sha256((V1_DIR / "model.json").read_bytes()).hexdigest(),
                         v1["model_sha256"])
        self.assertEqual(results["counts"]["tune_evaluate_boundary"], 12)
        self.assertEqual(results["counts"]["evaluate"], 44040)
        self.assertEqual(set(results["variants"]), set(VARIANTS))
        for name, info in results["variants"].items():
            self.assertEqual(info["feature_files"], v1["feature_files"])
            self.assertEqual(sha256((V2_DIR / name / "model.json").read_bytes()).hexdigest(),
                             info["model_sha256"])


if __name__ == "__main__":
    unittest.main()
