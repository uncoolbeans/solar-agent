"""Tabular comparison guards for schema, split isolation, persistence and matching."""

import csv
import gzip
from hashlib import sha256
import json
from pathlib import Path
import unittest

from solaralpha.historical_pipeline import ROOT
from solaralpha.price_forecasting.features import FEATURE_FIELDS
from solaralpha.price_forecasting.ml_dataset import Slice, Slices, period_for_row
from solaralpha.price_forecasting.tabular_comparison import (
    ALL_NAMES, NAMES, OUTPUT_DIR, LEADERBOARD_PATH, _fit_linear, _save_model,
    assert_schema, predict_saved,
)
from solaralpha.price_forecasting.xgb_v2_diagnostics import V2_DIR


class TabularComparisonTests(unittest.TestCase):
    def test_feature_schema_rejects_reordering_and_target_column(self):
        import numpy as np
        x = np.zeros((2, len(FEATURE_FIELDS)), dtype=np.float32)
        assert_schema(x, FEATURE_FIELDS)
        with self.assertRaises(ValueError):
            assert_schema(x, FEATURE_FIELDS[::-1])
        with self.assertRaises(ValueError):
            assert_schema(np.zeros((2, len(FEATURE_FIELDS) + 1)), FEATURE_FIELDS)

    def test_linear_scaler_uses_training_rows_only_and_predictions_reproduce(self):
        import numpy as np
        rng = np.random.default_rng(12)
        train_x = rng.normal(size=(140, len(FEATURE_FIELDS))).astype(np.float32)
        tune_x = rng.normal(loc=100, size=(30, len(FEATURE_FIELDS))).astype(np.float32)
        y = 0.3 * train_x[:, 0] - 0.1 * train_x[:, 1] + 0.04
        dataset = Slices(Slice(train_x, y, []), Slice(tune_x, np.ones(30), []),
                         Slice(tune_x, np.ones(30), []), 0.0, (), {}, {})
        for name in ("ridge", "lasso"):
            model, _, detail = _fit_linear(name, dataset)
            np.testing.assert_allclose(model.named_steps["standardscaler"].mean_,
                                       train_x.mean(axis=0, dtype=float), atol=1e-6)
            self.assertLess(abs(detail["scaler_mean"][0]), 1)
            path = ROOT / "tests" / "_tabular_test_model.joblib"
            meta = ROOT / "tests" / "_tabular_test_metadata.json"
            self.addCleanup(lambda: path.unlink(missing_ok=True))
            self.addCleanup(lambda: meta.unlink(missing_ok=True))
            path = _save_model(name, model, path.parent)
            path = path.replace(path.parent / "_tabular_test_model.joblib")
            meta.write_text(json.dumps({"name": name, "feature_order": FEATURE_FIELDS,
                                        "model_sha256": sha256(path.read_bytes()).hexdigest()}),
                            encoding="utf-8")
            np.testing.assert_allclose(predict_saved(name, path, meta, tune_x,
                                                     FEATURE_FIELDS), model.predict(tune_x))
            with self.assertRaises(ValueError):
                predict_saved(name, path, meta, tune_x, FEATURE_FIELDS[::-1])
            # Fixed solver and seed yield the same inference after a second fit.
            again, _, _ = _fit_linear(name, dataset)
            np.testing.assert_allclose(again.predict(tune_x), model.predict(tune_x))

    def test_partition_boundary_and_development_exclusion(self):
        common = {"origin_local_date": "2025-06-30", "origin_partition": "validation",
                  "target_partition": "validation", "include_in_validation_score": "1",
                  "is_development_origin": "0", "is_development_target": "0"}
        row = {**common, "decision_start_utc": "2025-06-30T22:30:00+00:00",
               "target_interval_start_utc": "2025-06-30T23:30:00+00:00",
               "origin_local_date": "2025-07-01"}
        self.assertEqual(period_for_row(row), "evaluate")
        boundary = {**common, "decision_start_utc": "2025-06-30T13:30:00+00:00",
                    "target_interval_start_utc": "2025-06-30T14:30:00+00:00"}
        self.assertIsNone(period_for_row(boundary))
        excluded = {**row, "include_in_validation_score": "0",
                    "is_development_origin": "1"}
        self.assertIsNone(period_for_row(excluded))

    def test_fixed_seed_tree_predictions_reproduce_with_negative_targets(self):
        import numpy as np
        from sklearn.ensemble import RandomForestRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, len(FEATURE_FIELDS))).astype(np.float32)
        y = np.where(x[:, 0] < 0, -0.12, 0.08) + 0.01 * x[:, 1]
        y[3] = 8.0  # The toy exercise retains a genuine extreme target.
        builders = (
            lambda: RandomForestRegressor(n_estimators=6, max_depth=3,
                                          random_state=42, n_jobs=1),
            lambda: LGBMRegressor(n_estimators=20, objective="regression_l1",
                                  metric="l1", min_child_samples=5,
                                  random_state=42, n_jobs=1, verbosity=-1),
            lambda: CatBoostRegressor(iterations=20, depth=2, loss_function="MAE",
                                      eval_metric="MAE", thread_count=1,
                                      random_seed=42, verbose=False,
                                      allow_writing_files=False),
        )
        for make in builders:
            first, second = make(), make()
            first.fit(x, y)
            second.fit(x, y)
            np.testing.assert_allclose(first.predict(x), second.predict(x),
                                       rtol=0, atol=1e-10)

    def test_evaluation_labels_cannot_affect_training_or_tuning_selection(self):
        import numpy as np
        rng = np.random.default_rng(7)
        train = Slice(rng.normal(size=(60, len(FEATURE_FIELDS))).astype(np.float32),
                      rng.normal(size=60), [])
        tune = Slice(rng.normal(size=(20, len(FEATURE_FIELDS))).astype(np.float32),
                     rng.normal(size=20), [])
        evaluation = Slice(rng.normal(size=(10, len(FEATURE_FIELDS))).astype(np.float32),
                           rng.normal(size=10), [])
        original = Slices(train, tune, evaluation, 0, (), {}, {})
        changed = Slices(train, tune, Slice(evaluation.x, evaluation.y + 1000, []),
                         0, (), {}, {})
        one, _, selected_one = _fit_linear("ridge", original)
        two, _, selected_two = _fit_linear("ridge", changed)
        self.assertEqual(selected_one["selected_alpha"], selected_two["selected_alpha"])
        np.testing.assert_array_equal(one.predict(evaluation.x),
                                      two.predict(evaluation.x))

    @unittest.skipUnless((OUTPUT_DIR / "results.json").exists(), "Tabular artifacts absent")
    def test_saved_models_and_leaderboard_match_results(self):
        import numpy as np
        report = json.loads((OUTPUT_DIR / "results.json").read_text(encoding="utf-8"))
        board = json.loads(LEADERBOARD_PATH.read_text(encoding="utf-8"))
        self.assertEqual([entry["name"] for entry in board["entries"][:len(ALL_NAMES)]],
                         list(ALL_NAMES))
        self.assertEqual(board["split_counts"]["train"], 208800)
        self.assertEqual(board["split_counts"]["tune"], 50100)
        self.assertEqual(board["split_counts"]["evaluate"], 44040)
        self.assertEqual(board["feature_order"], list(FEATURE_FIELDS))
        self.assertEqual(board["feature_files"], report["feature_files"])
        for entry in board["entries"][:len(ALL_NAMES)]:
            name = entry["name"]
            self.assertEqual(entry["reused_development_evaluation"],
                             report["splits"]["evaluate"]["models"][name])
            self.assertIsNone(entry["downstream_mpc_financial_evaluation"])
        matrix = np.zeros((1, len(FEATURE_FIELDS)), dtype=np.float32)
        for name in NAMES:
            detail = report["models"][name]
            path = ROOT / detail["artifact"]
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), detail["model_sha256"])
            value = predict_saved(name, path, path.parent / "metadata.json", matrix,
                                  FEATURE_FIELDS)
            self.assertTrue(np.isfinite(value).all())

    @unittest.skipUnless((OUTPUT_DIR / "evaluate_comparison.csv.gz").exists(),
                         "Tabular predictions absent")
    def test_predictions_match_frozen_xgboost_rows(self):
        with gzip.open(OUTPUT_DIR / "evaluate_comparison.csv.gz", "rt", encoding="utf-8") as stream:
            new = list(csv.DictReader(stream))
        with gzip.open(V2_DIR / "evaluate_comparison.csv.gz", "rt", encoding="utf-8") as stream:
            frozen = list(csv.DictReader(stream))
        self.assertEqual(len(new), 44040)
        self.assertEqual(len(new), len(frozen))
        fields = ("decision_start_utc", "target_interval_start_utc", "origin_local_date",
                  "actual_aud_per_kwh") + tuple(f"{n}_aud_per_kwh" for n in ALL_NAMES[:9])
        for a, b in zip(new, frozen):
            for field in fields:
                self.assertEqual(a[field], b[field])


if __name__ == "__main__":
    unittest.main()
