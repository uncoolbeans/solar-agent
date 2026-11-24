"""Horizon conversion, oracle boundary, and import-attribution tests."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from solaralpha.decision_time import decision_view
from solaralpha.horizon_sensitivity import (benchmark_decisions, horizon_steps,
                                            oracle_controller,
                                            run_horizon_sensitivity)
from solaralpha.oracle_diagnostic import (FORECAST_CASES, _OfflineOracleForecaster,
                                          run_comparison, run_episode)
from solaralpha.solar_sim import Config, Observation


class HorizonSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        start = datetime(2025, 1, 15, tzinfo=timezone(timedelta(hours=11)))
        self.episode = [
            Observation(start + timedelta(minutes=5 * index),
                        1.0 + index, 2.0, 0.20 + 0.01 * index,
                        0.05 + 0.01 * index)
            for index in range(4)
        ]

    def test_hour_conversion_and_default_are_unchanged(self):
        self.assertEqual([horizon_steps(hours, self.config)
                          for hours in (1, 3, 6, 12, 24)],
                         [12, 36, 72, 144, 288])
        with self.assertRaises(ValueError):
            horizon_steps(0, self.config)
        with self.assertRaises(ValueError):
            horizon_steps(0.1, self.config)
        reference = run_comparison(self.config, self.episode)["oracle_all"][0]
        one_hour = run_episode(self.config, self.episode,
                               oracle_controller(self.episode, 12))[0]
        for key in ("cash_flow", "reward", "adjusted_reward", "import_kwh",
                    "export_kwh", "wear", "final_soc"):
            self.assertAlmostEqual(reference[key], one_hour[key])

    def test_full_horizon_is_padded_without_reading_past_the_episode(self):
        view = decision_view(self.episode[-1].time, 0.5,
                             self.episode[-2], self.config)
        forecast = _OfflineOracleForecaster(
            self.episode, len(self.episode) - 1, FORECAST_CASES[-1]
        ).forecast(view, self.config, 288)
        self.assertEqual(len(forecast), 288)
        self.assertEqual(forecast[0], self.episode[-1])
        self.assertEqual(forecast[-1].time,
                         self.episode[-1].time + timedelta(minutes=5 * 287))
        self.assertEqual(forecast[-1].buy_price, self.episode[-2].buy_price)
        self.assertEqual(forecast[-1].solar_kw, self.episode[-2].solar_kw)

    def test_short_scenario_suite_orders_horizons_and_preserves_baselines(self):
        observed = self.episode
        capped = list(observed)
        capped[2] = replace(capped[2], sell_price=0.01)
        order = []
        results = run_horizon_sensitivity(
            self.config, {"observed": observed, "capped": capped},
            hours=(3, 1),
            on_result=lambda scenario, name, metrics: order.append((scenario, name)),
        )
        self.assertEqual(order[-4:], [("observed", "oracle_1h"),
                                      ("capped", "oracle_1h"),
                                      ("observed", "oracle_3h"),
                                      ("capped", "oracle_3h")])
        for scenario in results.values():
            self.assertEqual(set(scenario), {"decision_self", "decision_threshold",
                                             "oracle_1h", "oracle_3h"})
            for totals, _ in scenario.values():
                self.assertGreaterEqual(totals["decision_mean_ms"], 0)
                self.assertGreaterEqual(totals["simulation_runtime_s"], 0)

    def test_timing_sample_does_not_run_a_financial_episode(self):
        estimate = benchmark_decisions(self.config, self.episode, 1,
                                       sample_count=3)
        self.assertEqual(estimate["samples"], 3)
        self.assertGreaterEqual(estimate["mean_ms"], 0)
        self.assertGreaterEqual(estimate["estimated_episode_s"], 0)

    def test_grid_import_breakdown_uses_executed_action(self):
        start = self.episode[0].time
        data = [Observation(start + timedelta(minutes=5 * i),
                            0.0, 1.0, 0.20, 0.05) for i in range(3)]
        actions = (5.0, 0.0, -5.0)
        totals, _ = run_episode(self.config, data,
                                lambda view, config, index: actions[index])
        self.assertAlmostEqual(totals["import_while_charging_kwh"], 6 / 12)
        self.assertAlmostEqual(totals["import_due_to_charging_kwh"], 5 / 12)
        self.assertAlmostEqual(totals["import_while_holding_kwh"], 1 / 12)
        self.assertAlmostEqual(totals["import_while_discharging_kwh"], 0)
        self.assertAlmostEqual(sum(totals[key] for key in
                                   ("import_while_charging_kwh",
                                    "import_while_holding_kwh",
                                    "import_while_discharging_kwh")),
                               totals["import_kwh"])
        reserve_config = replace(self.config, initial_soc=self.config.reserve)
        constrained, _ = run_episode(reserve_config, data[:1],
                                     lambda view, config, index: -5.0)
        self.assertAlmostEqual(constrained["import_at_reserve_kwh"], 1 / 12)
        self.assertAlmostEqual(constrained["import_when_discharge_clipped_kwh"],
                               1 / 12)


if __name__ == "__main__":
    unittest.main()
