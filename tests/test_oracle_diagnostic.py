"""Forecast isolation and accounting checks for the offline MPC diagnostic."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from solaralpha.decision_time import (decision_view, evaluate_decision_policy,
                                      last_observed_self_consumption,
                                      last_observed_threshold)
from solaralpha.forecasting import PersistenceForecaster
from solaralpha.historical_replay import build_observations
from solaralpha.mpc_agent import forecast_action, mpc_action
from solaralpha.oracle_diagnostic import (FORECAST_CASES, _OfflineOracleForecaster,
                                          run_comparison)
from solaralpha.solar_sim import Config, Observation
from solaralpha.spike_sensitivity import cap_spot_prices


class OracleDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        start = datetime(2025, 1, 15, tzinfo=timezone(timedelta(hours=11)))
        self.episode = [
            Observation(start + timedelta(minutes=5 * index),
                        float(index), 1.0 + index,
                        0.20 + 0.10 * index, 0.05 + 0.15 * index)
            for index in range(4)
        ]

    def forecast_at(self, index, case, episode=None, horizon=3):
        data = self.episode if episode is None else episode
        view = decision_view(data[index].time, self.config.initial_soc,
                             data[index - 1] if index else None, self.config)
        return _OfflineOracleForecaster(data, index, case).forecast(
            view, self.config, horizon)

    def test_each_oracle_replaces_only_its_selected_fields(self):
        last = self.episode[0]
        expected = (
            (last.solar_kw, last.load_kw, last.buy_price, last.sell_price),
            (last.solar_kw, last.load_kw, self.episode[1].buy_price,
             self.episode[1].sell_price),
            (self.episode[1].solar_kw, self.episode[1].load_kw,
             last.buy_price, last.sell_price),
            (self.episode[1].solar_kw, self.episode[1].load_kw,
             self.episode[1].buy_price, self.episode[1].sell_price),
        )
        for case, first_expected in zip(FORECAST_CASES, expected):
            with self.subTest(case=case.name):
                records = self.forecast_at(1, case)
                self.assertEqual(len(records), 3)
                self.assertEqual((records[0].solar_kw, records[0].load_kw,
                                  records[0].buy_price, records[0].sell_price),
                                 first_expected)
                self.assertEqual([item.time for item in records],
                                 [item.time for item in self.episode[1:4]])

    def test_first_step_and_episode_edge_have_defined_forecasts(self):
        for case in FORECAST_CASES[:-1]:
            self.assertEqual(self.forecast_at(0, case), [])
        self.assertEqual(self.forecast_at(0, FORECAST_CASES[-1]),
                         self.episode[:3])
        records = self.forecast_at(3, FORECAST_CASES[-1], horizon=3)
        self.assertEqual(records[0], self.episode[3])
        self.assertEqual([item.time for item in records],
                         [self.episode[3].time + timedelta(minutes=5 * i)
                          for i in range(3)])

    def test_ordinary_mpc_is_independent_of_hidden_future(self):
        changed = list(self.episode)
        changed[1] = replace(changed[1], buy_price=9.0, sell_price=8.0)
        view = decision_view(self.episode[1].time, 0.5,
                             self.episode[0], self.config)
        ordinary = PersistenceForecaster().forecast(view, self.config, 3)
        self.assertEqual(forecast_action(view, self.config,
                                         PersistenceForecaster()),
                         mpc_action(view, self.config))
        self.assertEqual(ordinary[0].buy_price, self.episode[0].buy_price)
        self.assertNotEqual(self.forecast_at(1, FORECAST_CASES[1])[0].buy_price,
                            self.forecast_at(1, FORECAST_CASES[1], changed)[0].buy_price)
        self.assertEqual(self.forecast_at(1, FORECAST_CASES[2])[0].buy_price,
                         self.forecast_at(1, FORECAST_CASES[2], changed)[0].buy_price)

    def test_comparison_preserves_benchmarks_and_accounting(self):
        results = run_comparison(self.config, self.episode)
        self.assertEqual(set(results), {"decision_self", "decision_threshold"}
                         | {case.name for case in FORECAST_CASES})
        for name, policy in (("decision_self", last_observed_self_consumption),
                             ("decision_threshold", last_observed_threshold),
                             ("persistence_mpc", mpc_action)):
            reference = evaluate_decision_policy(policy, self.config,
                                                  self.episode)
            for key in reference:
                self.assertAlmostEqual(results[name][0][key], reference[key])
        for totals, rows in results.values():
            self.assertEqual(len(rows), len(self.episode))
            self.assertAlmostEqual(totals["cash_flow"],
                                   totals["export_revenue"] - totals["import_cost"])
            self.assertAlmostEqual(totals["reward"],
                                   totals["cash_flow"] - totals["wear"])
            self.assertAlmostEqual(totals["throughput_kwh"],
                                   totals["charge_kwh"] + totals["discharge_kwh"])
            self.assertAlmostEqual(totals["wear"],
                                   self.config.wear_per_kwh * totals["discharge_kwh"])
            self.assertAlmostEqual(totals["adjusted_reward"],
                                   totals["reward"]
                                   + (totals["final_soc"] - self.config.initial_soc)
                                   * self.config.capacity_kwh
                                   * self.config.eta_discharge * 0.20)
            self.assertGreaterEqual(totals["min_soc"], self.config.reserve - 1e-12)
            self.assertTrue(all(info["export_kwh"] <= self.config.export_kw
                                * self.config.dt + 1e-12
                                for _, info, _ in rows))

    def test_capped_scenario_prices_reach_oracle_only(self):
        observed, prices_mwh = build_observations(self.config)
        capped = cap_spot_prices(observed, prices_mwh, 1.10)
        index = next(i for i in range(len(observed))
                     if observed[i].buy_price != capped[i].buy_price)
        view = decision_view(capped[index].time, 0.5,
                             capped[index - 1], self.config)
        baseline = _OfflineOracleForecaster(capped, index, FORECAST_CASES[0]).forecast(
            view, self.config, 12)[0]
        oracle = _OfflineOracleForecaster(capped, index, FORECAST_CASES[1]).forecast(
            view, self.config, 12)[0]
        self.assertEqual(baseline.buy_price, capped[index - 1].buy_price)
        self.assertEqual(oracle.buy_price, capped[index].buy_price)
        self.assertEqual(oracle.sell_price, capped[index].sell_price)
        self.assertLess(oracle.buy_price, observed[index].buy_price)
        self.assertEqual(oracle.solar_kw, baseline.solar_kw)
        self.assertEqual(oracle.load_kw, baseline.load_kw)


if __name__ == "__main__":
    unittest.main()
