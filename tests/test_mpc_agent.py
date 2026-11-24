"""Tests for the first receding-horizon trading controller."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from solaralpha.decision_time import DecisionView, evaluate_decision_policy
from solaralpha.mpc_agent import mpc_action, persistence_forecast
from solaralpha.solar_sim import Config, Observation


class MPCAgentTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.now = datetime(2025, 1, 15, 15, tzinfo=timezone(timedelta(hours=11)))

    def view(self, solar, load, buy, sell):
        return DecisionView(self.now, 0.50, self.now, solar, load, buy, sell)

    def test_forecast_repeats_only_completed_readings(self):
        view = self.view(2.0, 1.0, 0.25, 0.05)
        forecast = persistence_forecast(view, self.config, horizon=3)
        self.assertEqual([record.time for record in forecast],
                         [self.now + timedelta(minutes=5 * index) for index in range(3)])
        self.assertEqual([(record.solar_kw, record.load_kw,
                           record.buy_price, record.sell_price) for record in forecast],
                         [(2.0, 1.0, 0.25, 0.05)] * 3)
        first = replace(view, last_solar_kw=None, last_load_kw=None,
                        last_buy_price=None, last_sell_price=None)
        self.assertEqual(persistence_forecast(first, self.config), [])
        self.assertEqual(mpc_action(first, self.config), 0.0)

    def test_plan_respects_price_and_energy_tradeoffs(self):
        self.assertEqual(mpc_action(self.view(0.0, 1.0, 0.30, 0.05),
                                    self.config), -1.0)
        self.assertEqual(mpc_action(self.view(0.0, 0.0, 0.30, 1.00),
                                    self.config), -self.config.discharge_kw)
        self.assertEqual(mpc_action(self.view(5.0, 0.0, 0.30, -0.10),
                                    self.config), self.config.charge_kw)

    def test_hidden_current_price_does_not_change_current_action(self):
        data = [Observation(self.now + timedelta(minutes=5 * i),
                            0.0, 0.0, 0.30, 0.05) for i in range(3)]
        changed = list(data)
        changed[1] = replace(changed[1], sell_price=1.00)
        actions = []
        for scenario in (data, changed):
            requests = []

            def record(view, config):
                request = mpc_action(view, config, horizon=3)
                requests.append(request)
                return request

            evaluate_decision_policy(record, self.config, scenario)
            actions.append(requests)
        self.assertEqual(actions[0][:2], actions[1][:2])
        self.assertEqual(actions[0][2], 0.0)
        self.assertEqual(actions[1][2], -self.config.discharge_kw)


if __name__ == "__main__":
    unittest.main()
