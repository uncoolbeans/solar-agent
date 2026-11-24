"""Decision-time information and accounting checks."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from solaralpha.decision_time import (evaluate_decision_policy,
                                      last_observed_threshold)
from solaralpha.solar_sim import Config, Observation, evaluate


class DecisionTimeTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        self.start = datetime(2025, 1, 15, 14, 50, tzinfo=timezone(timedelta(hours=11)))
        self.data = [
            Observation(self.start, 0.0, 0.0, 0.20, 0.05),
            Observation(self.start + timedelta(minutes=5), 0.0, 0.0, 0.20, 0.05),
            Observation(self.start + timedelta(minutes=10), 0.0, 0.0, 0.20, 0.05),
        ]

    def test_current_interval_is_hidden_until_the_next_decision(self):
        changed = list(self.data)
        changed[1] = replace(changed[1], solar_kw=10.0, load_kw=2.0,
                             buy_price=17.67, sell_price=17.49)
        decisions = []
        for scenario in (self.data, changed):
            views, actions = [], []

            def policy(view, config):
                views.append(view)
                action = last_observed_threshold(view, config)
                actions.append(action)
                return action

            result = evaluate_decision_policy(policy, self.config, scenario)
            decisions.append((views, actions, result))

        original_views, original_actions, original_result = decisions[0]
        changed_views, changed_actions, changed_result = decisions[1]
        self.assertEqual(original_views[0].time, self.start)
        self.assertEqual(original_views[0].soc, self.config.initial_soc)
        self.assertIsNone(original_views[0].last_buy_price)
        self.assertIsNone(original_views[0].last_solar_kw)
        self.assertEqual(original_views[1], changed_views[1])
        self.assertEqual(original_actions[:2], changed_actions[:2])
        self.assertEqual(original_views[1].last_interval_end, self.data[1].time)
        self.assertEqual(changed_views[2].last_sell_price, 17.49)
        self.assertEqual(changed_views[2].last_solar_kw, 10.0)
        self.assertEqual(original_actions[2], 0.0)
        self.assertEqual(changed_actions[2], -self.config.discharge_kw)
        self.assertFalse(hasattr(changed_views[1], "sell_price"))
        self.assertNotEqual(original_result["reward"], changed_result["reward"])

    def test_filtered_runner_keeps_existing_physics_and_reward(self):
        hold = lambda view, config: 0.0
        raw_hold = lambda obs, soc, config: 0.0
        filtered = evaluate_decision_policy(hold, self.config, self.data)
        raw = evaluate(raw_hold, self.config, self.data)
        self.assertEqual(filtered, raw)


if __name__ == "__main__":
    unittest.main()
