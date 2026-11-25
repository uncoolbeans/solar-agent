"""Hand-verifiable physical/economic cases for the offline MILP."""

import importlib.util
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from solaralpha.perfect_foresight import optimize_episode
from solaralpha.perfect_foresight_experiment import gaps_to_benchmark
from solaralpha.solar_sim import Config, Observation


@unittest.skipUnless(importlib.util.find_spec("highspy"),
                     "Install requirements-optimizer.txt")
class PerfectForesightTests(unittest.TestCase):
    def obs(self, solar=0, load=0, buy=0.3, sell=0.05, minute=0):
        return Observation(datetime(2025, 1, 1, tzinfo=timezone.utc)
                           + timedelta(minutes=minute), solar, load, buy, sell)

    def solve(self, config, episode):
        result = optimize_episode(config, episode, time_limit_s=10)
        self.assertTrue(result.proven_optimal, result.status)
        self.assertTrue(result.replay_agrees, result.max_replay_error)
        self.assertAlmostEqual(result.objective, result.metrics["adjusted_reward"], places=5)
        self.assertAlmostEqual(result.objective_bound, result.objective, places=5)
        return result

    def test_negative_export_price_cannot_curtail_below_export_cap(self):
        c = replace(Config(), charge_kw=0, discharge_kw=0)
        r = self.solve(c, [self.obs(solar=4, sell=-0.20)])
        self.assertAlmostEqual(r.metrics["export_kwh"], 4 * c.dt)
        self.assertAlmostEqual(r.metrics["curtailed_kwh"], 0)
        self.assertAlmostEqual(r.objective, -0.20 * 4 * c.dt)

    def test_solar_above_export_cap_forces_curtailment(self):
        c = replace(Config(), charge_kw=0, discharge_kw=0)
        r = self.solve(c, [self.obs(solar=8, sell=-0.20)])
        self.assertAlmostEqual(r.metrics["export_kwh"], 5 * c.dt)
        self.assertAlmostEqual(r.metrics["curtailed_kwh"], 3 * c.dt)

    def test_full_battery_cannot_charge_and_depleted_cannot_discharge(self):
        full = replace(Config(), initial_soc=1)
        r = self.solve(full, [self.obs(solar=0, load=0, buy=-1)])
        self.assertLessEqual(r.actions_kw[0], 1e-8)
        depleted = replace(Config(), initial_soc=0.2)
        r = self.solve(depleted, [self.obs(solar=0, load=2, buy=1)])
        self.assertGreaterEqual(r.actions_kw[0], -1e-8)
        self.assertAlmostEqual(r.metrics["import_kwh"], 2 * depleted.dt)

    def test_negative_buy_charges_without_simultaneous_grid_export(self):
        c = replace(Config(), initial_soc=0.5)
        r = self.solve(c, [self.obs(solar=0, load=0, buy=-1, sell=0.05)])
        self.assertAlmostEqual(r.actions_kw[0], c.charge_kw)
        self.assertAlmostEqual(r.metrics["import_kwh"], c.charge_kw * c.dt)
        self.assertAlmostEqual(r.metrics["export_kwh"], 0)

    def test_exclusive_grid_flows_prevent_negative_price_arbitrage(self):
        c = replace(Config(), charge_kw=0, discharge_kw=0)
        r = self.solve(c, [self.obs(buy=-1, sell=1)])
        self.assertAlmostEqual(r.metrics["import_kwh"], 0)
        self.assertAlmostEqual(r.metrics["export_kwh"], 0)
        self.assertAlmostEqual(r.objective, 0)

    def test_solar_surplus_and_export_constraint(self):
        c = replace(Config(), initial_soc=0.5)
        r = self.solve(c, [self.obs(solar=8, sell=1)])
        self.assertAlmostEqual(r.metrics["export_kwh"], c.export_kw * c.dt)
        self.assertGreaterEqual(r.metrics["curtailed_kwh"], 0)
        self.assertGreaterEqual(r.actions_kw[0], -1e-8)

    def test_discharge_to_cover_load_is_hand_calculable(self):
        c = replace(Config(), initial_soc=0.5)
        r = self.solve(c, [self.obs(load=2, buy=0.5)])
        self.assertAlmostEqual(r.actions_kw[0], -2)
        self.assertAlmostEqual(r.metrics["import_kwh"], 0)
        self.assertAlmostEqual(r.metrics["wear"], 2 * c.dt * c.wear_per_kwh)
        expected = -2 * c.dt * (c.wear_per_kwh + 0.20)
        self.assertAlmostEqual(r.objective, expected)

    def test_one_terminal_adjustment_only(self):
        c = replace(Config(), charge_kw=0, discharge_kw=0)
        r = self.solve(c, [self.obs(minute=0), self.obs(minute=5)])
        self.assertAlmostEqual(r.objective, 0)
        self.assertAlmostEqual(r.metrics["adjusted_reward"], 0)

    def test_two_step_charge_then_discharge_replays(self):
        c = Config()
        r = self.solve(c, [self.obs(buy=-1, minute=0),
                           self.obs(load=5, buy=1, minute=5)])
        self.assertGreater(r.actions_kw[0], 0)
        self.assertLess(r.actions_kw[1], 0)
        self.assertGreaterEqual(r.metrics["min_soc"], c.reserve - 1e-8)

    def test_gap_and_opportunity_capture_use_adjusted_reward(self):
        result = gaps_to_benchmark({"adjusted_reward": 25, "reward": 999}, 20, 30)
        self.assertEqual(result["remaining_gap_aud"], 5)
        self.assertEqual(result["opportunity_capture_pct"], 50)
        self.assertEqual(gaps_to_benchmark({"adjusted_reward": 20}, 20, 30)
                         ["opportunity_capture_pct"], 0)


if __name__ == "__main__":
    unittest.main()
