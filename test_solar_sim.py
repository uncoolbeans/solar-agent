"""Small, repeatable checks for the simulated battery and its accounting."""

import unittest
from dataclasses import replace
from datetime import datetime, timezone

from solar_sim import (
    BatteryEnv,
    Config,
    Observation,
    make_data,
    self_consumption,
    threshold_trader,
)


class SolarSimTests(unittest.TestCase):
    def setUp(self):
        self.c = Config()
        self.time = datetime(2027, 1, 1, tzinfo=timezone.utc)

    def test_synthetic_data_repeats_with_the_same_seed(self):
        first = make_data(self.c, days=1, seed=42)
        self.assertEqual(len(first), 288)
        self.assertEqual(first, make_data(self.c, days=1, seed=42))
        self.assertNotEqual(first, make_data(self.c, days=1, seed=43))

    def test_executed_power_sign_and_efficiency(self):
        obs = Observation(self.time, 0.0, 0.0, 0.20, 0.10)
        charge_env = BatteryEnv(self.c, [obs])
        _, _, _, charge = charge_env.step(100.0)
        self.assertEqual(charge["power_kw"], self.c.charge_kw)
        self.assertAlmostEqual(
            charge_env.energy,
            15.0 + self.c.eta_charge * self.c.charge_kw * self.c.dt,
        )

        discharge_env = BatteryEnv(self.c, [obs])
        _, _, _, discharge = discharge_env.step(-100.0)
        self.assertEqual(discharge["power_kw"], -self.c.discharge_kw)
        self.assertAlmostEqual(
            discharge_env.energy,
            15.0 - self.c.discharge_kw * self.c.dt / self.c.eta_discharge,
        )

    def test_capacity_reserve_and_export_limit_clip_requested_power(self):
        flat = Observation(self.time, 0.0, 0.0, 0.20, 0.10)
        full = BatteryEnv(replace(self.c, initial_soc=1.0), [flat])
        _, _, _, info = full.step(100.0)
        self.assertEqual(info["power_kw"], 0.0)
        self.assertAlmostEqual(full.energy, self.c.capacity_kwh)

        at_reserve = BatteryEnv(replace(self.c, initial_soc=self.c.reserve), [flat])
        _, _, _, info = at_reserve.step(-100.0)
        self.assertEqual(info["power_kw"], 0.0)
        self.assertAlmostEqual(at_reserve.energy, self.c.capacity_kwh * self.c.reserve)

        high_solar = Observation(self.time, 10.0, 0.0, 0.20, 0.10)
        surplus = BatteryEnv(self.c, [high_solar])
        _, _, _, info = surplus.step(-100.0)
        self.assertEqual(info["power_kw"], 0.0)
        self.assertAlmostEqual(info["export_kwh"], self.c.export_kw * self.c.dt)
        self.assertAlmostEqual(info["curtailed_kwh"], 5.0 * self.c.dt)

    def test_energy_balance_and_reward_on_shared_scenario(self):
        data = make_data(self.c, days=1, seed=42)
        policies = (
            self_consumption,
            threshold_trader,
            lambda obs, soc, c: 100.0,
            lambda obs, soc, c: -100.0,
        )
        for policy in policies:
            with self.subTest(policy=policy.__name__):
                env = BatteryEnv(self.c, data)
                obs, soc = env.reset()
                for _ in data:
                    previous_energy = env.energy
                    next_obs, reward, done, info = env.step(policy(obs, soc, self.c))
                    power = info["power_kw"]  # The executed action may be clipped.
                    charge_kw = max(power, 0.0)
                    discharge_kw = max(-power, 0.0)
                    self.assertGreaterEqual(power, -self.c.discharge_kw)
                    self.assertLessEqual(power, self.c.charge_kw)
                    self.assertGreaterEqual(env.energy, self.c.capacity_kwh * self.c.reserve - 1e-9)
                    self.assertLessEqual(env.energy, self.c.capacity_kwh + 1e-9)
                    self.assertAlmostEqual(
                        env.energy,
                        previous_energy + self.c.eta_charge * charge_kw * self.c.dt
                        - discharge_kw * self.c.dt / self.c.eta_discharge,
                    )
                    self.assertAlmostEqual(
                        obs.solar_kw * self.c.dt + info["import_kwh"] + discharge_kw * self.c.dt,
                        obs.load_kw * self.c.dt + info["export_kwh"]
                        + charge_kw * self.c.dt + info["curtailed_kwh"],
                    )
                    self.assertLessEqual(info["export_kwh"], self.c.export_kw * self.c.dt + 1e-9)
                    self.assertFalse(info["import_kwh"] > 0 and info["export_kwh"] > 0)
                    self.assertAlmostEqual(
                        reward,
                        info["export_kwh"] * obs.sell_price
                        - info["import_kwh"] * obs.buy_price
                        - discharge_kw * self.c.dt * self.c.wear_per_kwh,
                    )
                    if done:
                        self.assertIsNone(next_obs)
                        break
                    obs, soc = next_obs


if __name__ == "__main__":
    unittest.main()
