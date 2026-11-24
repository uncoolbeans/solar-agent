"""Checks for the pinned historical input slice and its time alignment."""

import unittest
from datetime import datetime, timedelta, timezone

from historical_replay import AEDT, STEP, UTC, build_observations, lagged_threshold_policy
from solar_sim import BatteryEnv, Config, Observation


class HistoricalReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.data, cls.prices_mwh = build_observations(cls.config)

    def test_week_has_complete_five_minute_sydney_grid(self):
        self.assertEqual(len(self.data), 7 * 288)
        self.assertEqual(self.data[0].time, datetime(2025, 1, 13, tzinfo=AEDT))
        self.assertEqual(self.data[-1].time, datetime(2025, 1, 19, 23, 55, tzinfo=AEDT))
        self.assertTrue(all(later.time - earlier.time == STEP
                            for earlier, later in zip(self.data, self.data[1:])))

    def test_aemo_period_end_and_price_unit_conversion(self):
        # The 00:00 Sydney interval is labelled 23:05 on the prior AEST day.
        first_utc = self.data[0].time.astimezone(UTC)
        self.assertAlmostEqual(self.prices_mwh[first_utc], 144.88)
        self.assertAlmostEqual(self.data[0].buy_price, 144.88 / 1000 + 0.17)
        self.assertAlmostEqual(self.data[0].sell_price, 144.88 / 1000 - 0.01)
        self.assertAlmostEqual(max(self.prices_mwh.values()), 17500.0)

    def test_hourly_irradiance_is_a_bounded_solar_proxy(self):
        noon_index = 2 * 288 + 12 * 12  # 15 January, 12:00 Sydney time
        expected_kw = 10.0 * 0.83 * 1014.0 / 1000.0
        self.assertAlmostEqual(self.data[noon_index].solar_kw, expected_kw)
        self.assertTrue(all(self.data[noon_index + i].solar_kw == self.data[noon_index].solar_kw
                            for i in range(12)))
        self.assertTrue(all(0 <= obs.solar_kw <= self.config.solar_kw for obs in self.data))

    def test_threshold_sees_previous_completed_price(self):
        now = datetime(2025, 1, 13, tzinfo=AEDT)
        data = [Observation(now, 0.0, 0.0, 0.20, 0.50),
                Observation(now + timedelta(minutes=5), 0.0, 0.0, 0.20, 0.00)]
        env = BatteryEnv(self.config, data)
        policy = lagged_threshold_policy()
        obs, soc = env.reset()
        next_obs, _, _, first = env.step(policy(obs, soc, self.config))
        obs, soc = next_obs
        _, _, _, second = env.step(policy(obs, soc, self.config))
        self.assertEqual(first["power_kw"], 0.0)
        self.assertEqual(second["power_kw"], -self.config.discharge_kw)
        self.assertEqual(second["export_revenue"], 0.0)


if __name__ == "__main__":
    unittest.main()
