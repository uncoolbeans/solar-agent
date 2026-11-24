"""Checks for the pinned historical input slice and its time alignment."""

import unittest
from datetime import datetime, timedelta, timezone
from statistics import median

from solaralpha.historical_replay import AEDT, STEP, UTC, build_observations, lagged_threshold_policy, modeled_load_kw
from solaralpha.solar_sim import (BatteryEnv, Config, Observation,
                                  evening_export_schedule, evaluate, self_consumption)
from solaralpha.spike_sensitivity import cap_spot_prices, score_policies
from solaralpha.tariffs import ILLUSTRATIVE_FIXED


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

    def test_typical_wholesale_price_and_spike_are_distinct(self):
        day = [self.prices_mwh[obs.time.astimezone(UTC)] / 1000.0
               for obs in self.data if obs.time.date() == datetime(2025, 1, 15).date()]
        self.assertAlmostEqual(median(day), 0.096235)
        self.assertEqual(sum(price == 17.50 for price in day), 3)

    def test_hourly_irradiance_is_a_bounded_solar_proxy(self):
        noon_index = 2 * 288 + 12 * 12  # 15 January, 12:00 Sydney time
        expected_kw = 10.0 * 0.83 * 1014.0 / 1000.0
        self.assertAlmostEqual(self.data[noon_index].solar_kw, expected_kw)
        self.assertTrue(all(self.data[noon_index + i].solar_kw == self.data[noon_index].solar_kw
                            for i in range(12)))
        self.assertTrue(all(0 <= obs.solar_kw <= self.config.solar_kw for obs in self.data))

    def test_cooling_load_responds_to_temperature(self):
        afternoon = datetime(2025, 1, 15, 16, tzinfo=AEDT)
        cool = modeled_load_kw(afternoon, 20.0)
        self.assertAlmostEqual(modeled_load_kw(afternoon, 24.0), cool)
        self.assertAlmostEqual(modeled_load_kw(afternoon, 28.0) - cool, 1.0)

        hot_day = self.data[2 * 288:3 * 288]
        cool_day = self.data[4 * 288:5 * 288]
        self.assertGreater(sum(obs.load_kw for obs in hot_day) * self.config.dt,
                           sum(obs.load_kw for obs in cool_day) * self.config.dt + 5.0)
        repeated, _ = build_observations(self.config)
        self.assertEqual([obs.load_kw for obs in self.data],
                         [obs.load_kw for obs in repeated])

    def test_tariff_comparison_keeps_physical_inputs_identical(self):
        fixed, _ = build_observations(self.config, ILLUSTRATIVE_FIXED)
        for fixed_obs, spot_obs in zip(fixed, self.data):
            self.assertEqual((fixed_obs.time, fixed_obs.solar_kw, fixed_obs.load_kw),
                             (spot_obs.time, spot_obs.solar_kw, spot_obs.load_kw))
            self.assertEqual((fixed_obs.buy_price, fixed_obs.sell_price), (0.30, 0.05))
        self_result = evaluate(self_consumption, self.config, fixed)
        trader_result = evaluate(lagged_threshold_policy(), self.config, fixed)
        self.assertAlmostEqual(self_result["reward"], trader_result["reward"])
        self.assertAlmostEqual(self_result["adjusted_reward"], trader_result["adjusted_reward"])
        fixed_schedule = evaluate(evening_export_schedule, self.config, fixed)
        spot_schedule = evaluate(evening_export_schedule, self.config, self.data)
        for key in ("import_kwh", "export_kwh", "final_soc"):
            self.assertAlmostEqual(fixed_schedule[key], spot_schedule[key])

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

    def test_spike_cap_changes_only_prices_and_preserves_dispatch(self):
        capped = cap_spot_prices(self.data, self.prices_mwh, 1.10)
        changed = 0
        for original, counterfactual in zip(self.data, capped):
            self.assertEqual((original.time, original.solar_kw, original.load_kw),
                             (counterfactual.time, counterfactual.solar_kw,
                              counterfactual.load_kw))
            changed += (original.buy_price, original.sell_price) != (
                counterfactual.buy_price, counterfactual.sell_price)
        self.assertEqual(changed, 3)

        observed_scores = score_policies(self.config, self.data)
        capped_scores = score_policies(self.config, capped)
        for name in observed_scores:
            with self.subTest(policy=name):
                for key in ("import_kwh", "export_kwh", "final_soc"):
                    self.assertAlmostEqual(observed_scores[name][key],
                                           capped_scores[name][key])


if __name__ == "__main__":
    unittest.main()
