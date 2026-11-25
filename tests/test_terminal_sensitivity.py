"""Internal-versus-evaluation terminal value and resumable sweep checks."""

import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from solaralpha.decision_time import decision_view
from solaralpha.horizon_sensitivity import oracle_controller
from solaralpha.inspect_terminal_holds import hold_examples
from solaralpha.mpc_agent import mpc_action
from solaralpha.solar_sim import Config, Observation
from solaralpha.terminal_sensitivity import (EVALUATION_TERMINAL_VALUE,
                                             load_checkpoint, load_reference,
                                             run_setting)
from solaralpha.historical_replay import build_observations


class TerminalSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.config = Config()
        start = datetime(2025, 1, 15, tzinfo=timezone(timedelta(hours=11)))
        self.episode = [Observation(start + timedelta(minutes=5 * i),
                                    0.0, 1.0, 0.20, 0.05) for i in range(288)]

    def test_internal_value_changes_policy_without_changing_evaluation_formula(self):
        view = decision_view(self.episode[0].time, self.config.initial_soc,
                             None, self.config)
        self.assertEqual(oracle_controller(self.episode, 12)(view, self.config, 0),
                         oracle_controller(self.episode, 12, 0.20)(view, self.config, 0))
        next_view = decision_view(self.episode[1].time,
                                  self.config.initial_soc,
                                  self.episode[0], self.config)
        self.assertEqual(mpc_action(next_view, self.config),
                         mpc_action(next_view, self.config,
                                    terminal_value_per_kwh=0.20))
        for internal_value in (0.00, 0.40):
            with self.subTest(internal_value=internal_value):
                metrics = run_setting(self.config, self.episode, 1,
                                      internal_value, plot_day=1)["metrics"]
                expected_adjustment = (
                    (metrics["final_soc"] - self.config.initial_soc)
                    * self.config.capacity_kwh * self.config.eta_discharge
                    * EVALUATION_TERMINAL_VALUE
                )
                self.assertAlmostEqual(metrics["adjusted_reward"]
                                       - metrics["reward"], expected_adjustment)
        low = run_setting(self.config, self.episode, 1, 0.00,
                          plot_day=1)["metrics"]
        high = run_setting(self.config, self.episode, 1, 0.40,
                           plot_day=1)["metrics"]
        self.assertLess(low["final_soc"], high["final_soc"])

    def test_prior_reference_matches_pinned_episode(self):
        observed, _ = build_observations(self.config)
        reference = load_reference(self.config, {"observed": observed})
        self.assertEqual(reference["results"]["observed:1"]["adjusted_reward"],
                         17.67)
        self.assertEqual(reference["evaluation_terminal_value_per_kwh"], 0.20)
        with self.assertRaises(ValueError):
            load_reference(replace(self.config, capacity_kwh=20.0),
                           {"observed": observed})

    def test_hold_diagnostic_checks_feasible_balance_at_same_soc(self):
        point = {"time": self.episode[1].time.isoformat(),
                 "import_kwh": self.config.dt, "power_kw": 0.0,
                 "soc_before": 0.50}
        examples = hold_examples(self.config, self.episode, [point], 1,
                                 comparison_values=(0.00, 0.40))
        self.assertEqual(len(examples), 1)
        self.assertAlmostEqual(examples[0]["balance_power_kw"], -1.0)
        self.assertIn("0.00", examples[0]["same_soc_actions_kw"])

    def test_checkpoint_accepts_new_scenario_without_discarding_observed(self):
        saved = {"metadata": {
            "config_sha256": sha256(repr(self.config).encode()).hexdigest(),
            "episode_sha256": {"observed": "existing-observed-digest"},
            "evaluation_terminal_value_per_kwh": 0.20,
            "plot_day": 1,
        }, "results": {"observed:1:0.10": {"source": "computed"}}}
        with patch.object(Path, "exists", return_value=True), patch.object(
                Path, "read_text", return_value=json.dumps(saved)):
            loaded = load_checkpoint(self.config, {"capped": self.episode},
                                     Path("unused.json"), plot_day=1)
        self.assertIn("capped", loaded["metadata"]["episode_sha256"])
        self.assertEqual(loaded["metadata"]["episode_sha256"]["observed"],
                         "existing-observed-digest")


if __name__ == "__main__":
    unittest.main()
