"""Offline all-oracle MPC planning-horizon sensitivity experiment.

Run: python -m solaralpha.horizon_sensitivity
The oracle reads recorded future observations only in this diagnostic path.
Ordinary mpc_action still uses persistence from the filtered DecisionView.
"""

import argparse
from math import ceil
from time import perf_counter

from solaralpha.decision_time import (decision_view, last_observed_self_consumption,
                                      last_observed_threshold)
from solaralpha.historical_replay import build_observations
from solaralpha.mpc_agent import (DEFAULT_ENERGY_BIN_KWH,
                                  DEFAULT_TERMINAL_VALUE, forecast_action)
from solaralpha.oracle_diagnostic import (FORECAST_CASES, _OfflineOracleForecaster,
                                          run_episode)
from solaralpha.solar_sim import Config
from solaralpha.spike_sensitivity import cap_spot_prices


HORIZON_HOURS = (1, 3, 6, 12, 24)
DEFAULT_RUN_HOURS = (1, 3, 6, 12)
BASELINE_POLICIES = (("decision_self", last_observed_self_consumption),
                     ("decision_threshold", last_observed_threshold))
ALL_ORACLE = FORECAST_CASES[-1]


def horizon_steps(hours, config):
    """Convert a wall-clock horizon to an integral number of data intervals."""
    if hours <= 0:
        raise ValueError("Planning horizon must be positive")
    steps = round(hours / config.dt)
    if abs(steps * config.dt - hours) > 1e-9:
        raise ValueError("Planning horizon must be a whole number of intervals")
    return steps


def oracle_controller(episode, horizon):
    """Create an offline-only controller with a fixed planning horizon."""
    def controller(view, config, index):
        provider = _OfflineOracleForecaster(episode, index, ALL_ORACLE)
        return forecast_action(view, config, provider, horizon,
                               DEFAULT_ENERGY_BIN_KWH, DEFAULT_TERMINAL_VALUE)
    return controller


def benchmark_decisions(config, episode, hours, sample_count=32):
    """Time spread-out oracle decisions at fixed 50% SOC, without a backtest."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    horizon = horizon_steps(hours, config)
    indices = sorted(set(round(i * (len(episode) - 1) / max(1, sample_count - 1))
                         for i in range(sample_count)))
    controller = oracle_controller(episode, horizon)
    durations = []
    for index in indices:
        previous = episode[index - 1] if index else None
        view = decision_view(episode[index].time, config.initial_soc,
                             previous, config)
        start = perf_counter()
        controller(view, config, index)
        durations.append(perf_counter() - start)
    ordered = sorted(durations)
    return {"samples": len(durations),
            "mean_ms": 1000 * sum(durations) / len(durations),
            "p95_ms": 1000 * ordered[ceil(0.95 * len(ordered)) - 1],
            "estimated_episode_s": len(episode) * sum(durations) / len(durations)}


def run_horizon_sensitivity(config, scenarios, hours=DEFAULT_RUN_HOURS,
                            on_result=None):
    """Run both scenarios at each horizon, shortest first, without changing MPC."""
    results = {name: {} for name in scenarios}
    for name, episode in scenarios.items():
        for policy_name, policy in BASELINE_POLICIES:
            result = run_episode(
                config, episode,
                lambda view, c, index, policy=policy: policy(view, c),
            )
            results[name][policy_name] = result
            if on_result is not None:
                on_result(name, policy_name, result[0])
    for hour in sorted(set(hours)):
        steps = horizon_steps(hour, config)
        for name, episode in scenarios.items():
            key = f"oracle_{hour}h"
            result = run_episode(config, episode,
                                 oracle_controller(episode, steps))
            results[name][key] = result
            if on_result is not None:
                on_result(name, key, result[0])
    return results


def print_result(scenario, policy, metrics):
    print(f"{scenario:<8} {policy:<18} "
          f"cash ${metrics['cash_flow']:.2f}, net ${metrics['reward']:.2f}, "
          f"adjusted ${metrics['adjusted_reward']:.2f}; "
          f"import {metrics['import_kwh']:.2f}, export {metrics['export_kwh']:.2f}, "
          f"throughput {metrics['throughput_kwh']:.2f} kWh; "
          f"wear ${metrics['wear']:.2f}; "
          f"SOC final {100 * metrics['final_soc']:.1f}%, "
          f"min {100 * metrics['min_soc']:.1f}%; "
          f"decision mean {metrics['decision_mean_ms']:.3f}, "
          f"p95 {metrics['decision_p95_ms']:.3f} ms; "
          f"episode {metrics['simulation_runtime_s']:.3f} s", flush=True)


def print_imports(scenario, results):
    print(f"\n{scenario} import attribution (kWh); charging increment is a subset "
          "of imports during charging:")
    print(f"{'Policy':<18} {'Charge intervals':>17} {'Charging increment':>19} "
          f"{'Hold intervals':>15} {'Discharge intervals':>20} "
          f"{'At reserve':>12} {'Clipped request':>16}")
    for policy, (metrics, _) in results.items():
        print(f"{policy:<18} {metrics['import_while_charging_kwh']:17.2f} "
              f"{metrics['import_due_to_charging_kwh']:19.2f} "
              f"{metrics['import_while_holding_kwh']:15.2f} "
              f"{metrics['import_while_discharging_kwh']:20.2f} "
              f"{metrics['import_at_reserve_kwh']:12.2f} "
              f"{metrics['import_when_discharge_clipped_kwh']:16.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=int, nargs="+", choices=HORIZON_HOURS,
                        default=DEFAULT_RUN_HOURS,
                        help="horizons to run, shortest first (default: 1 3 6 12; "
                             "24 is available but slow)")
    parser.add_argument("--scenario", choices=("observed", "capped", "both"),
                        default="both")
    parser.add_argument("--plot-day", type=int, choices=range(1, 8), default=3,
                        metavar="1-7")
    parser.add_argument("--no-plots", action="store_true",
                        help="benchmark without writing comparison SVGs")
    parser.add_argument("--timing-samples", type=int, default=0, metavar="N",
                        help="time N spread-out decisions at fixed initial SOC only; "
                             "do not run a financial simulation")
    args = parser.parse_args()
    config = Config()
    observed, prices_mwh = build_observations(config)
    capped = cap_spot_prices(observed, prices_mwh, 1.10)
    scenarios = {name: episode for name, episode in
                 (("observed", observed), ("capped", capped))
                 if args.scenario in ("both", name)}
    if args.timing_samples:
        for hour in sorted(set(args.hours)):
            for name, episode in scenarios.items():
                result = benchmark_decisions(config, episode, hour,
                                             args.timing_samples)
                print(f"{name} {hour}h: {result['samples']} sample decisions, "
                      f"mean {result['mean_ms']:.2f} ms, "
                      f"p95 {result['p95_ms']:.2f} ms, "
                      f"estimated 2016-decision episode "
                      f"{result['estimated_episode_s']:.1f} s")
        return
    print("Offline all-oracle MPC horizon sensitivity; seven days, 2016 decisions per policy")
    print(f"Horizon steps: {[(hour, horizon_steps(hour, config)) for hour in sorted(set(args.hours))]}")
    print("The same physical model, action choices, 0.25 kWh bins, and $0.20/kWh "
          "internal and evaluation terminal value apply to every run.")
    print("At the dataset edge, forecast slots past the final interval use persistence.")
    print("Runtime is wall-clock time on this machine; per-decision timing includes "
          "forecast construction and planning.", flush=True)
    results = run_horizon_sensitivity(config, scenarios, args.hours, print_result)
    for name, scenario_results in results.items():
        print_imports(name, scenario_results)
    if not args.no_plots:
        from solaralpha.plot_horizon_sensitivity import save_horizon_comparison

        for name, episode in scenarios.items():
            path = save_horizon_comparison(name, args.plot_day, episode,
                                           results[name], config)
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
