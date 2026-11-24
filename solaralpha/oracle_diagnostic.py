"""Offline forecast-oracle experiment for the simulated MPC.

Run: python -m solaralpha.oracle_diagnostic
Future recorded observations are accessed only inside this diagnostic module.
None of its oracle providers belong in a deployable agent or live control path.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import timedelta
from math import ceil
from time import perf_counter

from solaralpha.decision_time import (decision_view,
                                      last_observed_self_consumption,
                                      last_observed_threshold)
from solaralpha.forecasting import persistence_forecast
from solaralpha.historical_replay import build_observations
from solaralpha.mpc_agent import (DEFAULT_ENERGY_BIN_KWH, DEFAULT_HORIZON,
                                  DEFAULT_TERMINAL_VALUE, forecast_action,
                                  mpc_action)
from solaralpha.solar_sim import BatteryEnv, Config
from solaralpha.spike_sensitivity import cap_spot_prices


@dataclass(frozen=True)
class ForecastCase:
    name: str
    oracle_prices: bool = False
    oracle_solar_load: bool = False


FORECAST_CASES = (
    ForecastCase("persistence_mpc"),
    ForecastCase("oracle_prices", oracle_prices=True),
    ForecastCase("oracle_solar_load", oracle_solar_load=True),
    ForecastCase("oracle_all", oracle_prices=True, oracle_solar_load=True),
)


class _OfflineOracleForecaster:
    """Blend recorded future fields with persistence for one offline decision."""

    def __init__(self, episode, index, case):
        self._episode = episode
        self._index = index
        self._case = case

    def forecast(self, view, config, horizon):
        baseline = persistence_forecast(view, config, horizon)
        if not baseline:
            if not (self._case.oracle_prices and self._case.oracle_solar_load):
                return []  # Partial forecasts need a previous reading to fill other fields.
            available = list(self._episode[self._index:self._index + horizon])
            if not available:
                return []
            if available[0].time != view.time:
                raise ValueError("Oracle and decision timestamps are misaligned")
            # A fully informed diagnostic can plan at the first interval.
            # Repeat the final recorded values beyond the episode boundary.
            while len(available) < horizon:
                available.append(replace(
                    available[-1],
                    time=available[-1].time + timedelta(hours=config.dt),
                ))
            return available
        mixed = []
        for offset, predicted in enumerate(baseline):
            future_index = self._index + offset
            if future_index >= len(self._episode):
                mixed.append(predicted)  # Keep the horizon past the episode edge.
                continue
            actual = self._episode[future_index]
            if actual.time != predicted.time:
                raise ValueError("Oracle and forecast timestamps are misaligned")
            mixed.append(replace(
                predicted,
                solar_kw=actual.solar_kw if self._case.oracle_solar_load else predicted.solar_kw,
                load_kw=actual.load_kw if self._case.oracle_solar_load else predicted.load_kw,
                buy_price=actual.buy_price if self._case.oracle_prices else predicted.buy_price,
                sell_price=actual.sell_price if self._case.oracle_prices else predicted.sell_price,
            ))
        return mixed


def run_episode(config, episode, controller, terminal_value_per_kwh=DEFAULT_TERMINAL_VALUE):
    """Run one policy on truth while passing it only a filtered decision view."""
    simulation_start = perf_counter()
    env = BatteryEnv(config, episode)
    current, soc = env.reset()
    previous = None
    rows = []
    totals = {key: 0.0 for key in ("import_kwh", "export_kwh", "curtailed_kwh",
                                   "import_cost", "export_revenue", "wear", "reward")}
    charge_kwh = discharge_kwh = 0.0
    import_charging = import_hold = import_discharge = 0.0
    import_due_to_charging = import_at_reserve = import_constrained = 0.0
    decision_times = []
    min_soc = soc
    for index in range(len(episode)):
        view = decision_view(current.time, soc, previous, config)
        decision_start = perf_counter()
        request = controller(view, config, index)
        decision_times.append(perf_counter() - decision_start)
        next_view, reward, done, info = env.step(request)
        rows.append((current, info, reward))
        for key in totals:
            totals[key] += info[key]
        executed_kw = info["power_kw"]
        charge_kwh += max(executed_kw, 0.0) * config.dt
        discharge_kwh += max(-executed_kw, 0.0) * config.dt
        imported = info["import_kwh"]
        if executed_kw > 1e-10:
            import_charging += imported
            # Charging's incremental grid draw after solar serves household load.
            without_charge = max(0.0, current.load_kw - current.solar_kw) * config.dt
            import_due_to_charging += max(0.0, imported - without_charge)
        elif executed_kw < -1e-10:
            import_discharge += imported
        else:
            import_hold += imported
        if soc <= config.reserve + 1e-10:
            import_at_reserve += imported
        if request < -1e-10 and executed_kw > request + 1e-10:
            import_constrained += imported
        min_soc = min(min_soc, info["soc"])
        previous = current
        if done:
            break
        current, soc = next_view
    totals["cash_flow"] = totals["export_revenue"] - totals["import_cost"]
    totals["charge_kwh"] = charge_kwh
    totals["discharge_kwh"] = discharge_kwh
    totals["throughput_kwh"] = charge_kwh + discharge_kwh
    totals["import_while_charging_kwh"] = import_charging
    totals["import_while_holding_kwh"] = import_hold
    totals["import_while_discharging_kwh"] = import_discharge
    totals["import_due_to_charging_kwh"] = import_due_to_charging
    totals["import_at_reserve_kwh"] = import_at_reserve
    totals["import_when_discharge_clipped_kwh"] = import_constrained
    totals["min_soc"] = min_soc
    totals["final_soc"] = env.energy / config.capacity_kwh
    totals["adjusted_reward"] = (totals["reward"]
                                 + (env.energy - config.initial_soc * config.capacity_kwh)
                                 * config.eta_discharge * terminal_value_per_kwh)
    totals["decision_mean_ms"] = 1000 * sum(decision_times) / len(decision_times)
    totals["decision_p95_ms"] = 1000 * sorted(decision_times)[ceil(0.95 * len(decision_times)) - 1]
    totals["simulation_runtime_s"] = perf_counter() - simulation_start
    return totals, rows


def run_comparison(config, episode, horizon=DEFAULT_HORIZON):
    """Evaluate the same baseline policies and four MPC forecasts on one episode."""
    results = {}
    for name, policy in (("decision_self", last_observed_self_consumption),
                         ("decision_threshold", last_observed_threshold)):
        results[name] = run_episode(config, episode,
                                    lambda view, c, index, policy=policy: policy(view, c))
    for case in FORECAST_CASES:
        if case == FORECAST_CASES[0]:
            controller = lambda view, c, index: mpc_action(view, c, horizon=horizon)
        else:
            def controller(view, c, index, case=case):
                provider = _OfflineOracleForecaster(episode, index, case)
                return forecast_action(view, c, provider, horizon,
                                       DEFAULT_ENERGY_BIN_KWH, DEFAULT_TERMINAL_VALUE)
        results[case.name] = run_episode(config, episode, controller)
    return results


def print_comparison(name, results):
    print(f"\n{name}: AUD and kWh over the seven-day episode")
    print(f"{'Policy':<20} {'Cash':>9} {'Net':>9} {'Adjusted':>10} "
          f"{'Import':>9} {'Export':>9} {'Throughput':>11} {'Wear':>8} {'Final SOC':>10}")
    for policy, (totals, _) in results.items():
        print(f"{policy:<20} {totals['cash_flow']:9.2f} {totals['reward']:9.2f} "
              f"{totals['adjusted_reward']:10.2f} {totals['import_kwh']:9.2f} "
              f"{totals['export_kwh']:9.2f} {totals['throughput_kwh']:11.2f} "
              f"{totals['wear']:8.2f} {100 * totals['final_soc']:9.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="five-minute planning steps (default: 12, one hour)")
    parser.add_argument("--plot-day", type=int, choices=range(1, 8), default=3,
                        metavar="1-7", help="Sydney day to compare in SVG plots (default: 3)")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least one interval")
    config = Config()
    observed, prices_mwh = build_observations(config)
    capped = cap_spot_prices(observed, prices_mwh, 1.10)
    print("Offline oracle forecast diagnostic; illustrative spot-linked tariff")
    print("All policies start at 50% SOC with the same battery physics and $0.20/kWh terminal value.")
    print(f"MPC: {args.horizon} five-minute steps, 0.25 kWh energy bins, unchanged candidate actions.")
    print("Throughput = executed AC charge + discharge kWh; wear = discharged AC kWh x $0.05.")
    print("Oracle fields use recorded future values only inside this diagnostic.")
    for name, episode in (("observed", observed), ("capped", capped)):
        results = run_comparison(config, episode, args.horizon)
        print_comparison(name, results)
        from solaralpha.plot_oracle_comparison import save_comparison
        path = save_comparison(name, args.plot_day, episode, results, config,
                               args.horizon)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
