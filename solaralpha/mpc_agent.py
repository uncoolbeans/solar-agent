"""First simulation-only model predictive battery trading controller.

Run: python -m solaralpha.mpc_agent
Its forecast repeats the last completed five-minute readings; no trained model
or future recorded data is used for decisions.
"""

import argparse
from math import isfinite

from solaralpha.decision_time import (decision_policy_adapter,
                                      evaluate_decision_policy,
                                      last_observed_self_consumption,
                                      last_observed_threshold)
from solaralpha.forecasting import (ForecastProvider, PersistenceForecaster,
                                    persistence_forecast)
from solaralpha.solar_sim import Config, battery_transition


DEFAULT_HORIZON = 12  # one hour at five-minute intervals
DEFAULT_ENERGY_BIN_KWH = 0.25
DEFAULT_TERMINAL_VALUE = 0.20  # AUD per future discharged AC kWh


def forecast_action(view, config, forecaster: ForecastProvider,
                    horizon=DEFAULT_HORIZON,
                    energy_bin_kwh=DEFAULT_ENERGY_BIN_KWH,
                    terminal_value_per_kwh=DEFAULT_TERMINAL_VALUE):
    """Choose the first action from a forecast supplied through one interface.

    The score is forecast export revenue - import cost - discharge wear, plus
    the forecast ending battery energy valued as future AC output. States are
    merged in small battery-energy bins to keep the search inexpensive.
    """
    if not isfinite(energy_bin_kwh) or energy_bin_kwh <= 0:
        raise ValueError("energy_bin_kwh must be positive and finite")
    if not isfinite(terminal_value_per_kwh) or terminal_value_per_kwh < 0:
        raise ValueError("terminal_value_per_kwh must be nonnegative and finite")
    forecast = forecaster.forecast(view, config, horizon)
    if not forecast:
        return 0.0

    energy = view.soc * config.capacity_kwh
    terminal_weight = config.eta_discharge * terminal_value_per_kwh
    # key -> (exact energy within the bin, cumulative reward, first request)
    states = {round(energy / energy_bin_kwh): (energy, 0.0, 0.0)}
    for step, predicted in enumerate(forecast):
        next_states = {}
        surplus = predicted.solar_kw - predicted.load_kw
        requests = tuple(dict.fromkeys((0.0, surplus, -config.discharge_kw,
                                        config.charge_kw)))
        for stored_energy, reward_so_far, first_action in states.values():
            for request in requests:
                next_energy, info = battery_transition(config, stored_energy,
                                                       predicted, request)
                score = reward_so_far + info["reward"]
                first = request if step == 0 else first_action
                key = round(next_energy / energy_bin_kwh)
                old = next_states.get(key)
                if (old is None or score + terminal_weight * next_energy
                        > old[1] + terminal_weight * old[0] + 1e-12):
                    next_states[key] = (next_energy, score, first)
        states = next_states
    best = max(states.values(), key=lambda state: state[1] + terminal_weight * state[0])
    return best[2]


def mpc_action(view, config, horizon=DEFAULT_HORIZON,
               energy_bin_kwh=DEFAULT_ENERGY_BIN_KWH,
               terminal_value_per_kwh=DEFAULT_TERMINAL_VALUE):
    """Ordinary MPC: plan using persistence and no future recorded data."""
    return forecast_action(view, config, PersistenceForecaster(), horizon,
                           energy_bin_kwh, terminal_value_per_kwh)


def main():
    from solaralpha.historical_replay import (build_observations,
                                              plot_policy_day)
    from solaralpha.tariffs import ILLUSTRATIVE_SPOT_LINKED

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="number of five-minute intervals to plan (default: 12)")
    parser.add_argument("--plot-day", type=int, choices=range(1, 8),
                        metavar="1-7", help="also save a Sydney day SVG to plots/")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least one interval")

    config = Config()
    observations, prices_mwh = build_observations(config)
    policy = lambda view, c: mpc_action(view, c, horizon=args.horizon)
    results = (
        ("decision-time self", evaluate_decision_policy(
            last_observed_self_consumption, config, observations)),
        ("decision-time threshold", evaluate_decision_policy(
            last_observed_threshold, config, observations)),
        ("persistence MPC", evaluate_decision_policy(policy, config, observations)),
    )
    print(f"Seven-day historical replay; illustrative spot-linked tariff; {args.horizon}-step horizon")
    print("MPC repeats the last completed solar, load, and prices across its forecast horizon.")
    print("Adjusted net includes $0.20/kWh value for terminal stored energy.")
    print(f"{'Policy':<25} {'Net':>9} {'Adjusted':>10} {'Import':>9} "
          f"{'Export':>9} {'Wear':>8} {'Min SOC':>9} {'Final SOC':>10}")
    for name, result in results:
        print(f"{name:<25} {result['reward']:9.2f} "
              f"{result['adjusted_reward']:10.2f} "
              f"{result['import_kwh']:9.2f} {result['export_kwh']:9.2f} "
              f"{result['wear']:8.2f} {100 * result['min_soc']:8.1f}% "
              f"{100 * result['final_soc']:9.1f}%")
    if args.plot_day is not None:
        path = plot_policy_day(config, observations, prices_mwh, args.plot_day,
                               ILLUSTRATIVE_SPOT_LINKED, "persistence_mpc",
                               decision_policy_adapter(policy))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
