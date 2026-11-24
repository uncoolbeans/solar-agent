"""Agent-facing observations that contain only information available at decision time.

Run: python -m solaralpha.decision_time
Historical five-minute averages become visible after their interval completes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

from solaralpha.solar_sim import Config, Observation, evaluate, self_consumption


@dataclass(frozen=True)
class DecisionView:
    """Start time and SOC, then last completed five-minute power and prices.

    SOC is a fraction; powers are AC kW; prices are AUD/kWh. The historical
    replay assumes the previous interval's prices are already published.
    """

    time: datetime
    soc: float
    last_interval_end: datetime | None
    last_solar_kw: float | None
    last_load_kw: float | None
    last_buy_price: float | None
    last_sell_price: float | None


def decision_view(time, soc, previous: Observation | None, config: Config):
    """Build a view without exposing the interval about to be settled."""
    if previous is None:
        return DecisionView(time, soc, None, None, None, None, None)
    return DecisionView(time, soc, previous.time + timedelta(hours=config.dt),
                        previous.solar_kw, previous.load_kw,
                        previous.buy_price, previous.sell_price)


def decision_policy_adapter(policy):
    """Wrap policy(view, config) for one episode of the existing environment."""
    previous = None

    def filtered_policy(current, soc, config):
        nonlocal previous
        view = decision_view(current.time, soc, previous, config)
        action = policy(view, config)
        previous = current  # Available to the next decision, after env.step().
        return action

    return filtered_policy


def evaluate_decision_policy(policy, config, observations, terminal_value_per_kwh=0.20):
    """Evaluate a policy without passing it the raw current Observation."""
    return evaluate(decision_policy_adapter(policy), config, observations,
                    terminal_value_per_kwh)


def last_observed_self_consumption(view, config):
    """Use the last completed interval's solar-minus-load power as a forecast."""
    if view.last_solar_kw is None or view.last_load_kw is None:
        return 0.0
    return view.last_solar_kw - view.last_load_kw


def last_observed_threshold(view, config):
    """Use the last completed prices, with a lagged power fallback."""
    if view.last_sell_price is not None and view.last_sell_price >= 0.30:
        return -config.discharge_kw
    if (view.last_buy_price is not None and view.last_buy_price <= 0.08
            and view.soc < 0.85):
        return config.charge_kw
    return last_observed_self_consumption(view, config)


def main():
    from solaralpha.historical_replay import build_observations

    config = Config()
    observations, _ = build_observations(config)
    reference = evaluate(self_consumption, config, observations)
    lagged_self = evaluate_decision_policy(last_observed_self_consumption,
                                           config, observations)
    lagged_threshold = evaluate_decision_policy(last_observed_threshold,
                                                config, observations)
    print("Seven-day historical replay with illustrative spot-linked tariff")
    print("Previous five-minute solar, load, and prices become visible at the next decision.")
    print("The same-interval reference uses recorded current solar/load; it is idealized.")
    print(f"{'Policy':<28} {'Net':>9} {'Adjusted':>10} {'Import':>9} "
          f"{'Export':>9} {'Final SOC':>10}")
    for name, result in (("same-interval self reference", reference),
                         ("decision-time self", lagged_self),
                         ("decision-time threshold", lagged_threshold)):
        print(f"{name:<28} {result['reward']:9.2f} "
              f"{result['adjusted_reward']:10.2f} "
              f"{result['import_kwh']:9.2f} {result['export_kwh']:9.2f} "
              f"{100 * result['final_soc']:9.1f}%")


if __name__ == "__main__":
    main()
