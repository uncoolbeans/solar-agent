"""Forecast interface for simulated battery planning.

An ordinary forecast provider receives only the filtered decision-time view.
The offline oracle implementation lives in oracle_diagnostic.py.
"""

from datetime import timedelta
from typing import Protocol

from solaralpha.decision_time import DecisionView
from solaralpha.solar_sim import Config, Observation


class ForecastProvider(Protocol):
    def forecast(self, view: DecisionView, config: Config,
                 horizon: int) -> list[Observation]:
        """Return predicted interval values starting at view.time."""


def persistence_forecast(view, config, horizon=12):
    """Predict each next interval by repeating the last completed readings."""
    if horizon < 1:
        raise ValueError("horizon must be at least one interval")
    values = (view.last_solar_kw, view.last_load_kw,
              view.last_buy_price, view.last_sell_price)
    if any(value is None for value in values):
        return []  # No historical reading exists at the episode's first step.
    solar = min(config.solar_kw, max(0.0, view.last_solar_kw))
    load = max(0.0, view.last_load_kw)
    return [Observation(view.time + timedelta(hours=index * config.dt),
                        solar, load, view.last_buy_price, view.last_sell_price)
            for index in range(horizon)]


class PersistenceForecaster:
    """Deployable-information baseline with no learned parameters."""

    def forecast(self, view, config, horizon):
        return persistence_forecast(view, config, horizon)
