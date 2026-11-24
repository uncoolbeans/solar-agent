"""Replay one January 2025 Sydney week with historical prices and weather.

Inputs: AEMO NSW1 wholesale prices, Open-Meteo hourly irradiance and
temperature, temperature-responsive modeled load, and two illustrative tariffs.
Run python -m solaralpha.fetch_historical_sample first to obtain the local raw snapshots.
"""

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from random import Random
from statistics import median

from solaralpha.fetch_historical_sample import PRICE_FILE, WEATHER_FILE
from solaralpha.plot_day import save_chart
from solaralpha.solar_sim import (BatteryEnv, Config, Observation, bump,
                                  evening_export_schedule, evaluate,
                                  self_consumption, threshold_trader)
from solaralpha.tariffs import EXAMPLE_TARIFFS, ILLUSTRATIVE_SPOT_LINKED


AEST = timezone(timedelta(hours=10), "AEST")  # AEMO market time, no DST
AEDT = timezone(timedelta(hours=11), "AEDT")  # Sydney during this January week
UTC = timezone.utc
START_LOCAL = datetime(2025, 1, 13, tzinfo=AEDT)
END_LOCAL = datetime(2025, 1, 20, tzinfo=AEDT)
STEP = timedelta(minutes=5)
DAYS = 7
LOAD_SEED = 42
SOLAR_DERATE = 0.83  # horizontal irradiance to AC solar proxy


def load_wholesale_prices():
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Run python -m solaralpha.fetch_historical_sample first: {PRICE_FILE}")
    start_utc, end_utc = START_LOCAL.astimezone(UTC), END_LOCAL.astimezone(UTC)
    prices = {}
    with PRICE_FILE.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            if row["REGION"] != "NSW1" or row["PERIODTYPE"] != "TRADE":
                continue
            # AEMO's AEST label marks the end of its five-minute interval.
            interval_end = datetime.strptime(row["SETTLEMENTDATE"], "%Y/%m/%d %H:%M:%S")
            interval_start_utc = interval_end.replace(tzinfo=AEST).astimezone(UTC) - STEP
            if start_utc <= interval_start_utc < end_utc:
                if interval_start_utc in prices:
                    raise ValueError(f"Duplicate AEMO interval: {interval_start_utc}")
                prices[interval_start_utc] = float(row["RRP"])  # AUD/MWh
    expected = DAYS * 24 * 12
    if len(prices) != expected:
        raise ValueError(f"Expected {expected} AEMO prices, found {len(prices)}")
    return prices


def load_hourly_weather_field(field):
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Run python -m solaralpha.fetch_historical_sample first: {WEATHER_FILE}")
    payload = json.loads(WEATHER_FILE.read_text(encoding="utf-8"))
    if payload["timezone"] != "GMT" or payload["utc_offset_seconds"] != 0:
        raise ValueError("Expected UTC weather timestamps")
    hours = payload["hourly"]["time"]
    values = payload["hourly"][field]
    if len(hours) != len(values) or any(value is None for value in values):
        raise ValueError(f"Missing hourly {field}")
    return {datetime.fromisoformat(hour).replace(tzinfo=UTC): float(value)
            for hour, value in zip(hours, values)}


def load_hourly_irradiance():
    irradiance = load_hourly_weather_field("shortwave_radiation")  # preceding-hour mean, W/m²
    if any(value < 0 for value in irradiance.values()):
        raise ValueError("Negative hourly irradiance")
    return irradiance


def modeled_load_kw(local_time, temperature_c, noise_kw=0.0):
    """Illustrative household load; cooling rises above 24°C in the afternoon."""
    hour = local_time.hour + local_time.minute / 60
    cooling_kw = max(0.0, temperature_c - 24.0) * 0.25 * bump(hour, 16.0, 4.0)
    return max(0.15, 0.60 + bump(hour, 7.5, 1.15)
               + 1.05 * bump(hour, 19.0, 1.8) + cooling_kw + noise_kw)


def build_observations(config, tariff=ILLUSTRATIVE_SPOT_LINKED):
    prices_mwh = load_wholesale_prices()
    irradiance = load_hourly_irradiance()
    temperature = load_hourly_weather_field("temperature_2m")
    load_rng = Random(LOAD_SEED)
    start_utc = START_LOCAL.astimezone(UTC)
    observations = []
    for index in range(DAYS * 24 * 12):
        interval_start_utc = start_utc + index * STEP
        hour_start = interval_start_utc.replace(minute=0, second=0, microsecond=0)
        hour_end = interval_start_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        if (interval_start_utc not in prices_mwh or hour_end not in irradiance
                or hour_start not in temperature):
            raise ValueError(f"Missing historical input at {interval_start_utc}")
        # The hourly mean is repeated over its twelve five-minute intervals.
        solar_kw = min(config.solar_kw, config.solar_kw * SOLAR_DERATE
                       * irradiance[hour_end] / 1000.0)
        spot_aud_kwh = prices_mwh[interval_start_utc] / 1000.0
        buy_price, sell_price = tariff.prices(spot_aud_kwh)
        local_time = interval_start_utc.astimezone(AEDT)
        load_kw = modeled_load_kw(local_time, temperature[hour_start],
                                  load_rng.gauss(0.0, 0.10))
        observations.append(Observation(
            local_time, solar_kw, load_kw,
            buy_price, sell_price,
        ))
    return observations, prices_mwh


def lagged_threshold_policy():
    """Use the previous completed interval's price as a decision signal."""
    previous_prices = None

    def policy(obs, soc, config):
        nonlocal previous_prices
        if previous_prices is None:
            action = self_consumption(obs, soc, config)
        else:
            known_obs = replace(obs, buy_price=previous_prices[0],
                                sell_price=previous_prices[1])
            action = threshold_trader(known_obs, soc, config)
        previous_prices = (obs.buy_price, obs.sell_price)
        return action

    return policy


def plot_policy_day(config, data, prices_mwh, day, tariff, policy_name, policy):
    env = BatteryEnv(config, data)
    obs, soc = env.reset()
    rows = []
    for index in range(len(data)):
        next_obs, reward, done, info = env.step(policy(obs, soc, config))
        if (day - 1) * 288 <= index < day * 288:
            rows.append((obs, info, reward))
        if done:
            break
        obs, soc = next_obs
    date = rows[0][0].time.date().isoformat()
    output = (Path(__file__).resolve().parent.parent / "plots"
              / f"hybrid_{date}_{tariff.name}_{policy_name}.svg")
    output.parent.mkdir(exist_ok=True)
    spot_prices = [prices_mwh[obs.time.astimezone(UTC)] / 1000.0
                   for obs, _, _ in rows]
    readable_name = policy_name.replace("_", " ")
    save_chart(output, day, LOAD_SEED, readable_name, rows, config,
               title=f"Hybrid replay {date} | {tariff.name} tariff | {readable_name}",
               spot_prices=spot_prices)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-day", type=int, default=1, help="Sydney day 1-7 to plot")
    args = parser.parse_args()
    if not 1 <= args.plot_day <= DAYS:
        parser.error("--plot-day must be between 1 and 7")
    config = Config()
    reference_data, prices_mwh = build_observations(config)
    print(f"Hybrid replay: {START_LOCAL.date()} through {(END_LOCAL - timedelta(days=1)).date()} "
          "Sydney time (2,016 intervals).")
    print(f"AEMO NSW1 spot: {min(prices_mwh.values()):.2f} to {max(prices_mwh.values()):.2f} AUD/MWh")
    selected_date = START_LOCAL.date() + timedelta(days=args.plot_day - 1)
    selected_spot = [prices_mwh[obs.time.astimezone(UTC)] / 1000.0
                     for obs in reference_data if obs.time.date() == selected_date]
    print(f"{selected_date} wholesale spot median ${median(selected_spot):.3f}/kWh; "
          f"peak ${max(selected_spot):.2f}/kWh")
    print(f"Solar proxy: {sum(obs.solar_kw for obs in reference_data) * config.dt / DAYS:.1f} kWh/day; "
          f"temperature-responsive modeled load: "
          f"{sum(obs.load_kw for obs in reference_data) * config.dt / DAYS:.1f} kWh/day")
    temperature = load_hourly_weather_field("temperature_2m")
    print(f"{'Sydney day':<12} {'Max temp (C)':>12} {'Load':>10} {'Solar':>10}")
    for day in range(DAYS):
        records = reference_data[day * 288:(day + 1) * 288]
        temperatures = [temperature[obs.time.astimezone(UTC).replace(minute=0, second=0,
                                                                      microsecond=0)]
                        for obs in records]
        print(f"{records[0].time.date()!s:<12} {max(temperatures):12.1f} "
              f"{sum(obs.load_kw for obs in records) * config.dt:9.1f} kWh "
              f"{sum(obs.solar_kw for obs in records) * config.dt:9.1f} kWh")
    print("Both tariffs are illustrative; money values are AUD for the seven-day episode.")
    print("Fixed: buy $0.30, sell $0.05/kWh. Spot-linked: buy spot + $0.17, sell spot - $0.01/kWh.")
    print("Evening schedule: full discharge request 17:00-20:00 Sydney time; otherwise self-consumption.")
    print(f"{'Tariff':<13} {'Policy':<18} {'Import':>8} {'Export':>8} {'Cash':>9} "
          f"{'Wear':>8} {'Net':>9} {'Adjusted':>10} {'Final SOC':>10}")
    plots = []
    for tariff in EXAMPLE_TARIFFS:
        data, _ = build_observations(config, tariff)
        for name, policy in (("self_consumption", self_consumption),
                             ("lagged_threshold", lagged_threshold_policy()),
                             ("evening_schedule", evening_export_schedule)):
            result = evaluate(policy, config, data)
            cash = result["export_revenue"] - result["import_cost"]
            print(f"{tariff.name:<13} {name:<18} {result['import_kwh']:8.2f} "
                  f"{result['export_kwh']:8.2f} {cash:9.2f} {result['wear']:8.2f} "
                  f"{result['reward']:9.2f} {result['adjusted_reward']:10.2f} "
                  f"{100 * result['final_soc']:9.1f}%")
        for name, policy in (("lagged_threshold", lagged_threshold_policy()),
                             ("evening_schedule", evening_export_schedule)):
            plots.append(plot_policy_day(config, data, prices_mwh, args.plot_day,
                                         tariff, name, policy))
    for plot in plots:
        print(f"Wrote {plot}")


if __name__ == "__main__":
    main()
