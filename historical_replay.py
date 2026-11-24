"""Replay one January 2025 Sydney week with historical prices and weather.

Inputs: AEMO NSW1 wholesale prices, Open-Meteo hourly irradiance, synthetic
household load, and illustrative wholesale-linked buy/sell price adjustments.
Run fetch_historical_sample.py first to obtain the local raw snapshots.
"""

import argparse
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fetch_historical_sample import PRICE_FILE, WEATHER_FILE
from plot_day import save_chart
from solar_sim import BatteryEnv, Config, Observation, evaluate, make_data, self_consumption, threshold_trader


AEST = timezone(timedelta(hours=10), "AEST")  # AEMO market time, no DST
AEDT = timezone(timedelta(hours=11), "AEDT")  # Sydney during this January week
UTC = timezone.utc
START_LOCAL = datetime(2025, 1, 13, tzinfo=AEDT)
END_LOCAL = datetime(2025, 1, 20, tzinfo=AEDT)
STEP = timedelta(minutes=5)
DAYS = 7
LOAD_SEED = 42
BUY_ADDER = 0.17   # AUD/kWh; illustrative, not a retailer tariff
SELL_ADDER = -0.01  # AUD/kWh; illustrative, not a retailer tariff
SOLAR_DERATE = 0.83  # horizontal irradiance to AC solar proxy


def load_wholesale_prices():
    if not PRICE_FILE.exists():
        raise FileNotFoundError(f"Run python fetch_historical_sample.py first: {PRICE_FILE}")
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


def load_hourly_irradiance():
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Run python fetch_historical_sample.py first: {WEATHER_FILE}")
    payload = json.loads(WEATHER_FILE.read_text(encoding="utf-8"))
    if payload["timezone"] != "GMT" or payload["utc_offset_seconds"] != 0:
        raise ValueError("Expected UTC weather timestamps")
    hours = payload["hourly"]["time"]
    values = payload["hourly"]["shortwave_radiation"]  # preceding-hour mean, W/m²
    if len(hours) != len(values) or any(value is None or value < 0 for value in values):
        raise ValueError("Missing or invalid hourly irradiance")
    return {datetime.fromisoformat(hour).replace(tzinfo=UTC): float(value)
            for hour, value in zip(hours, values)}


def build_observations(config):
    prices_mwh = load_wholesale_prices()
    irradiance = load_hourly_irradiance()
    synthetic_load = make_data(config, days=DAYS, seed=LOAD_SEED, weather_variation=False)
    start_utc = START_LOCAL.astimezone(UTC)
    observations = []
    for index, load_record in enumerate(synthetic_load):
        interval_start_utc = start_utc + index * STEP
        hour_end = interval_start_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        if interval_start_utc not in prices_mwh or hour_end not in irradiance:
            raise ValueError(f"Missing historical input at {interval_start_utc}")
        # The hourly mean is repeated over its twelve five-minute intervals.
        solar_kw = min(config.solar_kw, config.solar_kw * SOLAR_DERATE
                       * irradiance[hour_end] / 1000.0)
        spot_aud_kwh = prices_mwh[interval_start_utc] / 1000.0
        observations.append(Observation(
            interval_start_utc.astimezone(AEDT), solar_kw, load_record.load_kw,
            spot_aud_kwh + BUY_ADDER, spot_aud_kwh + SELL_ADDER,
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


def plot_policy_day(config, data, day):
    env = BatteryEnv(config, data)
    policy = lagged_threshold_policy()
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
    output = Path(__file__).resolve().parent / "plots" / f"hybrid_{date}_lagged_threshold.svg"
    output.parent.mkdir(exist_ok=True)
    save_chart(output, day, LOAD_SEED, "lagged threshold", rows, config,
               title=f"Hybrid replay {date} | AEMO NSW1 + Open-Meteo | lagged threshold")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-day", type=int, default=1, help="Sydney day 1-7 to plot")
    args = parser.parse_args()
    if not 1 <= args.plot_day <= DAYS:
        parser.error("--plot-day must be between 1 and 7")
    config = Config()
    data, prices_mwh = build_observations(config)
    print(f"Hybrid replay: {START_LOCAL.date()} through {(END_LOCAL - timedelta(days=1)).date()} "
          "Sydney time (2,016 intervals).")
    print(f"AEMO NSW1 spot: {min(prices_mwh.values()):.2f} to {max(prices_mwh.values()):.2f} AUD/MWh")
    print(f"Solar proxy: {sum(obs.solar_kw for obs in data) * config.dt / DAYS:.1f} kWh/day; "
          f"synthetic load: {sum(obs.load_kw for obs in data) * config.dt / DAYS:.1f} kWh/day")
    for name, policy in (("self_consumption", self_consumption),
                         ("lagged_threshold", lagged_threshold_policy())):
        result = evaluate(policy, config, data)
        print(f"{name}: import {result['import_kwh']:.2f} kWh, "
              f"export {result['export_kwh']:.2f} kWh, net ${result['reward']:.2f}, "
              f"adjusted ${result['adjusted_reward']:.2f}, "
              f"minimum SOC {100 * result['min_soc']:.1f}%, "
              f"final SOC {100 * result['final_soc']:.1f}%")
    print("Buy/sell prices use illustrative spot-price adjustments; no household tariff is modeled.")
    print(f"Wrote {plot_policy_day(config, data, args.plot_day)}")


if __name__ == "__main__":
    main()
