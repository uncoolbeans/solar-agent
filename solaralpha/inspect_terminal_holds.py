"""Inspect offline oracle hold-and-import intervals at a fixed battery SOC.

Example: python -m solaralpha.inspect_terminal_holds --hours 6 --value 0.10
"""

import argparse
from datetime import datetime

from solaralpha.decision_time import decision_view
from solaralpha.historical_replay import build_observations
from solaralpha.horizon_sensitivity import horizon_steps, oracle_controller
from solaralpha.solar_sim import Config, battery_transition
from solaralpha.spike_sensitivity import cap_spot_prices
from solaralpha.terminal_sensitivity import (RESULT_PATH, load_results,
                                             result_key)


def hold_examples(config, episode, trace, hours, comparison_values=(0.0, 0.10, 0.40)):
    """Describe hold decisions with a feasible load-covering alternative."""
    by_time = {obs.time: index for index, obs in enumerate(episode)}
    steps = horizon_steps(hours, config)
    examples = []
    for point in trace:
        if (point["import_kwh"] <= 1e-9 or abs(point["power_kw"]) > 1e-9
                or point["soc_before"] <= config.reserve + 0.05):
            continue
        index = by_time[datetime.fromisoformat(point["time"])]
        obs = episode[index]
        balance_request = obs.solar_kw - obs.load_kw
        _, balance_info = battery_transition(
            config, point["soc_before"] * config.capacity_kwh,
            obs, balance_request)
        if balance_info["import_kwh"] > 1e-8:
            continue
        future = episode[index:min(len(episode), index + steps)]
        peak = max(future, key=lambda item: item.sell_price)
        examples.append({"index": index, "time": point["time"],
                         "soc_before": point["soc_before"],
                         "import_kwh": point["import_kwh"],
                         "buy_price": obs.buy_price,
                         "sell_price": obs.sell_price,
                         "balance_request_kw": balance_request,
                         "balance_power_kw": balance_info["power_kw"],
                         "peak_time": peak.time.isoformat(),
                         "peak_sell_price": peak.sell_price})
    controllers = {value: oracle_controller(episode, steps, value)
                   for value in comparison_values}
    for example in examples:
        index = example["index"]
        view = decision_view(episode[index].time, example["soc_before"],
                             episode[index - 1] if index else None, config)
        example["same_soc_actions_kw"] = {
            f"{value:.2f}": controller(view, config, index)
            for value, controller in controllers.items()
        }
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("observed", "capped"),
                        default="observed")
    parser.add_argument("--hours", type=int, choices=(1, 6, 12), default=1)
    parser.add_argument("--value", type=float, default=0.40,
                        help="planner value whose hold decisions are inspected")
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    config = Config()
    observed, prices_mwh = build_observations(config)
    episode = (observed if args.scenario == "observed"
               else cap_spot_prices(observed, prices_mwh, 1.10))
    results = load_results(RESULT_PATH)
    key = result_key(args.scenario, args.hours, args.value)
    entry = results.get(key)
    if entry is None or entry["trace"] is None:
        parser.error(f"No saved trace for {key}")
    examples = hold_examples(config, episode, entry["trace"], args.hours)
    examples.sort(key=lambda item: item["peak_sell_price"], reverse=True)
    print(f"{key}: {len(examples)} day-3 intervals imported while holding above "
          "reserve despite a feasible load-covering battery action")
    for item in examples[:args.limit]:
        print(f"{item['time']} SOC {100 * item['soc_before']:.1f}% "
              f"import {item['import_kwh']:.3f} kWh; "
              f"buy/sell ${item['buy_price']:.3f}/${item['sell_price']:.3f}; "
              f"feasible balance {item['balance_power_kw']:.2f} kW; "
              f"future peak ${item['peak_sell_price']:.3f} at {item['peak_time']}; "
              f"same-SOC actions {item['same_soc_actions_kw']}")


if __name__ == "__main__":
    main()
