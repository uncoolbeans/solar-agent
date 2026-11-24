"""Check how one extreme wholesale spike affects the historical comparison.

Run: python -m solaralpha.spike_sensitivity
The capped prices are a counterfactual; source snapshots are never modified.
"""

import argparse
from dataclasses import replace
from math import isfinite

from solaralpha.historical_replay import UTC, build_observations, lagged_threshold_policy
from solaralpha.solar_sim import (Config, evening_export_schedule, evaluate,
                                  self_consumption)
from solaralpha.tariffs import ILLUSTRATIVE_SPOT_LINKED


DEFAULT_CAP_AUD_KWH = 1.10


def cap_spot_prices(observations, prices_mwh, cap_aud_kwh):
    """Reprice the same physical observations after capping wholesale spot."""
    if not isfinite(cap_aud_kwh) or cap_aud_kwh < 0:
        raise ValueError("Spot cap must be a finite nonnegative AUD/kWh value")
    capped = []
    for obs in observations:
        spot_aud_kwh = prices_mwh[obs.time.astimezone(UTC)] / 1000.0
        buy, sell = ILLUSTRATIVE_SPOT_LINKED.prices(min(spot_aud_kwh, cap_aud_kwh))
        capped.append(replace(obs, buy_price=buy, sell_price=sell))
    return capped


def score_policies(config, observations):
    return {
        "self_consumption": evaluate(self_consumption, config, observations),
        "lagged_threshold": evaluate(lagged_threshold_policy(), config, observations),
        "evening_schedule": evaluate(evening_export_schedule, config, observations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, default=DEFAULT_CAP_AUD_KWH,
                        help="Counterfactual wholesale spot cap in AUD/kWh (default: 1.10)")
    args = parser.parse_args()
    if not isfinite(args.cap) or args.cap < 0:
        parser.error("--cap must be a finite nonnegative AUD/kWh value")

    config = Config()
    observed, prices_mwh = build_observations(config)
    capped = cap_spot_prices(observed, prices_mwh, args.cap)
    affected = [(obs.time, prices_mwh[obs.time.astimezone(UTC)] / 1000.0)
                for obs in observed if prices_mwh[obs.time.astimezone(UTC)] / 1000.0 > args.cap]
    print(f"Seven-day spot-linked tariff sensitivity: {len(observed)} five-minute intervals")
    print(f"Counterfactual cap ${args.cap:.2f}/kWh changes {len(affected)} wholesale intervals;")
    print("both illustrative retail buy and sell prices are recalculated for those intervals.")
    for time, spot in affected:
        print(f"  {time:%Y-%m-%d %H:%M} Sydney: spot ${spot:.2f} -> ${args.cap:.2f}/kWh")
    print("Historical inputs on disk, solar proxy, household load, and initial SOC are unchanged.")
    print("All money values are AUD over seven days; adjusted net uses $0.20/kWh terminal value.")

    results = {}
    for scenario, data in (("observed", observed), ("capped", capped)):
        results[scenario] = score_policies(config, data)
        reference = results[scenario]["self_consumption"]["adjusted_reward"]
        print(f"\n{scenario}: {'Policy':<18} {'Net':>9} {'Adjusted':>10} "
              f"{'vs self':>9} {'Import':>9} {'Export':>9} {'Final SOC':>10}")
        for name, result in results[scenario].items():
            print(f"{'':10}{name:<18} {result['reward']:9.2f} "
                  f"{result['adjusted_reward']:10.2f} "
                  f"{result['adjusted_reward'] - reference:9.2f} "
                  f"{result['import_kwh']:9.2f} {result['export_kwh']:9.2f} "
                  f"{100 * result['final_soc']:9.1f}%")

    physical_keys = ("import_kwh", "export_kwh", "final_soc")
    unchanged = all(all(abs(results["observed"][name][key]
                            - results["capped"][name][key]) < 1e-9
                        for key in physical_keys)
                    for name in results["observed"])
    print(f"\nPhysical dispatch unchanged for all three policies: {unchanged}")
    print("The cap is a stress test, not a historical price correction or household tariff estimate.")


if __name__ == "__main__":
    main()
