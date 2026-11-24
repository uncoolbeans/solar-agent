"""Compare the default synthetic data with a controlled low-solar stress case.

Only solar output is halved. Load and prices stay identical, so this tests
scarcity without pretending to model real weather or a different tariff.
"""

from dataclasses import replace

from solaralpha.solar_sim import Config, evaluate, make_data, self_consumption, threshold_trader


DAYS = 14
SEED = 42
LOW_SOLAR_FRACTION = 0.5


def main():
    config = Config()
    default = make_data(config, days=DAYS, seed=SEED)
    low_solar = [replace(obs, solar_kw=obs.solar_kw * LOW_SOLAR_FRACTION)
                 for obs in default]

    print(f"Same {DAYS} days and initial SOC for both policies; seed {SEED}.")
    print("Low-solar case changes only solar output; prices and load are unchanged.")
    print(f"{'Scenario':<12} {'Policy':<18} {'Solar/day':>9} {'Import':>9} "
          f"{'Export':>9} {'Cash':>9} {'Wear':>8} {'Net':>9} "
          f"{'Adjusted':>10} {'Min SOC':>9} {'Final SOC':>10}")
    for scenario, data in (("default", default), ("50% solar", low_solar)):
        solar_per_day = sum(obs.solar_kw for obs in data) * config.dt / DAYS
        for policy in (self_consumption, threshold_trader):
            result = evaluate(policy, config, data)
            cash = result["export_revenue"] - result["import_cost"]
            print(f"{scenario:<12} {policy.__name__:<18} {solar_per_day:9.1f} "
                  f"{result['import_kwh']:9.2f} {result['export_kwh']:9.2f} "
                  f"{cash:9.2f} {result['wear']:8.2f} {result['reward']:9.2f} "
                  f"{result['adjusted_reward']:10.2f} {100 * result['min_soc']:8.1f}% "
                  f"{100 * result['final_soc']:9.1f}%")
    print("Energy columns are kWh (solar per day; import/export for all 14 days).")
    print("Money columns are AUD over all 14 days; adjusted net includes terminal SOC value.")


if __name__ == "__main__":
    main()
