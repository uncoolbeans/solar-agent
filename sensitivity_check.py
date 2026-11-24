"""One-at-a-time sensitivity of the two baselines on synthetic data.

Each policy sees the same observations and starts at the same SOC for a seed.
Values below are probes, not calibrated ranges or real savings estimates.
"""

from dataclasses import dataclass, replace
from statistics import mean

from solar_sim import Config, evaluate, make_data, self_consumption, threshold_trader


DAYS = 14
SEEDS = range(42, 52)


@dataclass(frozen=True)
class Case:
    name: str
    solar_fraction: float = 1.0
    sell_price_shift: float = 0.0  # AUD/kWh
    wear_per_kwh: float = 0.05     # AUD per discharged AC kWh
    terminal_value: float = 0.20  # AUD per future discharged AC kWh


CASES = (
    Case("reference"),
    Case("solar x0.50", solar_fraction=0.50),
    Case("solar x1.25", solar_fraction=1.25),
    Case("sell -$0.10", sell_price_shift=-0.10),
    Case("sell +$0.10", sell_price_shift=0.10),
    Case("wear $0.00", wear_per_kwh=0.00),
    Case("wear $0.10", wear_per_kwh=0.10),
    Case("terminal $0.00", terminal_value=0.00),
    Case("terminal $0.40", terminal_value=0.40),
)


def main():
    default_config = Config()
    data_by_seed = {seed: make_data(default_config, days=DAYS, seed=seed)
                    for seed in SEEDS}
    print(f"{DAYS}-day synthetic episodes; seeds {min(SEEDS)}-{max(SEEDS)}.")
    print("One assumption changes per row; all other values use the reference case.")
    print("Advantage = threshold adjusted reward - self-consumption adjusted reward (AUD).")
    print(f"{'Case':<16} {'Self avg':>10} {'Trader avg':>11} {'Adv avg':>10} "
          f"{'Adv min':>10} {'Adv max':>10} {'Trader wins':>12}")

    for case in CASES:
        config = replace(default_config, wear_per_kwh=case.wear_per_kwh)
        self_scores, trader_scores = [], []
        for base_data in data_by_seed.values():
            if case.solar_fraction != 1.0 or case.sell_price_shift != 0.0:
                data = [replace(obs,
                                solar_kw=obs.solar_kw * case.solar_fraction,
                                sell_price=obs.sell_price + case.sell_price_shift)
                        for obs in base_data]
            else:
                data = base_data
            self_scores.append(evaluate(self_consumption, config, data,
                                        terminal_value_per_kwh=case.terminal_value)["adjusted_reward"])
            trader_scores.append(evaluate(threshold_trader, config, data,
                                          terminal_value_per_kwh=case.terminal_value)["adjusted_reward"])
        advantages = [trader - self_score
                      for self_score, trader in zip(self_scores, trader_scores)]
        print(f"{case.name:<16} {mean(self_scores):10.2f} {mean(trader_scores):11.2f} "
              f"{mean(advantages):10.2f} {min(advantages):10.2f} "
              f"{max(advantages):10.2f} {sum(gap > 0 for gap in advantages):9d}/{len(advantages)}")


if __name__ == "__main__":
    main()
