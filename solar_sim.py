"""SolarAlpha Phase 1: synthetic, reproducible battery trading simulator.
Python 3.10+, standard library only. No hardware or network access.
All powers are AC-side kW; prices are AUD/kWh; energy is kWh.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp, pi, sin
from random import Random


@dataclass(frozen=True)
class Config:
    solar_kw: float = 10.0
    capacity_kwh: float = 30.0
    charge_kw: float = 5.0
    discharge_kw: float = 5.0
    export_kw: float = 5.0
    reserve: float = 0.20
    initial_soc: float = 0.50
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    wear_per_kwh: float = 0.05  # per AC kWh discharged; illustrative
    dt: float = 5 / 60


@dataclass(frozen=True)
class Observation:
    time: datetime
    solar_kw: float
    load_kw: float
    buy_price: float
    sell_price: float


def bump(hour, center, width):
    return exp(-0.5 * ((hour - center) / width) ** 2)


def make_data(config, days=14, seed=42):
    """Synthetic summer profiles, not measured prices or weather."""
    rng = Random(seed)
    # January has no Sydney DST transition. Use UTC for later year-round datasets.
    start = datetime(2027, 1, 1, tzinfo=timezone(timedelta(hours=11), 'AEDT'))
    minutes = round(config.dt * 60)
    n = round(24 / config.dt)
    records = []
    noise = 0.0
    cloud = 1.0
    for day in range(days):
        sunshine = rng.uniform(0.40, 1.0)
        hot = rng.random() < 0.40
        spike = rng.random() < 0.18
        spike_hour = rng.uniform(17.0, 19.5)
        spike_height = rng.uniform(0.7, 2.0) if spike else 0.0
        for i in range(n):
            h = i * config.dt
            t = start + timedelta(days=day, minutes=i * minutes)
            cloud = max(0.60, min(1.0, 0.85 * cloud + 0.15 * rng.uniform(0.60, 1.0)))
            sunlight = max(0.0, sin(pi * (h - 5.75) / 14.25)) if 5.75 < h < 20 else 0.0
            solar = config.solar_kw * 0.83 * sunshine * cloud * sunlight ** 1.5
            load = max(0.15, 0.52 + 1.0 * bump(h, 7.5, 1.15)
                       + 1.05 * bump(h, 19, 1.8)
                       + (0.80 if hot else 0.18) * bump(h, 15.5, 3.0)
                       + rng.gauss(0, 0.10))
            noise = 0.85 * noise + rng.gauss(0, 0.009)
            wholesale = (0.055 + 0.065 * bump(h, 7.5, 1.3)
                         - (0.09 + 0.08 * sunshine) * bump(h, 12.5, 2.4)
                         + (0.19 if hot else 0.105) * bump(h, 18.5, 1.6)
                         + spike_height * bump(h, spike_hour, 0.30) + noise)
            # Hypothetical wholesale-linked retail tariffs, NOT a retailer quote.
            records.append(Observation(t, solar, load, wholesale + 0.17, wholesale - 0.01))
    return records


class BatteryEnv:
    """Action: requested battery AC power (positive charge, negative discharge)."""
    def __init__(self, config, data):
        self.c, self.data = config, data
        self.reset()

    def reset(self):
        self.i = 0
        self.energy = self.c.capacity_kwh * self.c.initial_soc
        return self.data[0], self.energy / self.c.capacity_kwh

    def step(self, action_kw):
        c = self.c
        if self.i >= len(self.data):
            raise RuntimeError('Episode ended; call reset().')
        obs = self.data[self.i]
        surplus = obs.solar_kw - obs.load_kw
        reserve_kwh = c.capacity_kwh * c.reserve
        max_charge = min(c.charge_kw, max(0.0, (c.capacity_kwh - self.energy) / (c.eta_charge * c.dt)))
        max_discharge = min(c.discharge_kw, max(0.0, (self.energy - reserve_kwh) * c.eta_discharge / c.dt),
                            max(0.0, c.export_kw - surplus))
        power = max(-max_discharge, min(max_charge, float(action_kw)))
        charge, discharge = max(power, 0.0), max(-power, 0.0)
        self.energy += (charge * c.eta_charge - discharge / c.eta_discharge) * c.dt
        net_grid_kw = obs.load_kw - obs.solar_kw + power
        imports = max(0.0, net_grid_kw) * c.dt
        exports = min(c.export_kw, max(0.0, -net_grid_kw)) * c.dt
        curtailed = max(0.0, -net_grid_kw - c.export_kw) * c.dt
        import_cost = imports * obs.buy_price
        export_revenue = exports * obs.sell_price
        wear = discharge * c.dt * c.wear_per_kwh
        reward = export_revenue - import_cost - wear
        assert reserve_kwh - 1e-8 <= self.energy <= c.capacity_kwh + 1e-8
        assert abs(obs.solar_kw * c.dt + imports + discharge * c.dt
                   - obs.load_kw * c.dt - exports - charge * c.dt - curtailed) < 1e-8
        assert not (imports > 0 and exports > 0)
        self.i += 1
        done = self.i == len(self.data)
        next_obs = None if done else (self.data[self.i], self.energy / c.capacity_kwh)
        info = dict(import_kwh=imports, export_kwh=exports, curtailed_kwh=curtailed,
                    import_cost=import_cost, export_revenue=export_revenue, wear=wear,
                    reward=reward, power_kw=power, soc=self.energy / c.capacity_kwh)
        return next_obs, reward, done, info


def self_consumption(obs, soc, c):
    return obs.solar_kw - obs.load_kw


def threshold_trader(obs, soc, c):
    # Fixed rules; no learning or future data. Thresholds are illustrative.
    if obs.sell_price >= 0.30:
        return -c.discharge_kw
    if obs.buy_price <= 0.08 and soc < 0.85:
        return c.charge_kw
    return self_consumption(obs, soc, c)


def evaluate(policy, c, data):
    env = BatteryEnv(c, data)
    obs, soc = env.reset()
    totals = {k: 0.0 for k in ('import_kwh', 'export_kwh', 'curtailed_kwh', 'import_cost', 'export_revenue', 'wear', 'reward')}
    while True:
        next_obs, reward, done, info = env.step(policy(obs, soc, c))
        for key in totals:
            totals[key] += info[key]
        if done:
            break
        obs, soc = next_obs
    totals['final_soc'] = env.energy / c.capacity_kwh
    # Approximate terminal inventory value to avoid rewarding an empty battery.
    # Compare sensitivity to this assumed value in later experiments.
    totals['adjusted_reward'] = totals['reward'] + (env.energy - c.initial_soc * c.capacity_kwh) * c.eta_discharge * 0.20
    return totals


def main():
    c = Config()
    data = make_data(c)
    print(f'Synthetic Sydney summer: {len(data)} intervals, {len(data) * c.dt / 24:.0f} days')
    print(f'Average daily solar: {sum(x.solar_kw for x in data) * c.dt / 14:.1f} kWh')
    print(f'Average daily load:  {sum(x.load_kw for x in data) * c.dt / 14:.1f} kWh')
    for policy in (self_consumption, threshold_trader):
        m = evaluate(policy, c, data)
        print(f'\n{policy.__name__}')
        print(f'  Grid imports:     {m["import_kwh"]:8.2f} kWh')
        print(f'  Grid exports:     {m["export_kwh"]:8.2f} kWh')
        print(f'  Import cost:     ${m["import_cost"]:8.2f}')
        print(f'  Export revenue:  ${m["export_revenue"]:8.2f}')
        print(f'  Battery wear:    ${m["wear"]:8.2f}')
        print(f'  Net reward:      ${m["reward"]:8.2f}')
        print(f'  Final SOC:        {100*m["final_soc"]:8.2f}%')
        print(f'  Adjusted reward: ${m["adjusted_reward"]:8.2f}')


if __name__ == '__main__':
    main()
