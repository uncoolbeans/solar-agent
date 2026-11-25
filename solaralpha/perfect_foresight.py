"""Offline whole-episode perfect-foresight MILP; never an agent controller.

Run with the optional dependency in requirements-optimizer.txt:
    python -m solaralpha.perfect_foresight

All episode observations, including future prices, are read here only. The
ordinary decision-time and persistence MPC interfaces are unchanged.
"""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

from solaralpha.oracle_diagnostic import run_episode
from solaralpha.solar_sim import Config, Observation


EVALUATION_TERMINAL_VALUE = 0.20
REPLAY_TOLERANCE = 2e-5


@dataclass
class OptimizerResult:
    actions_kw: list[float]
    metrics: dict
    rows: list
    status: str
    objective: float
    objective_bound: float
    relative_gap: float
    solver_runtime_s: float
    model_runtime_s: float
    total_runtime_s: float
    max_replay_error: float
    replay_agrees: bool
    proven_optimal: bool
    solver_version: str


def _check_inputs(config, episode):
    if not episode:
        raise ValueError("An episode must contain at least one observation")
    if (not all(isfinite(x) for x in (config.dt, config.capacity_kwh,
                                     config.charge_kw, config.discharge_kw,
                                     config.export_kw, config.reserve,
                                     config.initial_soc, config.eta_charge,
                                     config.eta_discharge, config.wear_per_kwh))
            or config.dt <= 0 or config.capacity_kwh <= 0 or config.charge_kw < 0
            or config.discharge_kw < 0 or config.export_kw < 0
            or not 0 <= config.reserve <= config.initial_soc <= 1
            or not 0 < config.eta_charge <= 1
            or not 0 < config.eta_discharge <= 1
            or config.wear_per_kwh < 0):
        raise ValueError("Battery parameters must be finite and physically valid")
    for obs in episode:
        if (not all(isfinite(x) for x in (obs.solar_kw, obs.load_kw,
                                          obs.buy_price, obs.sell_price))
                or obs.solar_kw < 0 or obs.load_kw < 0):
            raise ValueError("Episode powers and prices must be finite; powers nonnegative")


def optimize_episode(config: Config, episode: list[Observation], *,
                     time_limit_s=300.0, mip_rel_gap=1e-7) -> OptimizerResult:
    """Solve the exact, continuous-power simulator dispatch as an offline MILP.

    Three binary modes per interval enforce exclusive battery direction, grid
    direction, and forced-only curtailment even with negative tariffs.
    """
    _check_inputs(config, episode)
    if not isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be finite and positive")
    if not isfinite(mip_rel_gap) or mip_rel_gap < 0:
        raise ValueError("mip_rel_gap must be finite and nonnegative")
    try:
        import highspy
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Install requirements-optimizer.txt to run the offline HiGHS benchmark") from exc

    total_start = perf_counter()
    n = len(episode)
    names = ("charge", "discharge", "import", "export", "curtail",
             "energy", "battery_mode", "import_mode", "curtail_mode")
    offsets = {}
    cursor = 0
    for name in names:
        offsets[name] = cursor
        cursor += n + 1 if name == "energy" else n

    def at(name, t):
        return offsets[name] + t

    lower = np.zeros(cursor, dtype=np.float64)
    upper = np.zeros(cursor, dtype=np.float64)
    costs = np.zeros(cursor, dtype=np.float64)
    initial = config.capacity_kwh * config.initial_soc
    reserve = config.capacity_kwh * config.reserve
    lower[offsets["energy"]:offsets["energy"] + n + 1] = reserve
    upper[offsets["energy"]:offsets["energy"] + n + 1] = config.capacity_kwh
    lower[at("energy", 0)] = initial
    upper[at("energy", 0)] = initial
    costs[at("energy", n)] = config.eta_discharge * EVALUATION_TERMINAL_VALUE
    for t, obs in enumerate(episode):
        surplus = obs.solar_kw - obs.load_kw
        upper[at("charge", t)] = config.charge_kw
        # battery_transition caps discharge by the export headroom computed
        # before the requested action, even when solar is being curtailed.
        upper[at("discharge", t)] = min(config.discharge_kw,
                                         max(0.0, config.export_kw - surplus))
        upper[at("import", t)] = obs.load_kw + config.charge_kw
        upper[at("export", t)] = config.export_kw
        upper[at("curtail", t)] = max(0.0, surplus - config.export_kw)
        for name in ("battery_mode", "import_mode", "curtail_mode"):
            upper[at(name, t)] = 1
        costs[at("discharge", t)] = -config.dt * config.wear_per_kwh
        costs[at("import", t)] = -config.dt * obs.buy_price
        costs[at("export", t)] = config.dt * obs.sell_price

    row_lower, row_upper, indices, values, starts = [], [], [], [], [0]

    def row(terms, lo, hi):
        for index, value in terms:
            indices.append(index)
            values.append(value)
        starts.append(len(indices))
        row_lower.append(lo)
        row_upper.append(hi)

    inf = highspy.kHighsInf
    for t, obs in enumerate(episode):
        row(((at("energy", t + 1), 1), (at("energy", t), -1),
             (at("charge", t), -config.eta_charge * config.dt),
             (at("discharge", t), config.dt / config.eta_discharge)), 0, 0)
        row(((at("charge", t), 1),
             (at("battery_mode", t), -config.charge_kw)), -inf, 0)
        row(((at("discharge", t), 1),
             (at("battery_mode", t), upper[at("discharge", t)])),
            -inf, upper[at("discharge", t)])
        # Solar + imports + discharge = load + exports + charge + curtail.
        row(((at("import", t), 1), (at("discharge", t), 1),
             (at("export", t), -1), (at("charge", t), -1),
             (at("curtail", t), -1)), obs.load_kw - obs.solar_kw,
            obs.load_kw - obs.solar_kw)
        row(((at("import", t), 1),
             (at("import_mode", t), -upper[at("import", t)])), -inf, 0)
        row(((at("export", t), 1),
             (at("import_mode", t), config.export_kw)), -inf, config.export_kw)
        row(((at("curtail", t), 1),
             (at("curtail_mode", t), -upper[at("curtail", t)])), -inf, 0)
        # Curtailment is only allowed once exports have reached the cap.
        row(((at("export", t), 1),
             (at("curtail_mode", t), -config.export_kw)), 0, inf)
        row(((at("import_mode", t), 1),
             (at("curtail_mode", t), 1)), -inf, 1)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit_s)
    h.setOptionValue("mip_rel_gap", mip_rel_gap)
    def check(status, operation):
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS {operation} failed: {status}")
    check(h.addVars(cursor, lower, upper), "addVars")
    check(h.changeColsCost(cursor, np.arange(cursor, dtype=np.int32), costs),
          "changeColsCost")
    binary_indices = np.array([at(name, t)
                               for name in ("battery_mode", "import_mode", "curtail_mode")
                               for t in range(n)], dtype=np.int32)
    check(h.changeColsIntegrality(len(binary_indices), binary_indices,
                                  np.ones(len(binary_indices), dtype=np.uint8)),
          "changeColsIntegrality")
    check(h.addRows(len(row_lower), np.array(row_lower), np.array(row_upper),
                    len(indices), np.array(starts, dtype=np.int32),
                    np.array(indices, dtype=np.int32), np.array(values)),
          "addRows")
    check(h.changeObjectiveOffset(-initial * config.eta_discharge
                                  * EVALUATION_TERMINAL_VALUE),
          "changeObjectiveOffset")
    h.setMaximize()
    solve_start = perf_counter()
    check(h.run(), "run")
    solver_runtime_s = perf_counter() - solve_start
    status = str(h.getModelStatus()).split(".")[-1]
    solution = h.getSolution()
    if not solution.value_valid:
        raise RuntimeError(f"HiGHS has no feasible incumbent: {status}")
    v = solution.col_value
    actions = [v[at("charge", t)] - v[at("discharge", t)] for t in range(n)]
    position = 0

    def controller(_view, _config, index):
        nonlocal position
        if index != position:
            raise ValueError("Offline action replay is out of sequence")
        position += 1
        return actions[index]

    metrics, rows = run_episode(config, episode, controller,
                                terminal_value_per_kwh=EVALUATION_TERMINAL_VALUE)
    errors = [abs(metrics["adjusted_reward"] - h.getObjectiveValue())]
    for t, (_, info, _) in enumerate(rows):
        errors.extend((abs(info["power_kw"] - actions[t]),
                       abs(info["import_kwh"] - v[at("import", t)] * config.dt),
                       abs(info["export_kwh"] - v[at("export", t)] * config.dt),
                       abs(info["curtailed_kwh"] - v[at("curtail", t)] * config.dt),
                       abs(info["soc"] * config.capacity_kwh - v[at("energy", t + 1)])))
    max_error = max(errors)
    info = h.getInfo()
    return OptimizerResult(
        actions_kw=actions, metrics=metrics, rows=rows, status=status,
        objective=h.getObjectiveValue(), objective_bound=info.mip_dual_bound,
        relative_gap=info.mip_gap, solver_runtime_s=solver_runtime_s,
        model_runtime_s=solve_start - total_start,
        total_runtime_s=perf_counter() - total_start,
        max_replay_error=max_error, replay_agrees=max_error <= REPLAY_TOLERANCE,
        proven_optimal=(status == "kOptimal" and max_error <= REPLAY_TOLERANCE),
        solver_version=h.version(),
    )


if __name__ == "__main__":
    from solaralpha.perfect_foresight_experiment import main
    main()
