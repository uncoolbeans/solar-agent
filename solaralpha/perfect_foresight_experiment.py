"""Offline observed/capped replay against a whole-episode HiGHS benchmark."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from solaralpha.decision_time import (last_observed_self_consumption,
                                      last_observed_threshold)
from solaralpha.historical_replay import build_observations
from solaralpha.horizon_sensitivity import horizon_steps, oracle_controller
from solaralpha.mpc_agent import mpc_action
from solaralpha.oracle_diagnostic import run_episode
from solaralpha.perfect_foresight import optimize_episode
from solaralpha.solar_sim import Config
from solaralpha.spike_sensitivity import cap_spot_prices
from solaralpha.terminal_sensitivity import (EVALUATION_TERMINAL_VALUE,
                                             load_checkpoint, result_key)


ROOT = Path(__file__).resolve().parent.parent
RESULT_PATH = ROOT / "data" / "derived" / "perfect_foresight_results.json"
POLICIES = (
    ("decision_self", "Decision self", "#2563a6"),
    ("decision_threshold", "Decision threshold", "#198754"),
    ("persistence_mpc", "Persistence MPC", "#d18a00"),
    ("oracle_12h_020", "12 h oracle, $0.20", "#7555b2"),
    ("oracle_12h_010", "12 h oracle, $0.10", "#c34848"),
    ("whole_episode", "Whole episode", "#202b40"),
)


def _usable_checkpoint(config, scenarios, plot_day):
    try:
        return load_checkpoint(config, scenarios, plot_day=plot_day)
    except ValueError as exc:
        # The prior file stores only one selected-day trace. A different day
        # can be recomputed without relaxing config or episode hash checks.
        if str(exc) == "Checkpoint setting changed: plot_day":
            return {"results": {}}
        raise


def gaps_to_benchmark(metrics, threshold_adjusted, optimum):
    """Adjusted-dollar gap and share of improvement beyond threshold."""
    denominator = optimum - threshold_adjusted
    if denominator <= 0:
        raise ValueError("Benchmark must exceed the threshold reference")
    gap = optimum - metrics["adjusted_reward"]
    if abs(gap) < 1e-8:
        gap = 0.0
    return {"remaining_gap_aud": gap,
            "opportunity_capture_pct": 100 *
            (metrics["adjusted_reward"] - threshold_adjusted) / denominator}


def _trace_from_rows(rows, config, day):
    count = round(24 / config.dt)
    start, end = (day - 1) * count, day * count
    if len(rows[start:end]) != count:
        raise ValueError("Selected plot day is incomplete")
    return [{"soc": info["soc"], "power_kw": info["power_kw"],
             "import_kw": info["import_kwh"] / config.dt,
             "export_kw": info["export_kwh"] / config.dt, "reward": reward}
            for _, info, reward in rows[start:end]]


def _trace_from_checkpoint(trace, config):
    if len(trace) != round(24 / config.dt):
        raise ValueError("Cached oracle trace is incomplete")
    return [{"soc": item["soc_after"], "power_kw": item["power_kw"],
             "import_kw": item["import_kwh"] / config.dt,
             "export_kw": item["export_kwh"] / config.dt,
             "reward": item["reward"]} for item in trace]


def run_experiment(config, scenarios, *, plot_day=3, time_limit_s=300.0,
                   checkpoint=None, on_result=None):
    """Evaluate ordinary agents, finite-horizon oracles and offline MILP."""
    if checkpoint is None:
        checkpoint = _usable_checkpoint(config, scenarios, plot_day)
    results = {}
    for scenario, episode in scenarios.items():
        entries = {}

        def put(name, metrics, trace, source, solver=None):
            entries[name] = {"metrics": metrics, "trace": trace, "source": source}
            if solver is not None:
                entries[name]["solver"] = solver
            if on_result:
                on_result(scenario, name, entries[name])

        for name, policy in (("decision_self", last_observed_self_consumption),
                             ("decision_threshold", last_observed_threshold)):
            metrics, rows = run_episode(config, episode,
                                        lambda view, c, index, p=policy: p(view, c))
            put(name, metrics, _trace_from_rows(rows, config, plot_day), "computed")
        metrics, rows = run_episode(config, episode,
                                    lambda view, c, index: mpc_action(view, c))
        put("persistence_mpc", metrics, _trace_from_rows(rows, config, plot_day),
            "computed")

        # The $0.20 reference was rounded in the earlier horizon checkpoint;
        # recompute it here to get exact metrics and the same selected-day trace.
        metrics, rows = run_episode(config, episode,
                                    oracle_controller(episode, horizon_steps(12, config),
                                                      0.20))
        put("oracle_12h_020", metrics, _trace_from_rows(rows, config, plot_day),
            "computed")

        # Reuse exact earlier results when present. A fresh checkout without
        # the derived checkpoint remains independently reproducible.
        prior = checkpoint["results"].get(result_key(scenario, 12, 0.10))
        if prior is not None and prior["source"] == "computed" and prior["trace"] is not None:
            put("oracle_12h_010", prior["metrics"],
                _trace_from_checkpoint(prior["trace"], config), "prior_exact")
        else:
            metrics, rows = run_episode(
                config, episode,
                oracle_controller(episode, horizon_steps(12, config), 0.10))
            put("oracle_12h_010", metrics,
                _trace_from_rows(rows, config, plot_day), "computed")

        optimum = optimize_episode(config, episode, time_limit_s=time_limit_s)
        solver = {"status": optimum.status, "version": optimum.solver_version,
                  "incumbent_objective_aud": optimum.objective,
                  "objective_bound_aud": optimum.objective_bound,
                  "relative_gap": optimum.relative_gap,
                  "solver_runtime_s": optimum.solver_runtime_s,
                  "model_runtime_s": optimum.model_runtime_s,
                  "total_runtime_s": optimum.total_runtime_s,
                  "max_replay_error": optimum.max_replay_error,
                  "replay_agrees": optimum.replay_agrees,
                  "proven_optimal": optimum.proven_optimal}
        put("whole_episode", optimum.metrics,
            _trace_from_rows(optimum.rows, config, plot_day), "computed", solver)

        if optimum.proven_optimal:
            threshold = entries["decision_threshold"]["metrics"]["adjusted_reward"]
            for entry in entries.values():
                entry.update(gaps_to_benchmark(entry["metrics"], threshold,
                                               optimum.objective))
        results[scenario] = entries
    return results


def save_results(config, scenarios, results, plot_day, path=RESULT_PATH):
    from solaralpha.terminal_sensitivity import episode_hash
    payload = {"metadata": {
        "config_sha256": sha256(repr(config).encode()).hexdigest(),
        "episode_sha256": {name: episode_hash(episode)
                           for name, episode in scenarios.items()},
        "evaluation_terminal_value_per_kwh": EVALUATION_TERMINAL_VALUE,
        "plot_day": plot_day,
        "tariff": "illustrative_spot_linked",
        "capped_wholesale_aud_kwh": 1.10,
        "units": "money AUD; energy kWh; battery action AC kW; SOC fraction",
    }, "results": results}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def print_result(scenario, name, entry):
    m = entry["metrics"]
    print(f"{scenario:<8} {name:<17} cash {m['cash_flow']:7.2f} "
          f"net {m['reward']:7.2f} adj {m['adjusted_reward']:7.2f} "
          f"import {m['import_kwh']:6.2f} export {m['export_kwh']:7.2f} "
          f"curtail {m['curtailed_kwh']:5.2f} through {m['throughput_kwh']:7.2f} "
          f"wear {m['wear']:5.2f} SOC {100*m['final_soc']:5.1f}/"
          f"{100*m['min_soc']:5.1f}% "
          f"runtime {entry.get('solver', {}).get('total_runtime_s', m['simulation_runtime_s']):7.2f} s",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-day", type=int, choices=range(1, 8), default=3,
                        metavar="1-7")
    parser.add_argument("--time-limit", type=float, default=300.0,
                        help="HiGHS limit in seconds per scenario (default 300)")
    args = parser.parse_args()
    config = Config()
    observed, prices = build_observations(config)
    scenarios = {"observed": observed,
                 "capped": cap_spot_prices(observed, prices, 1.10)}
    print("Offline whole-episode perfect-foresight MILP; 2016 five-minute steps.",
          flush=True)
    print("$0.20/kWh evaluation terminal adjustment; exact prior $0.10 MPC "
          "reused after input hash checks when available.", flush=True)
    checkpoint = _usable_checkpoint(config, scenarios, args.plot_day)
    # Scenario-by-scenario writes preserve a finished result if interrupted.
    all_results = {}
    for name, episode in scenarios.items():
        part = run_experiment(config, {name: episode}, plot_day=args.plot_day,
                              time_limit_s=args.time_limit, checkpoint=checkpoint,
                              on_result=print_result)
        all_results.update(part)
        save_results(config, {key: scenarios[key] for key in all_results},
                     all_results, args.plot_day)
        from solaralpha.plot_perfect_foresight import save_comparison
        print(f"Wrote {save_comparison(name, args.plot_day, episode, part[name], config)}",
              flush=True)
        solver = part[name]["whole_episode"]["solver"]
        print(f"{name} HiGHS {solver['status']}: incumbent "
              f"${solver['incumbent_objective_aud']:.6f}, "
              f"bound ${solver['objective_bound_aud']:.6f}, "
              f"gap {100*solver['relative_gap']:.6g}%, "
              f"solver {solver['solver_runtime_s']:.2f} s, "
              f"replay max error {solver['max_replay_error']:.3g}; "
              f"proven and achievable {solver['proven_optimal']}", flush=True)
        if solver["proven_optimal"]:
            print("Gap AUD / opportunity capture vs decision threshold:")
            for policy, _, _ in POLICIES:
                item = part[name][policy]
                print(f"  {policy:<17} {item['remaining_gap_aud']:7.3f} / "
                      f"{item['opportunity_capture_pct']:7.2f}%")
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
