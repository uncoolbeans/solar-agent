"""Offline all-oracle MPC internal terminal-value sensitivity experiment.

Run observed sweep first: python -m solaralpha.terminal_sensitivity
Recorded future data is used only by the offline oracle provider. Evaluation
always values the change in terminal stored energy at AUD 0.20/kWh.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from solaralpha.historical_replay import build_observations
from solaralpha.horizon_sensitivity import horizon_steps, oracle_controller
from solaralpha.oracle_diagnostic import run_episode
from solaralpha.solar_sim import Config
from solaralpha.spike_sensitivity import cap_spot_prices


VALUES = (0.00, 0.05, 0.10, 0.20, 0.30, 0.40)
HOURS = (1, 6, 12)
EVALUATION_TERMINAL_VALUE = 0.20
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = PROJECT_ROOT / "data" / "derived" / "horizon_reference.json"
RESULT_PATH = PROJECT_ROOT / "data" / "derived" / "terminal_sensitivity_results.json"


def episode_hash(episode):
    digest = sha256()
    for obs in episode:
        digest.update((f"{obs.time.isoformat()}|{obs.solar_kw!r}|"
                       f"{obs.load_kw!r}|{obs.buy_price!r}|"
                       f"{obs.sell_price!r}\n").encode())
    return digest.hexdigest()


def result_key(scenario, hours, internal_value):
    return f"{scenario}:{hours}:{internal_value:.2f}"


def load_results(path=RESULT_PATH):
    """Read saved diagnostic summaries and selected-day traces."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["results"]


def load_reference(config, scenarios):
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if reference["config_sha256"] != sha256(repr(config).encode()).hexdigest():
        raise ValueError("Prior result battery configuration differs")
    for name, episode in scenarios.items():
        if reference["episode_sha256"][name] != episode_hash(episode):
            raise ValueError(f"Prior result episode differs: {name}")
    if (reference["evaluation_terminal_value_per_kwh"] != EVALUATION_TERMINAL_VALUE
            or reference["internal_terminal_value_per_kwh"] != 0.20):
        raise ValueError("Prior terminal-value settings differ")
    return reference


def load_checkpoint(config, scenarios, path=RESULT_PATH, plot_day=3):
    metadata = {
        "config_sha256": sha256(repr(config).encode()).hexdigest(),
        "episode_sha256": {name: episode_hash(episode)
                           for name, episode in scenarios.items()},
        "evaluation_terminal_value_per_kwh": EVALUATION_TERMINAL_VALUE,
        "plot_day": plot_day,
    }
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        old = saved["metadata"]
        for key in ("config_sha256", "evaluation_terminal_value_per_kwh",
                    "plot_day"):
            if old[key] != metadata[key]:
                raise ValueError(f"Checkpoint setting changed: {key}")
        for name, digest in metadata["episode_sha256"].items():
            if (name in old["episode_sha256"]
                    and old["episode_sha256"][name] != digest):
                raise ValueError(f"Checkpoint episode changed: {name}")
        old["episode_sha256"].update(metadata["episode_sha256"])
        return saved
    return {"metadata": metadata, "results": {}}


def save_checkpoint(checkpoint, path=RESULT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def trace_for_day(config, episode, rows, requests, day=3):
    steps_per_day = round(24 / config.dt)
    start, end = (day - 1) * steps_per_day, day * steps_per_day
    if len(rows[start:end]) != steps_per_day:
        raise ValueError("Incomplete plot day")
    trace = []
    for index in range(start, end):
        obs, info, reward = rows[index]
        prior_soc = (config.initial_soc if index == 0
                     else rows[index - 1][1]["soc"])
        trace.append({"time": obs.time.isoformat(),
                      "soc_before": prior_soc, "soc_after": info["soc"],
                      "requested_power_kw": requests[index],
                      "power_kw": info["power_kw"],
                      "solar_kw": obs.solar_kw, "load_kw": obs.load_kw,
                      "buy_price": obs.buy_price, "sell_price": obs.sell_price,
                      "import_kwh": info["import_kwh"],
                      "export_kwh": info["export_kwh"],
                      "reward": reward})
    return trace


def run_setting(config, episode, hours, internal_value, plot_day=3):
    """Evaluate one setting with fixed evaluation valuation and record a day."""
    controller = oracle_controller(episode, horizon_steps(hours, config),
                                   internal_value)
    requests = []

    def record(view, c, index):
        action = controller(view, c, index)
        requests.append(action)
        return action

    totals, rows = run_episode(config, episode, record,
                               terminal_value_per_kwh=EVALUATION_TERMINAL_VALUE)
    return {"metrics": totals,
            "trace": trace_for_day(config, episode, rows, requests, plot_day),
            "source": "computed"}


def evaluate_settings(config, scenarios, settings, path=RESULT_PATH,
                      reuse_reference=True, on_result=None, plot_day=3):
    reference = load_reference(config, scenarios) if reuse_reference else None
    checkpoint = load_checkpoint(config, scenarios, path, plot_day)
    for scenario, hours, internal_value in sorted(settings,
                                                   key=lambda item: (item[1], item[0], item[2])):
        if scenario not in scenarios or hours not in HOURS or internal_value not in VALUES:
            raise ValueError(f"Unsupported experiment setting: {(scenario, hours, internal_value)}")
        key = result_key(scenario, hours, internal_value)
        if (key in checkpoint["results"]
                and not (not reuse_reference
                         and checkpoint["results"][key]["source"] == "prior_rounded")):
            entry = checkpoint["results"][key]
        elif (reference is not None and internal_value == 0.20
              and f"{scenario}:{hours}" in reference["results"]):
            entry = {"metrics": reference["results"][f"{scenario}:{hours}"],
                     "trace": None, "source": "prior_rounded"}
            checkpoint["results"][key] = entry
            save_checkpoint(checkpoint, path)
        else:
            entry = run_setting(config, scenarios[scenario], hours, internal_value,
                                plot_day)
            checkpoint["results"][key] = entry
            save_checkpoint(checkpoint, path)
        if on_result is not None:
            on_result(key, entry)
    return checkpoint


def print_result(key, entry):
    m = entry["metrics"]
    print(f"{key:<18} {entry['source']:<13} "
          f"cash {m['cash_flow']:7.2f} net {m['reward']:7.2f} "
          f"adjusted {m['adjusted_reward']:7.2f} "
          f"import {m['import_kwh']:6.2f} export {m['export_kwh']:7.2f} "
          f"charge-draw {m['import_due_to_charging_kwh']:6.2f} "
          f"hold-import {m['import_while_holding_kwh']:6.2f} "
          f"throughput {m['throughput_kwh']:7.2f} wear {m['wear']:5.2f} "
          f"SOC {100 * m['final_soc']:5.1f}/{100 * m['min_soc']:5.1f}% "
          f"decision {m['decision_mean_ms']:6.2f} ms "
          f"episode {m['simulation_runtime_s']:7.2f} s", flush=True)


def parse_pairs(values, scenario):
    if not values:
        return [(scenario, hour, value) for hour in HOURS for value in VALUES]
    pairs = []
    for item in values:
        try:
            hour_text, value_text = item.split(":", 1)
            hour, value = int(hour_text), float(value_text)
        except ValueError as error:
            raise ValueError(f"Expected HOUR:VALUE, got {item!r}") from error
        pairs.append((scenario, hour, value))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=("observed", "capped"),
                        default="observed")
    parser.add_argument("--pairs", nargs="*", metavar="HOUR:VALUE",
                        help="specific combinations; default is the full 3-by-6 grid")
    parser.add_argument("--no-reference", action="store_true",
                        help="recompute the prior $0.20 cases rather than reuse rounded summaries")
    parser.add_argument("--plot-day", type=int, choices=range(1, 8), default=3,
                        metavar="1-7", help="day stored for diagnostic plots (default: 3)")
    args = parser.parse_args()
    config = Config()
    observed, prices_mwh = build_observations(config)
    scenarios = {"observed": observed}
    if args.scenario == "capped":
        scenarios = {"capped": cap_spot_prices(observed, prices_mwh, 1.10)}
    try:
        settings = parse_pairs(args.pairs, args.scenario)
        print("Evaluation terminal value fixed at $0.20/kWh; planner value varies.",
              flush=True)
        evaluate_settings(config, scenarios, settings,
                          reuse_reference=not args.no_reference,
                          on_result=print_result, plot_day=args.plot_day)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
