"""Build the offline feature dataset, score simple baselines and make plots."""

import argparse
import json

from .baselines import evaluate_baselines
from .features import build_feature_dataset, load_training_validation_points
from .plots import make_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    points = load_training_validation_points()
    report = build_feature_dataset(points)
    print(json.dumps(report["counts"], indent=2))
    if args.features_only:
        return
    metrics = evaluate_baselines()
    print(json.dumps({"training_median": metrics["fitted_training_median_aud_per_kwh"],
                      "models": metrics["models"]}, indent=2))
    plots = make_plots()
    print("Plots:", *(str(p) for p in [plots["prediction_vs_actual"],
                                    plots["error_distributions"], *plots["daily"]]), sep="\n  ")


if __name__ == "__main__":
    main()
