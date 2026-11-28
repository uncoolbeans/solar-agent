"""Original compact sequence bodies with ten unrestricted price heads."""

import torch
from torch import nn

from .multi_horizon import HORIZONS
from .sequence_models import MODEL_SPECS, make_model, parameter_count


def make_multi_output_model(name):
    model = make_model(name)
    width = model.output.in_features
    model.output = nn.Linear(width, len(HORIZONS))
    if name == "tcn" and model.receptive_field < 72:
        raise ValueError("TCN receptive field misses input history")
    return model


def equal_horizon_mae(prediction, target):
    if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[1] != len(HORIZONS):
        raise ValueError("Expected aligned (batch, ten horizons) prices")
    return torch.abs(prediction - target).mean(dim=0).mean()


def model_specification():
    result = {}
    for name in MODEL_SPECS:
        model = make_multi_output_model(name)
        result[name] = {"body": MODEL_SPECS[name], "input_shape": [72, 41],
                        "output_shape": [len(HORIZONS)],
                        "trainable_parameters": parameter_count(model),
                        "receptive_field": getattr(model, "receptive_field", None)}
    return result
