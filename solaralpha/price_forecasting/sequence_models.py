"""Three compact, direct-price PyTorch models for 72 × 41 inputs."""

import torch
from torch import nn
from torch.nn import functional as F

from .features import FEATURE_FIELDS
from .sequence_dataset import WINDOW


INPUT_WIDTH = len(FEATURE_FIELDS)
MODEL_SPECS = {
    "lstm": {"hidden_size": 32, "layers": 1},
    "gru": {"hidden_size": 40, "layers": 1},
    "tcn": {"channels": 24, "kernel_size": 2,
            "dilations": [1, 2, 4, 8, 16, 32, 64]},
}


class RecurrentPrice(nn.Module):
    def __init__(self, cell, hidden_size):
        super().__init__()
        kind = nn.LSTM if cell == "lstm" else nn.GRU
        self.recurrent = kind(INPUT_WIDTH, hidden_size, num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        states, _ = self.recurrent(x)
        return self.output(states[:, -1, :]).squeeze(-1)


class CausalResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dilation, kernel_size):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size,
                              dilation=dilation)
        self.skip = (nn.Identity() if input_channels == output_channels
                     else nn.Conv1d(input_channels, output_channels, 1))

    def forward(self, x):
        causal = self.conv(F.pad(x, (self.left_padding, 0)))
        return F.relu(causal + self.skip(x))


class TemporalConvPrice(nn.Module):
    def __init__(self, channels=24, kernel_size=2, dilations=(1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)
        if self.receptive_field < WINDOW:
            raise ValueError("TCN receptive field cannot cover 72 historical rows")
        blocks = []
        width = INPUT_WIDTH
        for dilation in dilations:
            blocks.append(CausalResidualBlock(width, channels, dilation, kernel_size))
            width = channels
        self.blocks = nn.Sequential(*blocks)
        self.output = nn.Linear(channels, 1)

    def forward_features(self, x):
        return self.blocks(x.transpose(1, 2))

    def forward(self, x):
        return self.output(self.forward_features(x)[:, :, -1]).squeeze(-1)


def make_model(name):
    if name == "lstm":
        return RecurrentPrice("lstm", MODEL_SPECS[name]["hidden_size"])
    if name == "gru":
        return RecurrentPrice("gru", MODEL_SPECS[name]["hidden_size"])
    if name == "tcn":
        return TemporalConvPrice(**MODEL_SPECS[name])
    raise ValueError(name)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)
