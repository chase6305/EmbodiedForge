"""Shared feedforward construction and explicit PPO exploration bounds."""

from itertools import pairwise

import torch
from torch import nn


def bounded_log_std(value: torch.Tensor) -> torch.Tensor:
    """Clamp to [-5, 2], retaining unit gradient at both endpoints.

    PyTorch 2.14 uses zero clamp gradients at the endpoints. Projection after an
    optimizer step can land exactly there, so explicit strict comparisons keep
    exploration trainable while preserving earlier PyTorch forward/grad values.
    Comparisons also preserve NaN rather than silently replacing it with a bound.
    """
    return torch.where(value < -5.0, -5.0, torch.where(value > 2.0, 2.0, value))


def mlp(widths, activation):
    """Build hidden activations in linear-layer order, preserving state_dict keys."""
    layers = []
    for index, (before, after) in enumerate(pairwise(widths)):
        layers.append(nn.Linear(before, after))
        if index < len(widths) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)
