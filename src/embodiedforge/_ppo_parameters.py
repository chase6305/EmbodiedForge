"""Explicit PPO exploration bounds independent of clamp's boundary derivative."""

import torch


def bounded_log_std(value: torch.Tensor) -> torch.Tensor:
    """Clamp to [-5, 2], retaining unit gradient at both endpoints.

    PyTorch 2.14 uses zero clamp gradients at the endpoints. Projection after an
    optimizer step can land exactly there, so explicit strict comparisons keep
    exploration trainable while preserving earlier PyTorch forward/grad values.
    Comparisons also preserve NaN rather than silently replacing it with a bound.
    """
    return torch.where(value < -5.0, -5.0, torch.where(value > 2.0, 2.0, value))
