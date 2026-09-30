"""Shared checks for Adam checkpoint state materialized on CPU."""

import math


def validate_adam_checkpoint(optimizer, state, *, label="Adam optimizer"):
    """Validate trained Adam state on CPU before restoring or publishing it."""
    import torch

    if not isinstance(state, dict) or not isinstance(state.get("state"), dict):
        raise ValueError(f"Invalid {label} state")
    groups = state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError(f"{label} parameter groups differ")
    parameters = {}
    for saved, live in zip(groups, optimizer.param_groups, strict=True):
        ids = saved.get("params") if isinstance(saved, dict) else None
        if not isinstance(ids, list) or len(ids) != len(live["params"]):
            raise ValueError(f"{label} parameter count differs")
        for name in ("lr", "eps", "weight_decay"):
            value = saved.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {label} {name}")
        betas = saved.get("betas")
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
                for v in betas
            )
        ):
            raise ValueError(f"Invalid {label} betas")
        # The reference optimizer defines the supported execution options.
        # Loading must not select a different algorithm or CUDA graph capture.
        for name in (
            "amsgrad",
            "maximize",
            "capturable",
            "differentiable",
            "fused",
            "foreach",
            "decoupled_weight_decay",
        ):
            value, expected = saved.get(name, live.get(name)), live.get(name)
            if type(value) not in (bool, type(None)) or value != expected:
                raise ValueError(f"Unsupported {label} option: {name}")
        for key, parameter in zip(ids, live["params"], strict=True):
            if type(key) is not int or key in parameters:
                raise ValueError(f"Invalid or duplicate {label} parameter ID")
            parameters[key] = parameter
    slots = state["state"]
    if any(type(key) is not int for key in slots) or set(slots) != set(parameters):
        raise ValueError(f"{label} state is incomplete or has extra parameters")
    for key, parameter in parameters.items():
        values = slots[key]
        if not isinstance(values, dict) or set(values) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError(f"Invalid {label} Adam buffers")
        step = values["step"]
        if (
            not isinstance(step, torch.Tensor)
            or not step.is_floating_point()
            or step.ndim != 0
            or step.device.type != "cpu"
            or not torch.isfinite(step)
            or step < 0
            or step != step.floor()
        ):
            raise ValueError(f"Invalid {label} step")
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
                or not torch.isfinite(value).all()
                or (name == "exp_avg_sq" and (value < 0).any())
            ):
                raise ValueError(f"Invalid {label} {name} for parameter {key}")
