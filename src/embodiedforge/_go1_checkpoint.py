"""One checkpoint contract for Go1 resume, evaluation and live inference."""

import hashlib
import io
import math
from pathlib import Path

_PROFILES = {
    "reward_profile": "original",
    "command_profile": "original",
    "task_semantics": "upstream-v1",
    "learning_rate_override": None,
}


def load_go1_checkpoint(path, *, sha256, contract):
    import torch

    from ._wuji_recipe import finite_tensors

    # Hash the same bytes that torch deserializes, even if the file is replaced.
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError("Checkpoint SHA256 mismatch in policy worker")
    checkpoint = torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Go1 checkpoint must be a dictionary")
    finite_tensors(checkpoint)
    if checkpoint.get("recipe", "go1-joystick") != "go1-joystick":
        raise ValueError("Checkpoint recipe differs from Go1")
    for name, default in _PROFILES.items():
        if checkpoint.get(name, default) != contract.get(name, default):
            raise ValueError(f"Checkpoint {name} differs from the managed run")
    iteration = checkpoint.get("iteration")
    expected = contract.get("checkpoint_iteration")
    if (
        type(iteration) is not int
        or iteration < 0
        or type(expected) is not int
        or iteration != expected
    ):
        raise ValueError("Checkpoint iteration differs from the managed run")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Go1 checkpoint is missing model_state_dict")
    mean, var, count = (state.get(name) for name in ("mean", "var", "count"))
    if (
        any(
            not isinstance(v, torch.Tensor) or not v.is_floating_point()
            for v in (mean, var, count)
        )
        or mean.ndim != 1
        or mean.numel() == 0
        or var.shape != mean.shape
        or count.ndim != 0
        or (var < 0).any()
        or count <= 0
    ):
        raise ValueError("Invalid policy normalization statistics")
    return checkpoint


def load_recipe_checkpoint(request):
    contract = {name: request["input_" + name] for name in _PROFILES}
    contract["checkpoint_iteration"] = request["input_checkpoint_iteration"]
    return load_go1_checkpoint(
        request["checkpoint"],
        sha256=request["input_checkpoint_sha256"],
        contract=contract,
    )


def validate_go1_optimizer(optimizer, state):
    """Validate complete Adam state against the parameters it will update."""
    import torch

    if not isinstance(state, dict) or not isinstance(state.get("state"), dict):
        raise ValueError("Invalid Go1 optimizer state")
    groups = state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(optimizer.param_groups):
        raise ValueError("Go1 optimizer parameter groups differ")
    parameters = {}
    for saved, live in zip(groups, optimizer.param_groups, strict=True):
        ids = saved.get("params") if isinstance(saved, dict) else None
        if not isinstance(ids, list) or len(ids) != len(live["params"]):
            raise ValueError("Go1 optimizer parameter count differs")
        for name in ("lr", "eps", "weight_decay"):
            value = saved.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid Go1 optimizer {name}")
        betas = saved.get("betas")
        if (
            not isinstance(betas, (tuple, list))
            or len(betas) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v < 1
                for v in betas
            )
        ):
            raise ValueError("Invalid Go1 optimizer betas")
        # Training uses ordinary CPU Adam; loading flags must not silently select
        # a different optimizer algorithm or require CUDA graph capture.
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
                raise ValueError(f"Unsupported Go1 optimizer option: {name}")
        for key, parameter in zip(ids, live["params"], strict=True):
            if type(key) is not int or key in parameters:
                raise ValueError("Invalid or duplicate Go1 optimizer parameter ID")
            parameters[key] = parameter
    slots = state["state"]
    if any(type(key) is not int for key in slots) or set(slots) != set(parameters):
        raise ValueError("Go1 optimizer state is incomplete or has extra parameters")
    for key, parameter in parameters.items():
        values = slots[key]
        if not isinstance(values, dict) or set(values) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError("Invalid Go1 optimizer Adam buffers")
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
            raise ValueError("Invalid Go1 optimizer step")
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
                raise ValueError(f"Invalid Go1 optimizer {name} for parameter {key}")


def restore_go1_optimizer(optimizer, state):
    validate_go1_optimizer(optimizer, state)
    optimizer.load_state_dict(state)


def save_go1_checkpoint(path, checkpoint, *, optimizer):
    """Publish only validated state; leave the last checkpoint intact on failure."""
    import torch

    from ._wuji_recipe import finite_tensors

    validate_go1_optimizer(optimizer, checkpoint.get("optimizer_state_dict"))
    finite_tensors(checkpoint)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
