"""One checkpoint contract for Go1 resume, evaluation and live inference."""

import hashlib
import io
import logging
from pathlib import Path

from ._adam_checkpoint import validate_adam_checkpoint

_PROFILES = {
    "learner_backend": "native",
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
    # Both native and Light Loco Go1 policies use FP32. Torch load_state_dict
    # otherwise silently casts finite FP64 weights to possibly infinite FP32.
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError(f"Go1 policy requires a float32 tensor: {name}")
    if "log_std" in state:
        std = state["log_std"].exp()
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Invalid Go1 policy action standard deviation")
    return checkpoint


def load_recipe_checkpoint(request):
    contract = {
        name: request["input_" + name]
        for name in _PROFILES
        if name != "learner_backend"
    }
    contract["learner_backend"] = request.get("input_learner_backend", "native")
    contract["checkpoint_iteration"] = request["input_checkpoint_iteration"]
    return load_go1_checkpoint(
        request["checkpoint"],
        sha256=request["input_checkpoint_sha256"],
        contract=contract,
    )


def validate_go1_optimizer(optimizer, state):
    validate_adam_checkpoint(optimizer, state, label="Go1 optimizer")


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
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not remove partial checkpoint %s: %s", temporary, exc
            )
        raise
