"""Publish and inspect committed Go1 checkpoints independently of run completion."""

import json
import re
import shutil
import tempfile
from pathlib import Path

from .recipes import sha256, write_json

STOPPED = {"interrupted", "timed_out", "failed"}


def publish_training_checkpoint(checkpoint, *, optimizer, request):
    from ._go1_checkpoint import save_go1_checkpoint

    iteration = checkpoint["iteration"]
    directory = Path("checkpoints")
    directory.mkdir(exist_ok=True)
    path = directory / f"model-{iteration:09d}.pt"
    save_go1_checkpoint(path, checkpoint, optimizer=optimizer)
    completed = iteration - request["start_iteration"] + 1
    result = {
        "checkpoint": path.as_posix(),
        "checkpoint_sha256": sha256(path),
        "checkpoint_iteration": iteration,
        "completed_updates": completed,
        "cumulative_updates": request.get("prior_updates", 0) + completed,
        "runtime": json.loads(Path("runtime.json").read_text()),
        "assets": json.loads(Path("assets.json").read_text()),
        **{
            key: checkpoint[key]
            for key in (
                "task_semantics",
                "reward_profile",
                "command_profile",
                "learning_rate_override",
            )
        },
    }
    receipt = {
        "schema": 1,
        "request_sha256": sha256(Path("request.json")),
        "result": result,
    }
    temporary = Path("recovery.json.tmp")
    try:
        write_json(temporary, receipt)
        temporary.replace("recovery.json")
    finally:
        temporary.unlink(missing_ok=True)
    # Preserve the completed-run interface. The receipt already points to a
    # separate immutable file if interruption occurs while updating model.pt.
    temporary = Path("model.pt.tmp")
    try:
        shutil.copyfile(path, temporary)
        temporary.replace("model.pt")
    finally:
        temporary.unlink(missing_ok=True)
    # Keep the newest two snapshots; unpublished files are never recovery input.
    snapshots = sorted(
        (p for p in directory.iterdir() if re.fullmatch(r"model-\d+\.pt", p.name)),
        key=lambda p: int(p.stem.split("-")[-1]),
        reverse=True,
    )
    for old in snapshots[2:]:
        old.unlink()


def recovery_result(run, manifest):
    """Read only a stopped run's committed receipt; no Torch import is needed."""
    run = Path(run).resolve()
    if (
        manifest.get("status") not in STOPPED
        or manifest.get("recipe") != "go1-joystick"
    ):
        raise ValueError("Go1 checkpoint recovery requires a stopped training run")
    receipt_path = run / "recovery.json"
    if not receipt_path.is_file():
        raise ValueError(
            "No committed Go1 recovery checkpoint; the run may have stopped before its first save"
        )
    for path in (
        receipt_path,
        run / "request.json",
        run / "runtime.json",
        run / "assets.json",
        run / "checkpoints",
    ):
        if path.is_symlink():
            raise ValueError("Go1 recovery files must not be symlinks")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("schema") != 1:
        raise ValueError("Invalid Go1 recovery receipt")
    if receipt.get("request_sha256") != sha256(run / "request.json"):
        raise ValueError("Go1 recovery request SHA256 mismatch")
    request = json.loads((run / "request.json").read_text())
    if request != manifest.get("request"):
        raise ValueError("Go1 recovery request differs from the run manifest")
    result = receipt.get("result")
    if not isinstance(result, dict):
        raise ValueError("Invalid Go1 recovery result")
    iteration = result.get("checkpoint_iteration")
    start, budget = request.get("start_iteration"), request.get("updates")
    prior = request.get("prior_updates", 0)
    if (
        any(type(v) is not int or v < 0 for v in (iteration, start, prior))
        or type(budget) is not int
        or budget <= 0
        or not start <= iteration < start + budget
        or type(result.get("completed_updates")) is not int
        or type(result.get("cumulative_updates")) is not int
        or result["completed_updates"] != iteration - start + 1
        or result["cumulative_updates"] != prior + iteration - start + 1
    ):
        raise ValueError("Go1 recovery progress differs from the training budget")
    for field, option in (
        ("task_semantics", "go1_semantics"),
        ("reward_profile", "go1_reward_profile"),
        ("command_profile", "go1_command_profile"),
        ("learning_rate_override", "go1_learning_rate"),
    ):
        if field not in result or result[field] != request.get(option):
            raise ValueError(f"Go1 recovery {field} differs from the training request")
    for name in ("runtime", "assets"):
        if result.get(name) != json.loads((run / f"{name}.json").read_text()):
            raise ValueError(f"Go1 recovery {name} differs from the training record")
    relative = f"checkpoints/model-{iteration:09d}.pt"
    if result.get("checkpoint") != relative:
        raise ValueError("Invalid Go1 recovery checkpoint path")
    path = run / relative
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(run)
    ):
        raise ValueError("Go1 recovery checkpoint is missing or outside the run")
    if sha256(path) != result.get("checkpoint_sha256"):
        raise ValueError("Go1 recovery checkpoint SHA256 mismatch")
    from ._go1_implementation import copy_go1_implementation

    with tempfile.TemporaryDirectory(
        prefix="embodiedforge-recovery-check-"
    ) as temporary:
        if (
            copy_go1_implementation(run, manifest, Path(temporary) / "implementation")
            is None
        ):
            raise ValueError("Go1 recovery requires a recorded implementation snapshot")
    return result
