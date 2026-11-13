"""Validate recipe checkpoints without importing the launcher or a learner."""

import json
from pathlib import Path

from embodiedforge.runtime.hashing import file_sha256 as sha256

from .recipe_catalog import GO1_STANDALONE_SOURCE, RECIPES, SOURCES


def checkpoint_input(
    run: Path, recipe: str, *, allow_recovery=False
) -> tuple[Path, dict]:
    data = json.loads((run / "run.json").read_text())
    if (
        data.get("schema") != 1
        or data.get("workflow") != "recipe_train"
        or (
            data.get("status") != "complete"
            and not (
                allow_recovery
                and recipe == "go1-joystick"
                and data.get("status") in {"interrupted", "timed_out", "failed"}
            )
        )
        or data.get("recipe") != recipe
        or not (
            data.get("source", {}).get("revision") == SOURCES[RECIPES[recipe]["source"]]
            or (
                recipe == "go1-joystick" and data.get("source") == GO1_STANDALONE_SOURCE
            )
        )
    ):
        raise ValueError(
            "Input must be a completed training run for this recipe/version"
        )
    if data["status"] != "complete":
        from .._go1_recovery import recovery_result

        data = {
            **data,
            "result": recovery_result(run, data),
            "checkpoint_origin": "recovery.json",
        }
    artifact = data["result"]
    path = (run / artifact["checkpoint"]).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        raise ValueError("Checkpoint is missing or outside the run")
    if sha256(path) != artifact["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    return path, data
