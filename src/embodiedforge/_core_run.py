"""Atomic core PPO run records and independently recoverable saved updates."""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from uuid import uuid4

from ._adam_checkpoint import validate_adam_checkpoint

LOGGER = logging.getLogger(__name__)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    except BaseException:
        _remove_partial(temporary)
        raise


def _remove_partial(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning(
            "Could not remove partial core PPO artifact %s", path, exc_info=True
        )


def publish_checkpoint(root: Path, checkpoint: dict, optimizer, metadata: dict) -> None:
    """Commit a versioned snapshot before replacing the convenience checkpoint.

    The run record references a versioned file, so a failed alias or manifest
    write cannot destroy the last committed recovery point. This is atomic file
    publication, not an fsync/host-power-loss durability guarantee.
    """
    import torch

    validate_adam_checkpoint(
        optimizer, checkpoint["optimizer"], label="core PPO optimizer"
    )
    if any(not torch.isfinite(value).all() for value in checkpoint["model"].values()):
        raise ValueError("Non-finite core PPO checkpoint policy")
    updates = checkpoint["updates"]
    if type(updates) is not int or updates <= 0:
        raise ValueError("Invalid core PPO checkpoint update count")
    directory = root / "checkpoints"
    directory.mkdir(exist_ok=True)
    target = directory / f"checkpoint-{updates:09d}.pt"
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    previous = metadata.get("checkpoint")
    try:
        torch.save(checkpoint, temporary)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(target)
    except BaseException:
        _remove_partial(temporary)
        raise
    committed = {
        **metadata,
        "saved_updates": updates,
        "checkpoint": target.relative_to(root).as_posix(),
        "checkpoint_sha256": checksum,
    }
    write_json(root / "run.json", committed)
    metadata.update(committed)
    alias = root / "checkpoint.pt"
    alias_temporary = alias.with_name(f"{alias.name}.{uuid4().hex}.tmp")
    try:
        shutil.copyfile(target, alias_temporary)
        alias_temporary.replace(alias)
    except BaseException:
        _remove_partial(alias_temporary)
        raise
    keep = {committed["checkpoint"], previous}
    for old in directory.glob("checkpoint-*.pt"):
        number = old.stem.removeprefix("checkpoint-")
        if (
            number.isdecimal()
            and old.name == f"checkpoint-{int(number):09d}.pt"
            and old.relative_to(root).as_posix() not in keep
        ):
            try:
                old.unlink()
            except OSError:
                LOGGER.warning(
                    "Could not remove old core PPO checkpoint %s", old, exc_info=True
                )


def read_resume(path: str | Path) -> dict:
    """Read exactly the verified bytes; a run directory selects its committed file.

    Reading a running directory is allowed: this selects its saved snapshot,
    while the source process continues independently in its original directory.
    """
    source = Path(path)
    metadata = None
    if source.is_dir():
        metadata = json.loads((source / "run.json").read_text())
        updates = metadata.get("saved_updates")
        if (
            metadata.get("schema_version") != 1
            or metadata.get("kind") != "core-ppo"
            or type(updates) is not int
            or updates <= 0
            or metadata.get("checkpoint") != f"checkpoints/checkpoint-{updates:09d}.pt"
        ):
            raise ValueError("Expected a core PPO run with a committed checkpoint")
        source = source / metadata["checkpoint"]
    payload = source.read_bytes()
    checksum = hashlib.sha256(payload).hexdigest()
    if metadata is not None and checksum != metadata.get("checkpoint_sha256"):
        raise ValueError("Core PPO checkpoint hash mismatch")
    return {
        "payload": payload,
        "sha256": checksum,
        "source": str(source.resolve()),
        "saved_updates": metadata["saved_updates"] if metadata is not None else None,
    }
