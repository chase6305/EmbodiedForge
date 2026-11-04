"""Atomic core PPO run records and independently recoverable saved updates."""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from uuid import uuid4

from .learning.checkpoints import validate_adam_checkpoint
from .runtime.artifacts import discard_partial
from .runtime.artifacts import write_json as write_json
from .runtime.hashing import file_sha256

LOGGER = logging.getLogger(__name__)


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
    # CPython 3.12.0 retains unique pathlib components through interning.
    # Keep high-volume snapshot and temporary names as filesystem strings.
    name = f"checkpoint-{updates:09d}.pt"
    target = os.path.join(os.fspath(directory), name)
    temporary = f"{target}.{uuid4().hex}.tmp"
    previous = metadata.get("checkpoint")
    try:
        torch.save(checkpoint, temporary)
        checksum = file_sha256(temporary)
        os.replace(temporary, target)
    except BaseException:
        discard_partial(temporary, label="core PPO artifact")
        raise
    committed = {
        **metadata,
        "saved_updates": updates,
        "checkpoint": f"checkpoints/{name}",
        "checkpoint_sha256": checksum,
    }
    write_json(root / "run.json", committed)
    metadata.update(committed)
    alias = root / "checkpoint.pt"
    alias_temporary = f"{os.fspath(alias)}.{uuid4().hex}.tmp"
    try:
        shutil.copyfile(target, alias_temporary)
        os.replace(alias_temporary, alias)
    except BaseException:
        discard_partial(alias_temporary, label="core PPO artifact")
        raise
    keep = {committed["checkpoint"], previous}
    for old in directory.glob("checkpoint-*.pt"):
        number = old.stem.removeprefix("checkpoint-")
        if (
            number.isdecimal()
            and old.name == f"checkpoint-{int(number):09d}.pt"
            and f"checkpoints/{old.name}" not in keep
        ):
            try:
                old.unlink()
            except OSError:
                LOGGER.warning(
                    "Could not remove old core PPO checkpoint %s", old, exc_info=True
                )


def _resume_manifest(root: str) -> dict:
    with open(os.path.join(root, "run.json"), encoding="utf-8") as stream:
        metadata = json.load(stream)
    updates = metadata.get("saved_updates")
    if (
        type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") != 1
        or metadata.get("kind") != "core-ppo"
        or type(updates) is not int
        or updates <= 0
        or metadata.get("checkpoint") != f"checkpoints/checkpoint-{updates:09d}.pt"
    ):
        raise ValueError("Expected a core PPO run with a committed checkpoint")
    return metadata


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as stream:
        return stream.read()


def read_resume(path: str | Path) -> dict:
    """Read verified bytes, retrying a directory reference pruned by live training.

    Reading a running directory is allowed: select a saved snapshot while the
    source continues independently. At most three reads are attempted if its
    committed progress advances and removes the previously selected snapshot.
    """
    # Avoid retaining every live-training checkpoint name in pathlib's intern
    # table on CPython 3.12.0. Read one immutable byte snapshot as before.
    source = os.fspath(path)
    metadata = None
    if os.path.isdir(source):
        root = source
        metadata = _resume_manifest(root)
        for attempt in range(3):
            source = os.path.join(root, metadata["checkpoint"])
            try:
                payload = _read_bytes(source)
                break
            except FileNotFoundError:
                newer = _resume_manifest(root)
                if attempt == 2 or newer["saved_updates"] <= metadata["saved_updates"]:
                    raise
                metadata = newer
    else:
        payload = _read_bytes(source)
    checksum = hashlib.sha256(payload).hexdigest()
    if metadata is not None and checksum != metadata.get("checkpoint_sha256"):
        raise ValueError("Core PPO checkpoint hash mismatch")
    return {
        "payload": payload,
        "sha256": checksum,
        "source": os.path.realpath(source),
        "saved_updates": metadata["saved_updates"] if metadata is not None else None,
    }
