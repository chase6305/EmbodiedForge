"""Resolve recorded Microduck checkpoints without importing its training SDK."""

import hashlib
import json
import re
from pathlib import Path


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_relative(recorded):
    if not isinstance(recorded, str):
        raise ValueError("Training run has no recorded checkpoint")
    parts = Path(recorded).parts[-5:]
    if (
        len(parts) != 5
        or parts[:3] != ("logs", "rsl_rl", "microduck")
        or parts[3] in (".", "..")
        or re.fullmatch(r"model_[0-9]+\.pt", parts[4]) is None
    ):
        raise ValueError("Recorded checkpoint is outside the Microduck training layout")
    return Path(*parts)


def resolve_run_checkpoint(directory, identity, task, *, allow_incomplete=False):
    """Use the recorded model inside this run, including after directory relocation."""
    directory = Path(directory).expanduser().resolve()
    raw = (directory / "run.json").read_bytes()
    report = json.loads(raw)
    recovery = (
        allow_incomplete
        and isinstance(report, dict)
        and report.get("status") in ("failed", "interrupted")
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != 1
        or report.get("workflow") not in ("train", "smoke")  # Read historical records.
        or report.get("task") != task
        or (report.get("status") != "complete" and not recovery)
    ):
        raise ValueError(
            "--run/--resume-run requires a completed Microduck training run"
        )
    source = report.get("source")
    for key in ("revision", "uv_lock_sha256"):
        if (
            not isinstance(source, dict)
            or not identity.get(key)
            or source.get(key) != identity[key]
        ):
            raise ValueError(f"Training run source differs: {key}")
    if recovery:
        from ._microduck_worker import launcher_status

        if not report.get("finished_at"):
            raise ValueError("Recovery requires a recorded finish time")
        if launcher_status(report.get("launcher"))["alive"] is True:
            raise ValueError("Training launcher is still alive; wait for it to exit")
        recorded = report.get("checkpoints")
        if not isinstance(recorded, list) or not recorded:
            raise ValueError(
                "Recovery run has no recorded checkpoints; use --resume for an explicitly selected model"
            )
        candidates = [_checkpoint_relative(path) for path in recorded]
        if len({path.parent for path in candidates}) != 1:
            raise ValueError("Recovery has ambiguous checkpoint sessions")
        iterations = [int(path.stem.removeprefix("model_")) for path in candidates]
        if len(set(iterations)) != len(iterations):
            raise ValueError("Recovery has duplicate checkpoint iterations")
        relative = candidates[iterations.index(max(iterations))]
    else:
        relative = _checkpoint_relative(report.get("checkpoint"))
    # Legacy runs saved absolute paths. Their fixed training layout gives an
    # unambiguous local suffix; never fall back to the old absolute location.
    if (
        not recovery
        and "checkpoint_relative" in report
        and report["checkpoint_relative"] != relative.as_posix()
    ):
        raise ValueError("Recorded checkpoint paths disagree")
    checkpoint = (directory / relative).resolve()
    if not checkpoint.is_relative_to(directory) or not checkpoint.is_file():
        raise ValueError("Recorded checkpoint is missing or resolves outside the run")
    digest = checkpoint_digest(checkpoint)
    expected = report.get("checkpoint_sha256")
    if recovery and (
        report.get("checkpoint") is None
        or _checkpoint_relative(report["checkpoint"]) != relative
    ):
        expected = None
    if expected is not None and expected != digest:
        raise ValueError("Training run checkpoint SHA256 mismatch")
    return checkpoint, {
        "run": str(directory),
        "run_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_relative": relative.as_posix(),
        "checkpoint_sha256": digest,
        "verification": "verified"
        if expected is not None
        else "recorded_at_recovery"
        if recovery
        else "unverified_legacy",
        "selection": "latest_recorded_recovery"
        if recovery
        else "completed_run_checkpoint",
        "run_status": report["status"],
    }
