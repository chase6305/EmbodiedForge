"""Microduck run ownership, checkpoint discovery and frozen evaluation inputs."""

import hashlib
import json
import os
import re
import socket
from pathlib import Path

from .runtime.hashing import copy_file_snapshot
from .runtime.hashing import file_sha256 as checkpoint_digest


def _process_state(pid):
    # comm may contain spaces and parentheses; fields after its final ')' start
    # at field 3. starttime is field 22, in kernel clock ticks since boot.
    fields = (
        Path(f"/proc/{pid}/stat").read_text(errors="replace").rsplit(")", 1)[1].split()
    )
    return fields[0], int(fields[19])


def launcher_identity():
    identity = {"pid": os.getpid(), "hostname": socket.gethostname()}
    try:
        identity["start_ticks"] = _process_state(identity["pid"])[1]
        identity["boot_id"] = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        )
    except (OSError, ValueError, IndexError):
        pass
    return identity


def launcher_status(identity):
    """Report local launcher liveness without mistaking a reused PID for the job."""
    if not isinstance(identity, dict):
        return {"alive": None, "reason": "not_recorded"}
    if identity.get("hostname") != socket.gethostname():
        return {"alive": None, "reason": "different_host"}
    if (
        type(identity.get("pid")) is not int
        or identity["pid"] <= 0
        or type(identity.get("start_ticks")) is not int
        or not identity.get("boot_id")
    ):
        return {"alive": None, "reason": "incomplete_identity"}
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if boot_id != identity["boot_id"]:
            return {"alive": False, "reason": "different_boot"}
        state, start = _process_state(identity["pid"])
    except FileNotFoundError:
        if not Path("/proc/self/stat").exists():
            return {"alive": None, "reason": "proc_unavailable"}
        return {"alive": False, "reason": "exited"}
    except (OSError, ValueError, IndexError):
        return {"alive": None, "reason": "proc_unreadable"}
    if start != identity["start_ticks"]:
        return {"alive": False, "reason": "pid_reused"}
    if state in ("Z", "X"):
        return {"alive": False, "reason": "exited"}
    return {"alive": True, "reason": "identity_matches"}


def training_checkpoints(directory: Path) -> list[Path]:
    """Find saved model files without loading them or including temporary files."""
    directory = directory.resolve()
    return sorted(
        (
            path
            for path in directory.glob("logs/rsl_rl/microduck/*/model_*.pt")
            if re.fullmatch(r"model_[0-9]+\.pt", path.name)
            and path.is_file()
            and path.resolve().is_relative_to(directory)
        ),
        key=lambda path: int(path.stem.removeprefix("model_")),
    )


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


def snapshot_inputs(output, checkpoint, options, source_run=None):
    directory = output / "inputs"
    directory.mkdir(exist_ok=False)
    sources = {"checkpoint": (checkpoint, "checkpoint.pt")}
    if options.get("onnx"):
        sources["onnx"] = (Path(options["onnx"]), "policy.onnx")
    result = {}
    for role, (source, filename) in sources.items():
        expected = checkpoint_digest(source)
        if role == "checkpoint" and source_run is not None:
            if expected != source_run["checkpoint_sha256"]:
                raise ValueError("Selected training run checkpoint SHA256 mismatch")
        target = directory / filename
        copied = copy_file_snapshot(source, target)
        if copied != expected or checkpoint_digest(target) != expected:
            raise ValueError(f"Evaluation {role} changed while taking snapshot; retry")
        result[role] = {"source": str(source), "path": str(target), "sha256": expected}
    return result


def verify_input_snapshot(snapshot):
    for role, item in snapshot.items():
        if checkpoint_digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Evaluation {role} snapshot SHA256 mismatch")


def verify_report_inputs(report, snapshot):
    if (
        report.get("checkpoint_metadata", {}).get("sha256")
        != snapshot["checkpoint"]["sha256"]
    ):
        raise ValueError("Evaluation report checkpoint differs from input snapshot")
    parity = report.get("onnx_parity")
    if "onnx" in snapshot:
        if (
            not isinstance(parity, dict)
            or parity.get("sha256") != snapshot["onnx"]["sha256"]
        ):
            raise ValueError("Evaluation report ONNX differs from input snapshot")
    elif parity is not None:
        raise ValueError("Evaluation report contains an unrequested ONNX comparison")
