"""Versioned run identity and a common read-only view of workflow status.

The workflow manifest remains authoritative. This module does not load models,
probe running processes or decide whether an artifact is safe to resume.
"""

import hashlib
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import write_json
from .process import termination_signals

STATUSES = frozenset(
    {
        "preparing",
        "running",
        "complete",
        "rejected",
        "failed",
        "interrupted",
        "timed_out",
    }
)
ACTIVE_STATUSES = frozenset({"preparing", "running"})
STATUS_ALIASES = {"recording": "running", "closed": "complete"}
LOGGER = logging.getLogger(__name__)


class RunRejected(RuntimeError):
    """Execution completed but explicit acceptance criteria were not met."""


def failure_record(error: BaseException) -> dict:
    """Describe an execution failure without loading a workflow or its SDK.

    CalledProcessError remains a worker failure, including a worker killed by a
    signal. Only a launcher interruption denotes cancellation of this run.
    """
    fields = {
        "status": (
            "rejected"
            if isinstance(error, RunRejected)
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "timed_out"
            if isinstance(error, subprocess.TimeoutExpired)
            else "failed"
        ),
        "error": f"{type(error).__name__}: {error}",
    }
    if isinstance(error, subprocess.CalledProcessError):
        fields.update(worker_exit_observed=True, worker_returncode=error.returncode)
    if isinstance(error, KeyboardInterrupt):
        fields["interrupt_signal"] = getattr(error, "signum", signal.SIGINT)
    return fields


@contextmanager
def run_lifecycle(record: dict, publish, *, finalize=None, label="run record"):
    """Track an owned workflow record through startup, execution and publication.

    ``publish(record)`` chooses the workflow's existing manifest path and may
    add checkpoint/progress fields. The caller owns the output and coordinates
    writers; this helper never claims an existing run. Initial preparing/running
    states are accepted, and successful execution ends complete or rejected.
    Termination handlers cover startup and final publication as well as the body.
    A final publication failure never replaces an active execution failure.
    Optional ``finalize(record)`` collects terminal metadata after execution. Its
    failure is recorded when execution succeeded, and logged when preserving an
    earlier execution failure. The final record is published in either case.
    """
    if record.get("status") not in ACTIVE_STATUSES:
        raise ValueError("Run lifecycle must start preparing or running")
    if "run_contract" in record:
        record["run_contract"] = validate_run_identity(record["run_contract"])
    record.setdefault("started_at", datetime.now(timezone.utc).isoformat())
    began = time.monotonic()
    with termination_signals():
        active_error = False
        try:
            publish(record)
            yield record
            if record.get("status") == "running":
                record["status"] = "complete"
            if record.get("status") not in ("complete", "rejected"):
                raise ValueError("A successful run must finish complete or rejected")
        except BaseException as error:
            active_error = True
            record.update(failure_record(error))
            raise
        finally:
            record.update(
                finished_at=datetime.now(timezone.utc).isoformat(),
                seconds=time.monotonic() - began,
            )
            finalization_error = None
            if finalize is not None:
                try:
                    finalize(record)
                except BaseException as error:
                    if active_error:
                        LOGGER.exception(
                            "Could not collect terminal metadata for %s; "
                            "preserving its original error",
                            label,
                        )
                    else:
                        record.update(failure_record(error))
                        finalization_error = error
            try:
                publish(record)
            except BaseException as error:
                if not active_error and finalization_error is None:
                    # Publication may fail after exposing a terminal manifest.
                    # Make one best-effort correction while keeping the original
                    # publication exception authoritative (including signals).
                    record.update(failure_record(error))
                    try:
                        publish(record)
                    except BaseException:
                        LOGGER.exception(
                            "Could not publish failed finalization for %s; "
                            "preserving its original error",
                            label,
                        )
                    raise
                LOGGER.exception(
                    "Could not finalize %s; preserving its original error", label
                )
            if finalization_error is not None:
                raise finalization_error


@contextmanager
def managed_run(directory, identity, *, metadata=None, writer=None):
    """Publish lifecycle state in a new run directory owned by the caller.

    The directory must exist and run.json must not already exist. Yield a mutable
    manifest for workflow-specific fields. A body failure is always re-raised;
    final publication errors cannot replace it. Callers may report complete or
    rejected explicitly; otherwise a successful body becomes complete.
    """
    root = Path(directory)
    path = root / "run.json"
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    identity = validate_run_identity(identity)
    record = dict(metadata or {})
    reserved = {"run_contract", "status", "started_at", "finished_at", "seconds"}
    if record.keys() & reserved:
        raise ValueError("Run metadata must not override lifecycle fields")
    record.update(
        run_contract=identity,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    publish = writer if writer is not None else write_json
    with run_lifecycle(record, lambda current: publish(path, current), label=path):
        yield record


def run_identity(workflow: str, operation: str, task: str | None = None) -> dict:
    """Identity stored under run_contract without changing legacy manifest fields."""
    for name, value in (("workflow", workflow), ("operation", operation)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Run {name} must be a nonempty name")
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ValueError("Run task must be a nonempty name or None")
    return {
        "schema_version": 1,
        "workflow": workflow,
        "operation": operation,
        "task": task,
    }


def validate_run_identity(value) -> dict:
    """Return a copied versioned identity, rejecting unknown or malformed schemas."""
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("Unsupported run_contract schema")
    return run_identity(
        value.get("workflow"), value.get("operation"), value.get("task")
    )


def _legacy_identity(manifest: dict, path: Path) -> dict:
    """Recognize only identifiable legacy formats; do not guess from directory names."""
    workflow = manifest.get("workflow")
    schema = manifest.get("schema_version", manifest.get("schema"))
    if manifest.get("kind") == "core-ppo" and type(schema) is int and schema == 1:
        env = manifest.get("env", {})
        if not isinstance(env, dict):
            raise ValueError("Invalid legacy core environment configuration")
        return run_identity("core", "train", env.get("task"))
    if (
        workflow in ("recipe_train", "recipe_evaluate", "recipe_solve")
        and type(schema) is int
        and schema == 1
    ):
        return run_identity(
            "recipes", workflow.removeprefix("recipe_"), manifest.get("recipe")
        )
    if workflow in ("h1_train", "h1_evaluate") and type(schema) is int and schema == 1:
        return run_identity(
            "h1-isaaclab", workflow.removeprefix("h1_"), manifest.get("task")
        )
    task = manifest.get("task")
    if workflow in ("assess", "compare") and type(schema) is int and schema == 1:
        # These older Microduck records omitted task but retained the input
        # implementation identity. Both comparison sides must identify it.
        inputs = (
            [manifest.get("input")]
            if workflow == "assess"
            else [manifest.get("before"), manifest.get("after")]
        )
        if all(_microduck_input(value) for value in inputs):
            return run_identity("microduck", workflow, "Mjlab-Velocity-Flat-MicroDuck")
    if task == "Mjlab-Velocity-Flat-MicroDuck" and type(schema) is int and schema == 1:
        return run_identity("microduck", workflow, task)
    if task == "h1-mjbatch-flat-v1":
        return run_identity("h1-native", "train", task)
    if (
        (path.name == "act-run.json" or path.name.endswith(".act-run.json"))
        and type(schema) is int
        and schema == 1
    ):
        return run_identity(
            "act", "resume" if manifest.get("resumed_from") else "train", "act"
        )
    if (
        type(schema) is int
        and schema == 1
        and {"config", "env_spec", "completed_episodes", "run_id"} <= manifest.keys()
    ):
        return run_identity("core-data", "record", task)
    backend = manifest.get("backend")
    if backend in ("gmr", "sonic"):
        return run_identity(backend, "retarget" if backend == "gmr" else "evaluate")
    # These SDK launchers stored their identity in request.json. Consult it only
    # for their otherwise indistinguishable legacy status-only run records.
    if (
        manifest.get("command") in ("train", "evaluate", "export")
        and (path.parent / "request.json").is_file()
    ):
        request = json.loads((path.parent / "request.json").read_bytes())
        if isinstance(request, dict) and request.get("command") == manifest["command"]:
            if request.get("config") == "maniskill_sac_mlp" and "revision" in request:
                return run_identity("rlinf", manifest["command"], request["config"])
            if (
                request.get("task") == "G1-Inspire-HOI-v0"
                and {"weave_revision", "isaaclab_revision"} <= request.keys()
            ):
                return run_identity("weave", manifest["command"], request["task"])
    raise ValueError("Unrecognized legacy run manifest; no run_contract identity")


def _microduck_input(value):
    if not isinstance(value, dict) or not isinstance(value.get("source"), dict):
        return False
    source = value["source"]
    return source.get("implementation") == "embodiedforge.locomotion.microduck" or (
        source.get("revision") == "53b8971b61baf5b7f3c16d135dd7cac37623de4b"
        and isinstance(source.get("uv_lock_sha256"), str)
    )


def read_run(directory: str | Path) -> dict:
    """Read a stable JSON snapshot of recorded status, preserving workflow semantics.

    A running status may be stale after a host crash. A complete status does not
    prove acceptance or checkpoint compatibility. Call the workflow's own loader
    for recovery, evaluation or artifact verification.
    """
    root = Path(directory).expanduser().resolve()
    path = root / "run.json"
    if not path.exists():
        for candidate in (
            root / "act-run.json",
            root.parent / f".{root.name}.act-run.json",
            root / "launcher.json",
        ):
            if candidate.is_file():
                path = candidate
                break
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be a JSON object")
    declared = "run_contract" in manifest
    identity = (
        validate_run_identity(manifest["run_contract"])
        if declared
        else _legacy_identity(manifest, path)
    )
    recorded = manifest.get("status")
    if not isinstance(recorded, str):
        raise ValueError(f"Unsupported recorded run status: {recorded!r}")
    status = (
        STATUS_ALIASES.get(recorded, recorded)
        if identity["workflow"] in ("core-data", "episode-data")
        else recorded
    )
    if status not in STATUSES:
        raise ValueError(f"Unsupported recorded run status: {recorded!r}")
    error = manifest.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("Recorded run error must be a string or null")
    return {
        **identity,
        "directory": str(root),
        "manifest": os.path.relpath(path, root),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "identity_source": "declared" if declared else "legacy",
        "status": status,
        "recorded_status": recorded,
        "terminal": status not in ACTIVE_STATUSES,
        "process_liveness": "not_checked",
        "artifact_verification": "not_checked",
        "error": error,
    }


def main(argv=None) -> None:
    """Inspect one run without importing its simulation or training environment."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Show recorded run identity and status")
    inspect.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(read_run(args.run), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Cannot inspect run: {exc}\n")
