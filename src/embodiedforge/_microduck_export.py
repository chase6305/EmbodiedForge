"""Validate standalone exports before publishing a deployment artifact."""

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ._microduck_native import snapshot_command, snapshot_runtime
from ._microduck_run import checkpoint_digest
from .logging import get_logger
from .runtime.records import run_identity, run_lifecycle

LOGGER = get_logger("microduck")


@dataclass(frozen=True)
class ExportOperations:
    """Launcher-supplied operations; the export workflow never imports its CLI."""

    task: str
    launcher_identity: Callable[[], dict]
    training_environment: Callable[[dict], dict]
    export_command: Callable[..., list[str]]
    worker_for: Callable[[dict], Path]
    run: Callable[..., None]
    write_json: Callable[[Path, dict], None]


def remove_export_links(model, report, output, validation):
    """Attempt each rollback independently without hiding the export failure."""
    for staged, published in ((model, output), (report, validation)):
        try:
            if os.path.samestat(staged.stat(), published.lstat()):
                published.unlink()
        except FileNotFoundError:
            pass
        except BaseException as exc:
            LOGGER.warning("Could not remove export file %s: %s", published, exc)


def publish_export(model, report, output, validation):
    """Publish on one filesystem without overwriting concurrent exports.

    The model becomes visible only after its report. Roll back our own links
    on failure, including interruption, without removing another writer's files.
    """
    try:
        os.link(report, validation)
        os.link(model, output)
    except BaseException:
        remove_export_links(model, report, output, validation)
        raise


def export_run(args, python, identity, env, *, operations: ExportOperations):
    requested = args.output.expanduser()
    output = requested.parent.resolve() / requested.name
    if output.suffix.lower() != ".onnx":
        raise ValueError("Export --output must have an .onnx extension")
    validation = output.with_suffix(".validation.json")
    for target in (output, validation):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {target}")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    attempt = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent)
    )
    LOGGER.info("Export diagnostics: %s", attempt)
    env = operations.training_environment(env)
    manifest = {
        "schema": 1,
        "workflow": "export",
        "task": operations.task,
        "run_contract": run_identity("microduck", "export", operations.task),
        "source": identity,
        "launcher": operations.launcher_identity(),
        "python": str(python),
        "checkpoint_source": str(checkpoint),
        "onnx": str(output),
        "validation": str(validation),
        "execution": {"headless": True, "mujoco_gl": "egl", "viewer": None},
        "status": "running",
        "phase": "snapshot",
        "commands": [],
        "logs": [],
    }
    if getattr(args, "source_run", None) is not None:
        manifest["source_run"] = args.source_run

    def execute(command, phase):
        command = snapshot_command(command, env)
        log = attempt / f"{len(manifest['commands']):02d}-{phase}.log"
        manifest.update(phase=phase, active_log=log.name)
        manifest["commands"].append(command)
        manifest["logs"].append(log.name)
        operations.write_json(attempt / "run.json", manifest)
        operations.run(
            command,
            cwd=attempt,
            env=env,
            log_path=log,
            echo=not getattr(args, "quiet", False),
        )

    staged = attempt / "policy.onnx"
    ready_report = attempt / "publish.validation.json"

    def publish(record):
        if record["status"] not in {"running", "complete"}:
            phase = record["phase"]
            record["failed_phase"] = "publish" if phase == "complete" else phase
            if phase in {"publish", "complete"}:
                remove_export_links(staged, ready_report, output, validation)
        operations.write_json(attempt / "run.json", record)

    with run_lifecycle(manifest, publish, label=attempt / "run.json"):
        env, source_snapshot = snapshot_runtime(attempt, identity, env)
        if source_snapshot is not None:
            manifest["implementation_snapshot"] = str(source_snapshot)
        frozen_checkpoint = attempt / "checkpoint.pt"
        shutil.copyfile(checkpoint, frozen_checkpoint)
        digest = checkpoint_digest(frozen_checkpoint)
        manifest.update(checkpoint=str(frozen_checkpoint), checkpoint_sha256=digest)
        expected = manifest.get("source_run", {}).get("checkpoint_sha256")
        if expected is not None and digest != expected:
            raise ValueError("Checkpoint SHA256 mismatch after export snapshot")
        native = env.get("EF_MICRODUCK_NATIVE") == "1"
        execute(
            operations.export_command(
                python,
                frozen_checkpoint,
                staged,
                native=native,
            ),
            "export",
        )
        # The local exporter validates against its loaded actor before exiting.
        # Legacy upstream exports still need standalone structural validation.
        if not native:
            execute(
                [
                    str(python),
                    "-I",
                    str(operations.worker_for(env)),
                    "onnx",
                    str(staged),
                ],
                "validate",
            )
        report = json.loads(staged.with_suffix(".validation.json").read_text())
        if (
            report.get("actor_dim") != 61
            or report.get("action_dim") != 14
            or report.get("finite_inference") is not True
            or report.get("onnx") != str(staged)
        ):
            raise ValueError("Export validation report is incomplete or inconsistent")
        if native:
            parity = report.get("policy_parity")
            if not isinstance(parity, dict) or parity.get("passed") is not True:
                raise ValueError("Native export is missing checkpoint actor parity")
        manifest["phase"] = "publish"
        manifest["onnx_sha256"] = checkpoint_digest(staged)
        operations.write_json(attempt / "run.json", manifest)
        report.update(
            onnx=str(output),
            onnx_sha256=manifest["onnx_sha256"],
            checkpoint_sha256=digest,
            export_run=str(attempt / "run.json"),
        )
        operations.write_json(ready_report, report)
        publish_export(staged, ready_report, output, validation)
        manifest.update(status="complete", phase="complete")
    LOGGER.info("Validated Microduck export: %s", output)
