"""Validate standalone exports before publishing a deployment artifact."""

import json
import os
import shutil
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ._microduck_native import snapshot_command, snapshot_runtime
from ._microduck_run import checkpoint_digest
from .logging import get_logger

LOGGER = get_logger("microduck")


def remove_export_links(model, report, output, validation):
    """Attempt each rollback independently without hiding the export failure."""
    for staged, published in ((model, output), (report, validation)):
        try:
            if os.path.samestat(staged.stat(), published.lstat()):
                published.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
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


def export_run(args, python, identity, env):
    # Import lazily to share the CLI's process runner without loading SDKs.
    from . import microduck as cli

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
    cli.LOGGER.info("Export diagnostics: %s", attempt)
    env = cli.training_environment(env)
    manifest = {
        "schema": 1,
        "workflow": "export",
        "task": cli.TASK,
        "source": identity,
        "launcher": cli.launcher_identity(),
        "python": str(python),
        "checkpoint_source": str(checkpoint),
        "onnx": str(output),
        "validation": str(validation),
        "execution": {"headless": True, "mujoco_gl": "egl", "viewer": None},
        "started_at": datetime.now(timezone.utc).isoformat(),
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
        cli.write_json(attempt / "run.json", manifest)
        cli.run(
            command,
            cwd=attempt,
            env=env,
            log_path=log,
            echo=not getattr(args, "quiet", False),
        )

    published = False
    try:
        cli.write_json(attempt / "run.json", manifest)
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
        staged = attempt / "policy.onnx"
        execute(
            cli.export_command(
                python,
                frozen_checkpoint,
                staged,
                native=env.get("EF_MICRODUCK_NATIVE") == "1",
            ),
            "export",
        )
        execute(
            [str(python), "-I", str(cli.worker_for(env)), "onnx", str(staged)],
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
        manifest["phase"] = "publish"
        manifest["onnx_sha256"] = checkpoint_digest(staged)
        cli.write_json(attempt / "run.json", manifest)
        report.update(
            onnx=str(output),
            onnx_sha256=manifest["onnx_sha256"],
            checkpoint_sha256=digest,
            export_run=str(attempt / "run.json"),
        )
        ready_report = attempt / "publish.validation.json"
        cli.write_json(ready_report, report)
        publish_export(staged, ready_report, output, validation)
        published = True
        manifest.update(status="complete", phase="complete")
        cli.finish_run(attempt, manifest)
    except KeyboardInterrupt as exc:
        manifest.update(
            status="interrupted",
            failed_phase="publish" if published else manifest["phase"],
            interrupt_signal=getattr(exc, "signum", signal.SIGINT),
        )
        raise
    except Exception as exc:
        manifest.update(
            status="failed",
            failed_phase="publish" if published else manifest["phase"],
            error=str(exc),
        )
        raise
    finally:
        if manifest["status"] != "complete":
            if published:
                remove_export_links(staged, ready_report, output, validation)
            cli.finish_run(attempt, manifest)
    cli.LOGGER.info("Validated Microduck export: %s", output)
