"""Run repository-owned Microduck walking in an isolated Python environment.

This is a process integration, not an implementation of Microduck in VectorEnv.
No training or simulation SDK is imported into the EmbodiedForge process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from ._microduck_export import export_run
from ._microduck_inputs import (
    snapshot_inputs,
    verify_input_snapshot,
)
from ._microduck_native import native_command, snapshot_command, snapshot_runtime
from ._microduck_worker import launcher_identity
from ._microduck_worker import training_checkpoints as checkpoints
from .logging import get_logger, setup_logging

REVISION = "53b8971b61baf5b7f3c16d135dd7cac37623de4b"
TASK = "Mjlab-Velocity-Flat-MicroDuck"
WORKER = Path(__file__).with_name("_microduck_worker.py")
LOGGER = get_logger("microduck")


def worker_for(env):
    if env.get("EF_MICRODUCK_NATIVE") == "1":
        from ._microduck_native import WORKER as native_worker

        return native_worker
    return WORKER


class EvaluationRejected(RuntimeError):
    """Evaluation ran successfully, but the policy missed explicit criteria."""


def evaluation_criteria(args) -> dict:
    from ._microduck_reports import validate_criteria

    return validate_criteria(
        {
            name: value
            for name in ("max_planar_rmse", "max_yaw_rmse", "min_survival_fraction")
            if (value := getattr(args, name, None)) is not None
        }
    )


def save_acceptance(output: Path, reports: list[dict], criteria: dict) -> None:
    if not criteria:
        return
    from ._microduck_reports import assess_evaluations

    assessment = assess_evaluations(reports, criteria)
    write_json(output / "acceptance.json", assessment)
    if not assessment["passed"]:
        failures = [
            f"seed {check['seed']}: {check['criterion']} measured {check['actual']:.6g}, required {check['operator']} {check['threshold']:.6g}"
            for check in assessment["checks"]
            if not check["passed"]
        ]
        raise EvaluationRejected("; ".join(failures))


def source_identity(repo: Path) -> dict:
    """Fail before installing/running a different or locally modified recipe."""
    for name in ("uv.lock", "pyproject.toml", "src/mjlab_microduck/train_cli.py"):
        if not (repo / name).is_file():
            raise ValueError(f"Missing upstream file: {repo / name}")
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected Microduck revision {REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("Microduck checkout has local changes; use a clean checkout")
    return {
        "repo": str(repo),
        "revision": revision,
        "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
    }


def assess_run(args) -> None:
    from ._microduck_reports import load_evaluation_run

    criteria = evaluation_criteria(args)
    if not criteria:
        raise ValueError("assess requires at least one acceptance criterion")
    reports, provenance, inputs = load_evaluation_run(args.run)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "workflow": "assess",
        "input": provenance,
        "acceptance_criteria": criteria,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run.json", manifest)
    try:
        for relative, raw in inputs.items():
            snapshot = output / "inputs" / relative
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(raw)
        save_acceptance(output, reports, criteria)
        manifest["status"] = "complete"
    except EvaluationRejected as exc:
        manifest.update(status="rejected", error=str(exc))
        raise
    except KeyboardInterrupt as exc:
        manifest.update(
            status="interrupted", interrupt_signal=getattr(exc, "signum", signal.SIGINT)
        )
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        finish_run(output, manifest)


def compare_runs(args) -> None:
    from ._microduck_reports import (
        compare_evaluations,
        compare_sources,
        load_evaluation_run,
    )

    before, before_info, before_inputs = load_evaluation_run(args.before)
    after, after_info, after_inputs = load_evaluation_run(args.after)
    source_verification = compare_sources(
        before_info.get("source"), after_info.get("source")
    )
    comparison = compare_evaluations(before, after)
    comparison["source_verification"] = source_verification
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "workflow": "compare",
        "source_verification": source_verification,
        "before": before_info,
        "after": after_info,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run.json", manifest)
    try:
        for name, inputs in (("before", before_inputs), ("after", after_inputs)):
            for relative, raw in inputs.items():
                snapshot = output / "inputs" / name / relative
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(raw)
        write_json(output / "comparison.json", comparison)
        manifest["status"] = "complete"
    except KeyboardInterrupt as exc:
        manifest.update(
            status="interrupted", interrupt_signal=getattr(exc, "signum", signal.SIGINT)
        )
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        finish_run(output, manifest)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    # The caller may be running from ef/ef-viewer or a source PYTHONPATH.
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "EF_MICRODUCK_NATIVE",
        "EF_MICRODUCK_SNAPSHOT_WORKER",
    ):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
    )
    return env


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def training_environment(env: dict[str, str]) -> dict[str, str]:
    """Select offscreen GL before SDK imports, without inheriting a desktop."""
    result = env.copy()
    for key in ("DISPLAY", "WAYLAND_DISPLAY"):
        result.pop(key, None)
    result.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    return result


def run(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path=None, echo=True
) -> None:
    from ._microduck_process import run_process

    LOGGER.info("Running %s", command)
    if log_path is not None:
        LOGGER.info("Worker log: %s", log_path)
    run_process(command, cwd=cwd, env=env, log_path=log_path, echo=echo)


def training_command(
    python: Path, *, num_envs: int, iterations: int, native=False, seed=0
) -> list[str]:
    if num_envs <= 0 or iterations <= 0:
        raise ValueError("Environment and iteration counts must be positive")
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must be between 0 and 2**32 - 1")
    command = [
        str(python),
        "-I",
        "-m",
        "mjlab_microduck.train_cli",
        TASK,
        "--env.scene.num-envs",
        str(num_envs),
        "--agent.max-iterations",
        str(iterations),
        "--agent.logger",
        "tensorboard",
        "--agent.experiment-name",
        "microduck",
        "--agent.run-name",
        "embodiedforge",
        "--agent.seed",
        str(seed),
        "--enable-nan-guard",
        "True",
        "--video",
        "False",
    ]
    return native_command(command) if native else command


def export_command(
    python: Path, checkpoint: Path, output: Path, *, native=False
) -> list[str]:
    command = [
        str(python),
        "-I",
        "-m",
        "mjlab_microduck.export",
        TASK,
        "--checkpoint-file",
        str(checkpoint),
        "--onnx-file",
        str(output),
        "--num-envs",
        "1",
    ]
    return native_command(command) if native else command


def record_worker_failure(output, manifest):
    name = "evaluation" if manifest.get("phase") == "evaluate" else "runtime"
    try:
        failure = json.loads((output / f"{name}.failure.json").read_text())
    except (OSError, ValueError):
        return
    if isinstance(failure, dict):
        manifest[f"{name}_failure"] = failure


def finish_run(output, manifest, *, include_checkpoints=False):
    """Save final state without replacing an active failure or interrupt."""
    original_error = sys.exc_info()[1]
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    try:
        if include_checkpoints:
            manifest["checkpoints"] = [str(path) for path in checkpoints(output)]
        write_json(output / "run.json", manifest)
    except OSError as exc:
        if original_error is None or manifest.get("status") == "complete":
            raise
        LOGGER.warning("Could not finalize run record %s: %s", output / "run.json", exc)


def train_run(args, python: Path, identity: dict, env: dict[str, str]) -> None:
    env = training_environment(env)
    output = args.output.expanduser().resolve()
    seed = getattr(args, "seed", 0)
    resume = getattr(args, "resume", None)
    if resume is not None:
        resume = resume.expanduser().resolve()
        if not resume.is_file():
            raise ValueError(f"Resume checkpoint not found: {resume}")
    command = training_command(
        python,
        num_envs=args.num_envs,
        iterations=args.iterations,
        native=env.get("EF_MICRODUCK_NATIVE") == "1",
        seed=seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "launcher": launcher_identity(),
        "workflow": args.command,
        "seed": seed,
        "num_envs": args.num_envs,
        "iterations": args.iterations,
        "execution": {
            "headless": True,
            "viewer": None,
            "video": False,
            "mujoco_gl": env["MUJOCO_GL"],
        },
        "task": TASK,
        "source": identity,
        "python": str(python),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "commands": [],
    }
    if getattr(args, "source_run", None) is not None:
        manifest["source_run"] = args.source_run

    def execute(command: list[str], phase: str) -> None:
        command = snapshot_command(command, env)
        log_path = output / f"{len(manifest['commands']):02d}-{phase}.log"
        manifest["phase"] = phase
        manifest["active_log"] = log_path.name
        manifest.setdefault("logs", []).append(log_path.name)
        manifest["commands"].append(command)
        write_json(output / "run.json", manifest)
        run(
            command,
            cwd=output,
            env=env,
            log_path=log_path,
            echo=not getattr(args, "quiet", False),
        )

    try:
        manifest["phase"] = "snapshot"
        write_json(output / "run.json", manifest)
        env, source_snapshot = snapshot_runtime(output, identity, env)
        if source_snapshot is not None:
            manifest["implementation_snapshot"] = str(source_snapshot)
        execute(
            [
                str(python),
                "-I",
                str(worker_for(env)),
                "check",
                str(output / "runtime.json"),
            ],
            "check",
        )
        start_iteration = 0
        if resume is not None:
            snapshot = output / "logs/rsl_rl/microduck/resume_source/checkpoint.pt"
            snapshot.parent.mkdir(parents=True)
            shutil.copyfile(resume, snapshot)
            execute(
                [
                    str(python),
                    "-I",
                    str(worker_for(env)),
                    "checkpoint",
                    str(snapshot),
                    str(output / "resume.json"),
                ],
                "checkpoint",
            )
            metadata = json.loads((output / "resume.json").read_text())
            if (
                getattr(args, "source_run", None) is not None
                and metadata["sha256"] != args.source_run["checkpoint_sha256"]
            ):
                raise RuntimeError(
                    "Selected training run checkpoint changed before resume"
                )
            start_iteration = metadata["iteration"]
            # The worker adds startup/reset events to the upstream TrainConfig.
            # Its startup restores curricula before the wrapper resets the env.
            command = [
                str(python),
                "-I",
                str(worker_for(env)),
                "resume-train",
                str(snapshot),
                "--env.scene.num-envs",
                str(args.num_envs),
                "--agent.max-iterations",
                str(args.iterations),
                "--seed",
                str(seed),
            ]
            manifest["resume"] = {
                "original": str(resume),
                "snapshot": str(snapshot),
                **metadata,
            }
        execute(command, "train")
        if resume is not None:
            audit = json.loads((output / "resume.initialization.json").read_text())
            if (
                audit["counter_at_first_reset"] != metadata["common_step_counter"]
                or audit["checkpoint_sha256"] != metadata["sha256"]
                or audit["num_envs"] != args.num_envs
            ):
                raise RuntimeError("Resume initialization disagrees with checkpoint")
            manifest["resume_initialization"] = audit
        models = checkpoints(output)
        if not models:
            raise RuntimeError("Training exited without producing a checkpoint")
        if len({path.parent for path in models}) > 1:
            raise ValueError("Ambiguous checkpoint directories")
        manifest["checkpoint"] = str(models[-1])
        from ._microduck_run import checkpoint_digest

        manifest["checkpoint_relative"] = models[-1].relative_to(output).as_posix()
        manifest["checkpoint_sha256"] = checkpoint_digest(models[-1])
        execute(
            [
                str(python),
                "-I",
                str(worker_for(env)),
                "metrics",
                str(models[-1]),
                str(args.iterations),
                str(output / "metrics.validation.json"),
                str(start_iteration),
            ],
            "metrics",
        )
        manifest["status"] = "complete"
        manifest["phase"] = "complete"
    except KeyboardInterrupt as exc:
        manifest.update(
            status="interrupted", interrupt_signal=getattr(exc, "signum", signal.SIGINT)
        )
        raise
    except Exception as exc:
        record_worker_failure(output, manifest)
        manifest.update(
            status="failed", failed_phase=manifest.get("phase"), error=str(exc)
        )
        raise
    finally:
        finish_run(output, manifest, include_checkpoints=True)
    LOGGER.info("Microduck %s completed: %s", args.command, output)


def evaluation_inputs(args) -> tuple[Path, dict]:
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("--steps and --num-envs must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
    if not 0 <= args.seed < 2**32:
        raise ValueError("--seed must be between 0 and 2**32 - 1")
    if args.velocity is not None and not all(map(math.isfinite, args.velocity)):
        raise ValueError("--velocity values must be finite")
    curriculum_step = getattr(args, "curriculum_step", None)
    if curriculum_step is not None and curriculum_step < 0:
        raise ValueError("--curriculum-step must be nonnegative")
    onnx = args.onnx.expanduser().resolve() if args.onnx else None
    if onnx is not None and not onnx.is_file():
        raise ValueError(f"ONNX policy not found: {onnx}")
    options = {
        "velocity": args.velocity,
        "no_pushes": args.no_pushes,
        "onnx": str(onnx) if onnx else None,
        "curriculum_step": curriculum_step,
    }
    if getattr(args, "video", False):
        options["video"] = True
    return checkpoint, options


def evaluate_once(
    args, python: Path, identity: dict, env: dict[str, str], *, input_snapshot=None
) -> dict:
    env = training_environment(env)
    checkpoint, options = evaluation_inputs(args)
    criteria = evaluation_criteria(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(python),
        "-I",
        str(worker_for(env)),
        "evaluate",
        str(checkpoint),
        str(args.num_envs),
        str(args.steps),
        str(args.seed),
        str(output / "evaluation.json"),
        json.dumps(options, allow_nan=False),
    ]
    manifest = {
        "schema": 1,
        "launcher": launcher_identity(),
        "workflow": "evaluate",
        "python": str(python),
        "execution": {
            "headless": True,
            "viewer": None,
            "video": bool(options.get("video")),
            "mujoco_gl": env["MUJOCO_GL"],
        },
        "evaluation_options": options,
        "acceptance_criteria": criteria,
        "task": TASK,
        "source": identity,
        "status": "running",
        "commands": [
            [
                str(python),
                "-I",
                str(worker_for(env)),
                "check",
                str(output / "runtime.json"),
            ],
            command,
        ],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if getattr(args, "source_run", None) is not None:
        manifest["source_run"] = args.source_run
    write_json(output / "run.json", manifest)
    try:
        manifest["phase"] = "inputs"
        write_json(output / "run.json", manifest)
        if input_snapshot is None:
            input_snapshot = snapshot_inputs(
                output, checkpoint, options, getattr(args, "source_run", None)
            )
        verify_input_snapshot(input_snapshot)
        manifest["input_snapshot"] = input_snapshot
        manifest["checkpoint_sha256"] = input_snapshot["checkpoint"]["sha256"]
        command[4] = input_snapshot["checkpoint"]["path"]
        options["onnx"] = input_snapshot.get("onnx", {}).get("path")
        command[9] = json.dumps(options, allow_nan=False)
        manifest["phase"] = "snapshot"
        write_json(output / "run.json", manifest)
        env, source_snapshot = snapshot_runtime(output, identity, env)
        if source_snapshot is not None:
            manifest["implementation_snapshot"] = str(source_snapshot)
        manifest["commands"] = [
            snapshot_command(step, env) for step in manifest["commands"]
        ]
        write_json(output / "run.json", manifest)
        for index, step in enumerate(manifest["commands"]):
            phase = "check" if index == 0 else "evaluate"
            log_path = output / f"{index:02d}-{phase}.log"
            manifest.update(phase=phase, active_log=log_path.name)
            manifest.setdefault("logs", []).append(log_path.name)
            write_json(output / "run.json", manifest)
            run(
                step,
                cwd=output,
                env=env,
                log_path=log_path,
                echo=not getattr(args, "quiet", False),
            )
        if not (output / "evaluation.json").is_file():
            raise RuntimeError("Evaluation exited without a report")
        manifest["phase"] = "validate"
        write_json(output / "run.json", manifest)
        from ._microduck_reports import (
            decode_evaluation_json,
            validate_evaluation_report,
        )

        report = decode_evaluation_json((output / "evaluation.json").read_bytes())
        validate_evaluation_report(report, manifest)
        if criteria:
            save_acceptance(output, [report], criteria)
        manifest.update(status="complete", phase="complete")
    except EvaluationRejected as exc:
        manifest.update(status="rejected", error=str(exc))
        raise
    except KeyboardInterrupt as exc:
        manifest.update(
            status="interrupted", interrupt_signal=getattr(exc, "signum", signal.SIGINT)
        )
        raise
    except Exception as exc:
        record_worker_failure(output, manifest)
        manifest.update(
            status="failed", failed_phase=manifest.get("phase"), error=str(exc)
        )
        raise
    finally:
        finish_run(output, manifest)
    return report


def evaluate_run(args, python: Path, identity: dict, env: dict[str, str]) -> None:
    env = training_environment(env)
    seeds = getattr(args, "seeds", None)
    if seeds is None:
        evaluate_once(args, python, identity, env)
        return
    from ._microduck_reports import aggregate_evaluations

    checkpoint, options = evaluation_inputs(args)
    criteria = evaluation_criteria(args)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain distinct seeds")
    if any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("--seeds values must be between 0 and 2**32 - 1")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "launcher": launcher_identity(),
        "workflow": "evaluate_seeds",
        "python": str(python),
        "execution": {
            "headless": True,
            "viewer": None,
            "video": bool(options.get("video")),
            "mujoco_gl": env["MUJOCO_GL"],
        },
        "source": identity,
        "task": TASK,
        "evaluation_options": options,
        "acceptance_criteria": criteria,
        "seeds": seeds,
        "completed_seeds": [],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if getattr(args, "source_run", None) is not None:
        manifest["source_run"] = args.source_run
    write_json(output / "run.json", manifest)
    reports = []
    try:
        manifest["phase"] = "inputs"
        write_json(output / "run.json", manifest)
        input_snapshot = snapshot_inputs(
            output, checkpoint, options, getattr(args, "source_run", None)
        )
        manifest["input_snapshot"] = input_snapshot
        manifest["checkpoint_sha256"] = input_snapshot["checkpoint"]["sha256"]
        options["onnx"] = input_snapshot.get("onnx", {}).get("path")
        manifest["phase"] = "snapshot"
        write_json(output / "run.json", manifest)
        env, source_snapshot = snapshot_runtime(output, identity, env)
        if source_snapshot is not None:
            manifest["implementation_snapshot"] = str(source_snapshot)
        write_json(output / "run.json", manifest)
        for seed in seeds:
            child = argparse.Namespace(**vars(args))
            child.seed, child.seeds, child.output = seed, None, output / f"seed-{seed}"
            child.checkpoint = Path(input_snapshot["checkpoint"]["path"])
            child.onnx = Path(options["onnx"]) if options["onnx"] else None
            # Complete all requested seed runs before assessing the batch.
            for name in criteria:
                setattr(child, name, None)
            LOGGER.info(
                "Evaluating seed %s (%s/%s)", seed, len(reports) + 1, len(seeds)
            )
            manifest.update(phase="evaluate", active_seed=seed)
            write_json(output / "run.json", manifest)
            report = evaluate_once(
                child, python, identity, env, input_snapshot=input_snapshot
            )
            if reports:
                # Previous reports already passed; compare only the new seed.
                aggregate_evaluations([reports[0], report])
            reports.append(report)
            manifest["completed_seeds"].append(seed)
            write_json(output / "run.json", manifest)
        summary = aggregate_evaluations(reports)
        summary["reports"] = [f"seed-{seed}/evaluation.json" for seed in seeds]
        write_json(output / "summary.json", summary)
        save_acceptance(output, reports, criteria)
        manifest.update(status="complete", phase="complete")
        manifest.pop("active_seed", None)
    except EvaluationRejected as exc:
        manifest.update(status="rejected", error=str(exc))
        raise
    except KeyboardInterrupt as exc:
        manifest.update(
            status="interrupted", interrupt_signal=getattr(exc, "signum", signal.SIGINT)
        )
        raise
    except Exception as exc:
        manifest.update(
            status="failed", failed_phase=manifest.get("phase"), error=str(exc)
        )
        raise
    finally:
        finish_run(output, manifest)


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "setup",
        "check",
        "train",
        "evaluate",
        "play",
        "export",
        "assess",
        "compare",
        "progress",
    ):
        sub = commands.add_parser(command)
        if command not in ("assess", "compare"):
            sub.add_argument(
                "--repo",
                type=Path,
                help="Optional legacy upstream checkout; default: repository-owned walking task",
            )
            sub.add_argument(
                "--env-dir",
                type=Path,
                help="Defaults to .cache/microduck-native-venv (legacy: .cache/microduck-venv)",
            )
        elif command == "assess":
            sub.add_argument(
                "--run",
                type=Path,
                required=True,
                help="Completed single-seed or batch evaluation directory",
            )
        else:
            sub.add_argument("--before", type=Path, required=True)
            sub.add_argument("--after", type=Path, required=True)
        sub.add_argument(
            "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO"
        )
        sub.add_argument("--log-json", action="store_true")
        if command == "progress":
            sub.add_argument(
                "--run", type=Path, required=True, help="Training run directory"
            )
        if command in ("train", "evaluate", "export", "assess", "compare"):
            sub.add_argument("--output", type=Path, required=True)
        if command in ("train", "evaluate", "export"):
            sub.add_argument(
                "--quiet",
                action="store_true",
                help="Save worker output to stage logs without echoing it to the terminal",
            )
        if command == "train":
            sub.add_argument(
                "--seed",
                type=int,
                default=0,
                help="Training/reset random seed (default: 0)",
            )
        if command in ("train", "evaluate", "export"):
            sub.add_argument(
                "--headless",
                action="store_true",
                default=True,
                help="Run without a desktop or viewer (always enabled); evaluation video uses offscreen EGL",
            )
        if command == "train":
            sub.add_argument("--num-envs", type=int, default=4096)
            sub.add_argument("--iterations", type=int, required=True)
            resume_source = sub.add_mutually_exclusive_group()
            resume_source.add_argument(
                "--resume",
                type=Path,
                help="Checkpoint; iterations are additional updates",
            )
            resume_source.add_argument(
                "--resume-run",
                type=Path,
                help="Completed, interrupted or failed training run; uses recorded checkpoints and adds iterations",
            )
        if command == "evaluate":
            sub.add_argument("--num-envs", type=int, default=16)
            sub.add_argument("--steps", type=int, default=250)
            seed_group = sub.add_mutually_exclusive_group()
            seed_group.add_argument("--seed", type=int, default=0)
            seed_group.add_argument(
                "--seeds",
                type=int,
                nargs="+",
                help="Evaluate distinct seeds sequentially and write summary.json",
            )
            sub.add_argument(
                "--velocity",
                type=float,
                nargs=3,
                metavar=("VX", "VY", "WZ"),
                help="Fixed body-frame velocity (m/s, m/s, rad/s); neutral head/body pose",
            )
            sub.add_argument(
                "--no-pushes", action="store_true", help="Disable push events"
            )
            sub.add_argument(
                "--onnx",
                type=Path,
                help="Compare ONNX actions with the checkpoint on live observations",
            )
            sub.add_argument(
                "--curriculum-step",
                type=int,
                help="Start evaluation curricula at this counter instead of the checkpoint counter",
            )
            sub.add_argument(
                "--video",
                action="store_true",
                help="Record environment 0 to policy.mp4 during each evaluation seed",
            )
        if command in ("evaluate", "assess"):
            sub.add_argument(
                "--max-planar-rmse",
                type=float,
                help="Maximum planar velocity RMSE in m/s, per seed",
            )
            sub.add_argument(
                "--max-yaw-rmse",
                type=float,
                help="Maximum yaw-rate RMSE in rad/s, per seed",
            )
            sub.add_argument(
                "--min-survival-fraction",
                type=float,
                help="Minimum fraction of initial episodes surviving the full horizon, per seed",
            )
        if command in ("play", "evaluate", "export"):
            checkpoint_source = sub.add_mutually_exclusive_group(required=True)
            checkpoint_source.add_argument("--checkpoint", type=Path)
            checkpoint_source.add_argument(
                "--run", type=Path, help="Completed training run"
            )
        if command == "play":
            sub.add_argument("--viewer", choices=("native", "viser"), default="native")
            sub.add_argument(
                "--steps",
                type=int,
                help="Native viewer: stop after this many simulation steps",
            )
            sub.add_argument(
                "--seed", type=int, default=0, help="Native viewer environment seed"
            )
    args = parser.parse_args(argv)
    setup_logging(args.log_level, with_json_format=args.log_json)
    try:
        if args.command == "assess":
            assess_run(args)
            return
        if args.command == "compare":
            compare_runs(args)
            return
        native = args.repo is None
        repo = args.repo.expanduser().resolve() if args.repo else None
        env_dir = (
            (
                args.env_dir
                or Path(
                    ".cache/microduck-native-venv"
                    if native
                    else ".cache/microduck-venv"
                )
            )
            .expanduser()
            .resolve()
        )
        python = env_dir / "bin/python"
        env = child_environment()
        if args.command == "progress":
            if not python.is_file():
                raise ValueError("Microduck environment is not ready; run setup first")
            run(
                [
                    str(python),
                    "-I",
                    str(WORKER),
                    "progress",
                    str(args.run.expanduser().resolve()),
                ],
                cwd=Path.cwd(),
                env=env,
            )
            return
        if native:
            from ._microduck_native import REQUIREMENTS, dependency_identity
            from ._microduck_native import source_identity as native_identity

            identity = native_identity()
            expected_stamp = dependency_identity()
        else:
            identity = source_identity(repo)
            expected_stamp = identity
        if native:
            env["EF_MICRODUCK_NATIVE"] = "1"
        stamp = env_dir / "embodiedforge-source.json"
        if args.command == "setup":
            stamp.unlink(missing_ok=True)
            if native:
                if not python.is_file():
                    run(
                        ["uv", "venv", "--python", "3.12", str(env_dir)],
                        cwd=Path.cwd(),
                        env=env,
                    )
                run(
                    [
                        "uv",
                        "pip",
                        "sync",
                        "--python",
                        str(python),
                        str(REQUIREMENTS),
                    ],
                    cwd=Path.cwd(),
                    env=env,
                )
                write_json(stamp, expected_stamp)
                LOGGER.info("Native Microduck environment ready: %s", env_dir)
                return
            env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
            run(
                ["uv", "sync", "--locked", "--python", "3.12", "--project", str(repo)],
                cwd=Path.cwd(),
                env=env,
            )
            if source_identity(repo) != identity:
                raise RuntimeError(
                    "Upstream source changed during setup; run setup again"
                )
            write_json(stamp, identity)
            LOGGER.info("Microduck environment ready: %s", env_dir)
            return
        if not python.is_file() or not stamp.is_file():
            raise ValueError("Microduck environment is not ready; run setup first")
        if json.loads(stamp.read_text()) != expected_stamp:
            raise ValueError(
                "Environment dependencies differ from the requested implementation; run setup again"
            )
        selected_run = (
            getattr(args, "resume_run", None)
            if args.command == "train"
            else getattr(args, "run", None)
            if args.command in ("play", "evaluate", "export")
            else None
        )
        if selected_run is not None:
            from ._microduck_run import resolve_run_checkpoint

            checkpoint, args.source_run = resolve_run_checkpoint(
                selected_run, identity, TASK, allow_incomplete=args.command == "train"
            )
            if args.source_run["verification"] == "unverified_legacy":
                LOGGER.warning(
                    "Legacy run has no saved checkpoint hash; recording current bytes only"
                )
            LOGGER.info("Selected recorded checkpoint: %s", checkpoint)
            if args.command == "train":
                args.resume = checkpoint
            else:
                args.checkpoint = checkpoint
        if args.command == "check":
            run(
                [str(python), "-I", str(worker_for(env)), "check"],
                cwd=Path.cwd(),
                env=env,
            )
        elif args.command == "train":
            train_run(args, python, identity, env)
        elif args.command == "evaluate":
            evaluate_run(args, python, identity, env)
        else:
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint not found: {checkpoint}")
            if args.command == "play":
                if args.steps is not None and args.steps <= 0:
                    raise ValueError("--steps must be positive")
                command = [
                    str(python),
                    "-I",
                    str(worker_for(env)),
                    "play",
                    str(checkpoint),
                    args.viewer,
                    str(args.steps) if args.steps is not None else "none",
                    str(args.seed),
                ]
                if args.viewer == "viser":
                    if args.steps is not None or args.seed != 0:
                        raise ValueError(
                            "--steps and --seed overrides require --viewer native"
                        )
                    # Preserve upstream Viser's checkpoint browser and hot swapping.
                    command = [
                        str(python),
                        "-I",
                        "-m",
                        "mjlab.scripts.play",
                        TASK,
                        "--checkpoint-file",
                        str(checkpoint),
                        "--num-envs",
                        "1",
                        "--viewer",
                        "viser",
                    ]
                run(
                    native_command(command)
                    if env.get("EF_MICRODUCK_NATIVE") == "1"
                    else command,
                    cwd=Path.cwd(),
                    env=env,
                )
            else:
                export_run(args, python, identity, env)
    except KeyboardInterrupt as exc:
        signum = getattr(exc, "signum", signal.SIGINT)
        LOGGER.info("Microduck interrupted by signal %s", signum)
        raise SystemExit(128 + signum) from None
    except EvaluationRejected as exc:
        LOGGER.error("Microduck acceptance criteria not met: %s", exc)
        raise SystemExit(3) from None
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        LOGGER.error("Microduck failed: %s", exc)
        raise SystemExit(1) from None


def main(argv: list[str] | None = None) -> None:
    from ._microduck_process import ProcessInterrupted, termination_signals

    try:
        with termination_signals():
            _main(argv)
    except ProcessInterrupted as exc:
        raise SystemExit(128 + exc.signum) from None


if __name__ == "__main__":
    main()
