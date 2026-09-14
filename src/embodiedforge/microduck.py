"""Run the pinned upstream Microduck recipe in an isolated Python environment.

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
from datetime import datetime, timezone
from pathlib import Path

from .logging import get_logger, setup_logging

REVISION = "53b8971b61baf5b7f3c16d135dd7cac37623de4b"
TASK = "Mjlab-Velocity-Flat-MicroDuck"
WORKER = Path(__file__).with_name("_microduck_worker.py")
LOGGER = get_logger("microduck")


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
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)


def compare_runs(args) -> None:
    from ._microduck_reports import compare_evaluations, load_evaluation_run

    before, before_info, before_inputs = load_evaluation_run(args.before)
    after, after_info, after_inputs = load_evaluation_run(args.after)
    for key in ("revision", "uv_lock_sha256"):
        if not (before_info.get("source") or {}).get(key) or before_info["source"][
            key
        ] != (after_info.get("source") or {}).get(key):
            raise ValueError(f"Comparison upstream source differs or is missing: {key}")
    comparison = compare_evaluations(before, after)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "workflow": "compare",
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
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)


def child_environment() -> dict[str, str]:
    env = os.environ.copy()
    # The caller may be running from ef/ef-viewer or a source PYTHONPATH.
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
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


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    LOGGER.info("Running %s", command)
    # A separate process group receives one forwarded SIGINT. subprocess.run()
    # otherwise kills an interrupted child before its render thread can finish.
    with subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True) as process:

        def send(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        try:
            returncode = process.wait()
        except KeyboardInterrupt:
            send(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                send(signal.SIGKILL)
                process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def training_command(python: Path, *, num_envs: int, iterations: int) -> list[str]:
    if num_envs <= 0 or iterations <= 0:
        raise ValueError("Environment and iteration counts must be positive")
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
        "0",
        "--enable-nan-guard",
        "True",
    ]
    return command


def export_command(python: Path, checkpoint: Path, output: Path) -> list[str]:
    return [
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


def checkpoints(output: Path) -> list[Path]:
    return sorted(
        output.glob("logs/rsl_rl/microduck/*/model_*.pt"),
        key=lambda path: int(path.stem.removeprefix("model_")),
    )


def train_run(args, python: Path, identity: dict, env: dict[str, str]) -> None:
    output = args.output.expanduser().resolve()
    smoke = args.command == "smoke"
    resume = getattr(args, "resume", None)
    if resume is not None:
        resume = resume.expanduser().resolve()
        if not resume.is_file():
            raise ValueError(f"Resume checkpoint not found: {resume}")
    command = training_command(
        python,
        num_envs=64 if smoke else args.num_envs,
        iterations=5 if smoke else args.iterations,
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "workflow": args.command,
        "task": TASK,
        "source": identity,
        "python": str(python),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "commands": [],
    }

    def execute(command: list[str]) -> None:
        manifest["commands"].append(command)
        write_json(output / "run.json", manifest)
        run(command, cwd=output, env=env)

    try:
        execute([str(python), "-I", str(WORKER), "check", str(output / "runtime.json")])
        start_iteration = 0
        if resume is not None:
            snapshot = output / "logs/rsl_rl/microduck/resume_source/checkpoint.pt"
            snapshot.parent.mkdir(parents=True)
            shutil.copyfile(resume, snapshot)
            execute(
                [
                    str(python),
                    "-I",
                    str(WORKER),
                    "checkpoint",
                    str(snapshot),
                    str(output / "resume.json"),
                ]
            )
            metadata = json.loads((output / "resume.json").read_text())
            start_iteration = metadata["iteration"]
            # The worker adds startup/reset events to the upstream TrainConfig.
            # Its startup restores curricula before the wrapper resets the env.
            command = [
                str(python),
                "-I",
                str(WORKER),
                "resume-train",
                str(snapshot),
                "--env.scene.num-envs",
                str(args.num_envs),
                "--agent.max-iterations",
                str(args.iterations),
            ]
            manifest["resume"] = {
                "original": str(resume),
                "snapshot": str(snapshot),
                **metadata,
            }
        execute(command)
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
        manifest["checkpoint"] = str(models[-1])
        execute(
            [
                str(python),
                "-I",
                str(WORKER),
                "metrics",
                str(models[-1]),
                str(5 if smoke else args.iterations),
                str(output / "metrics.validation.json"),
                str(start_iteration),
            ]
        )
        if smoke:
            policy = output / "policy.onnx"
            execute(export_command(python, models[-1], policy))
            execute([str(python), "-I", str(WORKER), "onnx", str(policy)])
            manifest["onnx"] = str(policy)
        manifest["status"] = "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["checkpoints"] = [str(path) for path in checkpoints(output)]
        write_json(output / "run.json", manifest)
    LOGGER.info("Microduck %s completed: %s", args.command, output)


def evaluation_inputs(args) -> tuple[Path, dict]:
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("--steps and --num-envs must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint}")
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


def evaluate_once(args, python: Path, identity: dict, env: dict[str, str]) -> None:
    checkpoint, options = evaluation_inputs(args)
    criteria = evaluation_criteria(args)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [
        str(python),
        "-I",
        str(WORKER),
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
        "workflow": "evaluate",
        "evaluation_options": options,
        "acceptance_criteria": criteria,
        "task": TASK,
        "source": identity,
        "status": "running",
        "commands": [
            [str(python), "-I", str(WORKER), "check", str(output / "runtime.json")],
            command,
        ],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run.json", manifest)
    try:
        for step in manifest["commands"]:
            run(step, cwd=output, env=env)
        if not (output / "evaluation.json").is_file():
            raise RuntimeError("Evaluation exited without a report")
        if criteria:
            save_acceptance(
                output, [json.loads((output / "evaluation.json").read_text())], criteria
            )
        manifest["status"] = "complete"
    except EvaluationRejected as exc:
        manifest.update(status="rejected", error=str(exc))
        raise
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)


def evaluate_run(args, python: Path, identity: dict, env: dict[str, str]) -> None:
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
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "workflow": "evaluate_seeds",
        "source": identity,
        "task": TASK,
        "checkpoint_sha256": digest.hexdigest(),
        "evaluation_options": options,
        "acceptance_criteria": criteria,
        "seeds": seeds,
        "completed_seeds": [],
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "run.json", manifest)
    reports = []
    try:
        for seed in seeds:
            child = argparse.Namespace(**vars(args))
            child.seed, child.seeds, child.output = seed, None, output / f"seed-{seed}"
            # Complete all requested seed runs before assessing the batch.
            for name in criteria:
                setattr(child, name, None)
            LOGGER.info(
                "Evaluating seed %s (%s/%s)", seed, len(reports) + 1, len(seeds)
            )
            evaluate_once(child, python, identity, env)
            report = json.loads((child.output / "evaluation.json").read_text())
            if (
                report.get("checkpoint_metadata", {}).get("sha256")
                != digest.hexdigest()
            ):
                raise RuntimeError("Checkpoint changed during multi-seed evaluation")
            if report.get("seed") != seed:
                raise RuntimeError(
                    "Evaluation report seed does not match requested seed"
                )
            reports.append(report)
            aggregate_evaluations(reports)  # Reject incompatible/invalid reports early.
            manifest["completed_seeds"].append(seed)
            write_json(output / "run.json", manifest)
        summary = aggregate_evaluations(reports)
        summary["reports"] = [f"seed-{seed}/evaluation.json" for seed in seeds]
        write_json(output / "summary.json", summary)
        save_acceptance(output, reports, criteria)
        manifest["status"] = "complete"
    except EvaluationRejected as exc:
        manifest.update(status="rejected", error=str(exc))
        raise
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "setup",
        "check",
        "smoke",
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
                "--repo", type=Path, required=True, help="Clean upstream checkout"
            )
            sub.add_argument(
                "--env-dir", type=Path, default=Path(".cache/microduck-venv")
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
        if command in ("smoke", "train", "evaluate", "export", "assess", "compare"):
            sub.add_argument("--output", type=Path, required=True)
        if command == "train":
            sub.add_argument("--num-envs", type=int, default=4096)
            sub.add_argument("--iterations", type=int, required=True)
            sub.add_argument(
                "--resume",
                type=Path,
                help="Checkpoint; iterations are additional updates",
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
            sub.add_argument("--checkpoint", type=Path, required=True)
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
        repo = args.repo.expanduser().resolve()
        env_dir = args.env_dir.expanduser().resolve()
        identity = source_identity(repo)
        python = env_dir / "bin/python"
        env = child_environment()
        stamp = env_dir / "embodiedforge-source.json"
        if args.command == "setup":
            stamp.unlink(missing_ok=True)
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
        if json.loads(stamp.read_text()) != identity:
            raise ValueError(
                "Environment source differs from checkout; run setup again"
            )
        if args.command == "check":
            run([str(python), "-I", str(WORKER), "check"], cwd=Path.cwd(), env=env)
        elif args.command == "progress":
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
        elif args.command in ("smoke", "train"):
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
                    str(WORKER),
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
                run(command, cwd=Path.cwd(), env=env)
            else:
                output = args.output.expanduser().resolve()
                if output.exists():
                    raise FileExistsError(f"Refusing to overwrite {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                run(export_command(python, checkpoint, output), cwd=Path.cwd(), env=env)
                run(
                    [str(python), "-I", str(WORKER), "onnx", str(output)],
                    cwd=Path.cwd(),
                    env=env,
                )
    except KeyboardInterrupt:
        LOGGER.info("Microduck interrupted")
        raise SystemExit(130) from None
    except EvaluationRejected as exc:
        LOGGER.error("Microduck acceptance criteria not met: %s", exc)
        raise SystemExit(3) from None
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        LOGGER.error("Microduck failed: %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
