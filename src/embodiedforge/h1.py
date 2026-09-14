"""Isolated IsaacLab H1 training; no physics or learning SDK imports here."""

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

REVISION = "2e44ddb2e19536579140496023b5ccb060bc4152"
TASK = "Isaac-Velocity-Flat-H1-v0"
PHYSICS = "newton_mjwarp"
WORKER = Path(__file__).with_name("_h1_worker.py")
EVALUATOR = Path(__file__).with_name("_h1_evaluate.py")
LOGGER = get_logger("h1")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def child_environment(environment: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update(
        CONDA_PREFIX=str(environment),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
    )
    env["PATH"] = str(environment / "bin") + os.pathsep + env.get("PATH", "")
    return env


def source_identity(repo: Path) -> dict:
    for relative in (
        "isaaclab.sh",
        "scripts/reinforcement_learning/rsl_rl/train_rsl_rl.py",
    ):
        if not (repo / relative).is_file():
            raise ValueError(f"Missing IsaacLab file: {repo / relative}")
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected IsaacLab revision {REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if dirty:
        raise ValueError("IsaacLab checkout has local changes; use a clean checkout")
    return {"repo": str(repo), "revision": revision}


def run_process(
    command: list[str], *, cwd: Path, env: dict, timeout: float | None
) -> None:
    """Forward interruption to the whole child group and retain native output."""
    LOGGER.info("Running %s; output: %s", command, cwd / "console.log")
    with (
        (cwd / "console.log").open("a") as log,
        subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            returncode = process.wait(timeout=timeout)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except (subprocess.TimeoutExpired, KeyboardInterrupt, ProcessLookupError):
                pass
            finally:
                # Also remove descendants if the wrapper exited before its worker.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def resume_input(run: Path) -> tuple[Path, dict]:
    manifest = json.loads((run / "run.json").read_text())
    if (
        manifest.get("schema") != 1
        or manifest.get("workflow") != "h1_train"
        or manifest.get("status") != "complete"
        or manifest.get("task") != TASK
        or manifest.get("physics") != PHYSICS
        or manifest.get("source", {}).get("revision") != REVISION
    ):
        raise ValueError("Resume requires a completed H1 run with the same recipe")
    artifact = manifest["result"]
    checkpoint = (run / artifact["checkpoint"]).resolve()
    if not checkpoint.is_relative_to(run) or not checkpoint.is_file():
        raise ValueError("Resume checkpoint is missing or outside its run")
    if sha256(checkpoint) != artifact["checkpoint_sha256"]:
        raise ValueError("Resume checkpoint SHA256 differs from its recorded value")
    return checkpoint, artifact


def train(args: argparse.Namespace) -> None:
    repo = args.repo.expanduser().resolve()
    environment = args.environment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source = source_identity(repo)
    if not (environment / "bin/python").is_file():
        raise ValueError(f"Python environment not found: {environment}")
    previous = None
    if args.resume_run:
        previous = resume_input(args.resume_run.expanduser().resolve())
    output.mkdir(parents=True, exist_ok=False)
    env = child_environment(environment)
    wrapper = str(repo / "isaaclab.sh")
    manifest = {
        "schema": 1,
        "workflow": "h1_train",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "launcher_sha256": sha256(Path(__file__)),
        "worker_sha256": sha256(WORKER),
        "task": TASK,
        "physics": PHYSICS,
        "environment": str(environment),
        "num_envs": args.num_envs,
        "requested_updates": args.updates,
        "seed": args.seed,
        "timeout_seconds": args.timeout,
        "commands": [],
    }
    write_json(output / "run.json", manifest)

    def execute(command: list[str], timeout: float | None) -> None:
        manifest["commands"].append(command)
        write_json(output / "run.json", manifest)
        run_process(command, cwd=output, env=env, timeout=timeout)

    try:
        checkpoint_args = []
        start_iteration = 0
        if previous:
            checkpoint, artifact = previous
            # The upstream resolver searches run names as regexes under log_root.
            target = output / "logs/rsl_rl/h1_flat/_resume_input/model.pt"
            target.parent.mkdir(parents=True)
            shutil.copyfile(checkpoint, target)
            if sha256(target) != artifact["checkpoint_sha256"]:
                raise ValueError("Resume checkpoint changed while being copied")
            start_iteration = artifact["checkpoint_iteration"]
            manifest["resume"] = {
                "run": str(args.resume_run.expanduser().resolve()),
                "checkpoint_sha256": artifact["checkpoint_sha256"],
                "iteration": start_iteration,
            }
            checkpoint_args = ["--checkpoint", str(target)]
        execute(
            [
                wrapper,
                "-p",
                "-P",
                str(WORKER),
                "preflight",
                "--repo",
                str(repo),
                "--output",
                str(output / "runtime.json"),
                *checkpoint_args,
            ],
            60,
        )
        manifest["runtime"] = json.loads((output / "runtime.json").read_text())
        command = [
            wrapper,
            "train",
            "--rl_library",
            "rsl_rl",
            "--task",
            TASK,
            f"physics={PHYSICS}",
            "--visualizer",
            "none",
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.updates),
            "--seed",
            str(args.seed),
            "--logger",
            "tensorboard",
            "--run_name",
            "embodiedforge",
            "agent.obs_groups={'actor':['policy'],'critic':['policy']}",
        ]
        if previous:
            if manifest["runtime"]["checkpoint_iteration"] != start_iteration:
                raise ValueError("Resume iteration differs from its recorded value")
            command.extend(
                [
                    "--resume",
                    "--load_run",
                    "^_resume_input$",
                    "--checkpoint",
                    "^model\\.pt$",
                ]
            )
        execute(command, args.timeout)
        execute(
            [
                wrapper,
                "-p",
                "-P",
                str(WORKER),
                "verify",
                "--run",
                str(output),
                "--start",
                str(start_iteration),
                "--updates",
                str(args.updates),
                "--output",
                str(output / "verification.json"),
            ],
            60,
        )
        manifest["result"] = json.loads((output / "verification.json").read_text())
        manifest["status"] = "complete"
        LOGGER.info("H1 training verified: %s", output)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except subprocess.TimeoutExpired as exc:
        manifest.update(status="timed_out", error=str(exc))
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def evaluate(args: argparse.Namespace) -> None:
    from ._h1_metrics import BASIC_COMMANDS, assess

    repo = args.repo.expanduser().resolve()
    environment = args.environment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source = source_identity(repo)
    checkpoint, artifact = resume_input(args.run.expanduser().resolve())
    if not (environment / "bin/python").is_file():
        raise ValueError(f"Python environment not found: {environment}")
    criteria = {
        name: getattr(args, name)
        for name in ("max_planar_rmse", "max_yaw_rmse", "min_survival_fraction")
        if getattr(args, name) is not None
    }
    suite = getattr(args, "suite", None)
    cases = BASIC_COMMANDS if suite else {"custom": args.velocity}
    record_motion = getattr(args, "record_motion", False)
    record_env = getattr(args, "record_env", 0)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": 1,
        "workflow": "h1_evaluate",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "input_run": str(args.run.expanduser().resolve()),
        "checkpoint_sha256": artifact["checkpoint_sha256"],
        "checkpoint_iteration": artifact["checkpoint_iteration"],
        "evaluator_sha256": sha256(EVALUATOR),
        "metrics_sha256": sha256(Path(__file__).with_name("_h1_metrics.py")),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seeds": args.seeds,
        "velocity_command": args.velocity,
        "suite": suite,
        "cases": cases,
        "record_motion": record_motion,
        "record_env": record_env if record_motion else None,
        "motion_code_sha256": sha256(Path(__file__).with_name("_h1_motion.py")),
        "criteria": criteria,
        "timeout_seconds_per_seed": args.timeout,
        "commands": [],
        "reports": [],
    }
    write_json(output / "run.json", manifest)
    try:
        target = output / "input.pt"
        shutil.copyfile(checkpoint, target)
        if sha256(target) != artifact["checkpoint_sha256"]:
            raise ValueError("Input checkpoint changed while being copied")
        reports = []
        for seed in args.seeds:
            report_path = output / f"evaluation-seed-{seed}.json"
            command = [
                str(repo / "isaaclab.sh"),
                "-p",
                "-P",
                str(EVALUATOR),
                "--repo",
                str(repo),
                "--checkpoint",
                str(target),
                "--output",
                str(report_path),
                "--num-envs",
                str(args.num_envs),
                "--steps",
                str(args.steps),
                "--seed",
                str(seed),
            ]
            command.extend(
                ["--suite", suite]
                if suite
                else ["--velocity", *map(str, args.velocity)]
            )
            if record_motion:
                command.extend(["--record-motion", "--record-env", str(record_env)])
            manifest["commands"].append(command)
            write_json(output / "run.json", manifest)
            run_process(
                command,
                cwd=output,
                env=child_environment(environment),
                timeout=args.timeout,
            )
            payload = json.loads(report_path.read_text())
            if suite and (payload.get("suite") != suite or payload.get("seed") != seed):
                raise ValueError(
                    "Evaluation suite report does not match its requested inputs"
                )
            case_reports = payload["cases"] if suite else [payload]
            if len(case_reports) != len(cases):
                raise ValueError("Incomplete evaluation suite")
            for (case, velocity), report in zip(
                cases.items(), case_reports, strict=True
            ):
                if (
                    report["seed"] != seed
                    or report.get("case", "custom") != case
                    or report["num_envs"] != args.num_envs
                    or report["steps_limit"] != args.steps
                    or report["velocity_command"] != velocity
                    or report["task"] != "Isaac-Velocity-Flat-H1-Play-v0"
                    or report["physics"] != PHYSICS
                    or report["runtime"]["checkpoint_iteration"]
                    != artifact["checkpoint_iteration"]
                ):
                    raise ValueError(
                        "Evaluation report does not match its requested inputs"
                    )
                reports.append(report)
                if record_motion:
                    from ._h1_motion import load_motion, render_motion

                    motion_path = (output / report["motion"]["path"]).resolve()
                    if not motion_path.is_relative_to(output):
                        raise ValueError("Motion file is outside its run directory")
                    motion = load_motion(motion_path)
                    metadata = motion["metadata"]
                    if (
                        metadata["case"] != case
                        or metadata["seed"] != seed
                        or metadata["env_id"] != record_env
                        or metadata["velocity_command"] != velocity
                        or metadata["quaternion_order"] != "xyzw"
                        or metadata["dt"] != report["dt"]
                        or len(motion["time"])
                        != report["per_env"]["observed_steps"][record_env]
                        or motion["terminated"]
                        != report["per_env"]["terminated"][record_env]
                        or motion["truncated"]
                        != report["per_env"]["truncated"][record_env]
                    ):
                        raise ValueError(
                            "Motion does not match the evaluated first episode"
                        )
                    html_path = motion_path.with_suffix(".html")
                    render_motion(motion_path, html_path)
                    manifest.setdefault("motion", []).append(
                        {
                            "path": motion_path.name,
                            "sha256": sha256(motion_path),
                            "replay": html_path.name,
                        }
                    )
            manifest["reports"].append(report_path.name)
            write_json(output / "run.json", manifest)
        assessment = assess(reports, criteria)
        write_json(output / "acceptance.json", assessment)
        manifest["acceptance"] = assessment
        manifest["status"] = "rejected" if assessment["passed"] is False else "complete"
        if manifest["status"] == "rejected":
            LOGGER.error(
                "H1 policy missed acceptance criteria: %s", output / "acceptance.json"
            )
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        raise
    except subprocess.TimeoutExpired as exc:
        manifest.update(status="timed_out", error=str(exc))
        raise
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)
    if manifest["status"] == "rejected":
        raise SystemExit(2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", help="Train or resume the pinned H1 flat recipe"
    )
    training.add_argument(
        "--repo", type=Path, default=Path("/home/ubuntu/workspace/3rdparty/IsaacLab")
    )
    training.add_argument(
        "--environment",
        type=Path,
        default=Path("/home/ubuntu/miniconda3/envs/isaaclab"),
    )
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--num-envs", type=positive_int, default=64)
    training.add_argument(
        "--updates",
        type=positive_int,
        default=5,
        help="Additional PPO updates in this invocation",
    )
    training.add_argument("--seed", type=int, default=0)
    training.add_argument(
        "--timeout",
        type=positive_int,
        help="Training wall-clock limit in seconds, including initialization",
    )
    training.add_argument(
        "--resume-run", type=Path, help="Completed managed H1 run to resume"
    )
    status = commands.add_parser(
        "status", help="Print a run manifest without loading training SDKs"
    )
    status.add_argument("--run", type=Path, required=True)
    replay = commands.add_parser(
        "replay", help="Create an offline HTML replay from a motion recording"
    )
    replay.add_argument("--motion", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser(
        "evaluate", help="Bounded fixed-command policy evaluation"
    )
    selection = evaluation.add_mutually_exclusive_group()
    selection.add_argument(
        "--suite",
        choices=["basic"],
        help="Paired stand, forward and left/right turning trials",
    )
    evaluation.add_argument(
        "--repo", type=Path, default=Path("/home/ubuntu/workspace/3rdparty/IsaacLab")
    )
    evaluation.add_argument(
        "--environment",
        type=Path,
        default=Path("/home/ubuntu/miniconda3/envs/isaaclab"),
    )
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--num-envs", type=positive_int, default=32)
    evaluation.add_argument("--steps", type=positive_int, default=500)
    evaluation.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    evaluation.add_argument(
        "--record-motion",
        action="store_true",
        help="Save rigid-body motion and offline HTML replay",
    )
    evaluation.add_argument(
        "--record-env", type=int, default=0, help="Environment to record (default: 0)"
    )
    selection.add_argument(
        "--velocity",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "YAW"),
    )
    evaluation.add_argument("--timeout", type=positive_int, default=240)
    for flag in ("--max-planar-rmse", "--max-yaw-rmse", "--min-survival-fraction"):
        evaluation.add_argument(flag, type=float)
    args = parser.parse_args(argv)
    setup_logging("INFO")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        if args.command == "status":
            print(json.dumps(json.loads((args.run / "run.json").read_text()), indent=2))
        elif args.command == "replay":
            from ._h1_motion import render_motion

            render_motion(args.motion.expanduser(), args.output.expanduser())
        elif args.command == "evaluate":
            if not 0 <= args.record_env < args.num_envs:
                parser.error("--record-env must identify an existing environment")
            if args.record_env and not args.record_motion:
                parser.error("--record-env requires --record-motion")
            if args.velocity is None and not args.suite:
                args.velocity = [0.5, 0.0, 0.0]
            if len(set(args.seeds)) != len(args.seeds) or any(
                seed < 0 for seed in args.seeds
            ):
                parser.error("--seeds must be distinct nonnegative integers")
            if args.velocity and not all(
                math.isfinite(value) for value in args.velocity
            ):
                parser.error("--velocity must be finite")
            for name in ("max_planar_rmse", "max_yaw_rmse", "min_survival_fraction"):
                value = getattr(args, name)
                if value is not None and (
                    not math.isfinite(value)
                    or value < 0
                    or (name == "min_survival_fraction" and value > 1)
                ):
                    parser.error(f"Invalid acceptance limit: {name}")
            evaluate(args)
        else:
            if args.seed < 0:
                parser.error("--seed must be nonnegative for reproducible runs")
            train(args)
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
