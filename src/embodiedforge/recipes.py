"""Pinned external task recipes, isolated from the lightweight core runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .locomotion.go1_config import COMMAND_PROFILES as GO1_COMMAND_PROFILES
from .locomotion.go1_config import REWARD_PROFILES as GO1_REWARD_PROFILES
from .logging import get_logger, setup_logging
from .runtime.artifacts import write_json as write_json
from .runtime.hashing import file_sha256 as sha256
from .runtime.process import run_process as _run_process
from .runtime.provenance import snapshot_implementation as snapshot_implementation
from .runtime.provenance import validate_snapshot as validate_snapshot
from .runtime.records import run_identity, run_lifecycle
from .workflows.go1_settings import (
    resolve_go1_command_profile as resolve_go1_command_profile,
)
from .workflows.go1_settings import (
    resolve_go1_learning_rate as resolve_go1_learning_rate,
)
from .workflows.go1_settings import (
    resolve_go1_reward_profile as resolve_go1_reward_profile,
)
from .workflows.go1_settings import resolve_go1_semantics as resolve_go1_semantics
from .workflows.recipe_artifacts import checkpoint_input as checkpoint_input
from .workflows.recipe_catalog import GO1_STANDALONE_SOURCE as GO1_STANDALONE_SOURCE
from .workflows.recipe_catalog import RECIPES as RECIPES
from .workflows.recipe_catalog import SOURCES as SOURCES
from .workflows.recipe_catalog import SWITCHING_SUITES as SWITCHING_SUITES
from .workflows.recipe_progress import go1_training_progress as go1_training_progress
from .workflows.recipe_validation import EvaluationRejected as EvaluationRejected
from .workflows.recipe_validation import assess as assess
from .workflows.recipe_validation import criteria_for as criteria_for
from .workflows.recipe_validation import validate_result as validate_result

WORKER = Path(__file__).with_name("_recipe_worker.py")
LOGGER = get_logger("recipes")


def source_identity(repo: Path, name: str) -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repo), *args], text=True
        ).strip()

    revision = git("rev-parse", "HEAD")
    if revision != SOURCES[name]:
        raise ValueError(f"Expected {name} revision {SOURCES[name]}, found {revision}")
    if git("status", "--porcelain"):
        raise ValueError(f"{name} checkout must be clean")
    files = git("ls-files", "-z").split("\0")
    hashes = {}
    for relative in filter(None, files):
        path = repo / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Unsupported source entry: {relative}")
        hashes[relative] = sha256(path)
    return {"name": name, "revision": revision, "repo": str(repo), "files": hashes}


def child_environment(project: Path | None) -> dict:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
        MUJOCO_GL="egl",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    if project is None:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PATH"] = (
            str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
        )
        return env
    env["VIRTUAL_ENV"] = str(project / ".venv")
    env["MENAGERIE_CACHE_DIR"] = str(project / "assets-cache")
    if project.name == "mjbatch":
        env["CUDA_VISIBLE_DEVICES"] = ""
    env["PATH"] = str(project / ".venv/bin") + os.pathsep + env.get("PATH", "")
    return env


def run_process(command, output: Path, env: dict, timeout: float | None):
    LOGGER.info("Running recipe; full output: %s", output / "console.log")
    _run_process(
        command,
        cwd=output,
        env=env,
        log_path=output / "console.log",
        echo=False,
        timeout=timeout,
        append_log=True,
    )


def setup(args):
    name = args.source
    repo = (args.repo or Path("/home/ubuntu/workspace/3rdparty") / name).resolve()
    source = source_identity(repo, name)
    project = args.cache.resolve() / name
    if project.exists():
        validate_snapshot(project, source)
    else:
        project.mkdir(parents=True)
        for relative in source["files"]:
            target = project / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(repo / relative, target)
    write_json(project / "source.json", source)
    command = [
        "uv",
        "sync",
        "--project",
        str(project),
        "--locked",
        "--no-dev",
        "--python",
        args.python,
    ]
    if name == "mjbatch":
        command += ["--group", "examples"]
    run_process(command, project, child_environment(project), args.timeout)
    validate_snapshot(project, source)
    write_json(
        project / "ready.json",
        {"revision": source["revision"], "lock_sha256": sha256(project / "uv.lock")},
    )
    LOGGER.info("Prepared %s at %s", name, project)


def execute(args):
    recipe = RECIPES[args.task]
    if getattr(args, "go1_learner", None) and args.task != "go1-joystick":
        raise ValueError("--go1-learner applies to Go1 only")
    if not getattr(args, "headless", True) and not (
        args.task.startswith("wuji-") or args.task == "go1-joystick"
    ):
        raise ValueError(
            "Training --no-headless currently supports Wuji and Go1 tasks only"
        )
    if not getattr(args, "headless", True):
        LOGGER.info(
            "Training preview: http://127.0.0.1:%s (available after the first rollout step)",
            args.viewer_port,
        )
    standalone = getattr(args, "standalone", False)
    if standalone and args.task != "go1-joystick":
        raise ValueError("--standalone currently supports Go1 only")
    if (
        getattr(args, "go1_learning_rate", None) is not None
        and args.task != "go1-joystick"
    ):
        raise ValueError("--go1-learning-rate applies to Go1 only")
    if getattr(args, "go1_command_profile", None) and args.task != "go1-joystick":
        raise ValueError("--go1-command-profile applies to Go1 only")
    if getattr(args, "go1_reward_profile", None) and args.task != "go1-joystick":
        raise ValueError("--go1-reward-profile applies to Go1 only")
    if args.command == "evaluate" and args.steps is None:
        args.steps = 3000 if args.suite in SWITCHING_SUITES else 500
    if getattr(args, "suite", None) in SWITCHING_SUITES and args.task == "go1-joystick":
        from ._go1_metrics import switching_schedule

        switching_schedule(args.steps, args.suite)
    if getattr(args, "go1_semantics", None) and args.task != "go1-joystick":
        raise ValueError("--go1-semantics applies to Go1 only")
    if args.command == "evaluate":
        is_wuji = recipe["source"] == "wuji_unilab"
        if args.num_envs is None:
            args.num_envs = 1 if is_wuji else 32
        if is_wuji and (args.num_envs != 1 or args.velocity is not None):
            raise ValueError(
                "Wuji evaluation runs sequential single-environment trials and has no velocity command"
            )
        if not is_wuji and args.velocity is None:
            args.velocity = [0.5, 0.0, 0.0]
    if (
        getattr(args, "policy", "trained") != "trained"
        and recipe["source"] != "wuji_unilab"
    ):
        raise ValueError("--policy zero currently applies to Wuji only")
    if getattr(args, "record_video", False):
        if recipe["source"] != "wuji_unilab":
            raise ValueError("--record-video currently applies to Wuji only")
        if shutil.which("ffprobe") is None:
            raise ValueError("--record-video requires ffprobe on PATH")
        if (args.steps + 1) * args.num_trials * 640 * 368 * 3 > 256 * 1024**2:
            raise ValueError(
                "Requested video exceeds 256 MiB of raw frames; reduce trials or steps"
            )
    criteria = criteria_for(args)
    if args.task.startswith("wuji-") and any(
        getattr(args, name, None) for name in ("suite", "seeds", "record_motion")
    ):
        raise ValueError(
            "--suite, --seeds and --record-motion currently apply to Go1 only"
        )
    if args.command not in recipe["methods"]:
        raise ValueError(f"{args.task} supports {recipe['methods']}")
    if args.command == "train":
        minibatches = 32 if recipe["source"] == "wuji_unilab" else 4
        samples = args.num_envs * args.horizon
        if samples < 2 * minibatches or samples % minibatches:
            raise ValueError(
                f"num-envs * horizon must be divisible by {minibatches} and at least {2 * minibatches}"
            )
    if standalone:
        project = None
        source = dict(GO1_STANDALONE_SOURCE)
    else:
        project = args.cache.resolve() / recipe["source"]
        source = json.loads((project / "source.json").read_text())
        ready = json.loads((project / "ready.json").read_text())
        if (
            source["revision"] != SOURCES[recipe["source"]]
            or ready["revision"] != source["revision"]
            or ready["lock_sha256"] != source["files"]["uv.lock"]
        ):
            raise ValueError("Environment source mismatch; run recipes setup")
        validate_snapshot(project, source)
    previous = (
        checkpoint_input(args.run.resolve(), args.task)
        if args.command == "evaluate"
        else None
    )
    if args.command == "train" and getattr(args, "resume_run", None):
        previous = checkpoint_input(
            args.resume_run.resolve(), args.task, allow_recovery=True
        )
        if previous[1].get("checkpoint_origin"):
            LOGGER.info(
                "Recovering Go1 checkpoint at iteration %s from %s run",
                previous[1]["result"]["checkpoint_iteration"],
                previous[1]["status"],
            )
    if args.task == "go1-joystick":
        inherited_backend = (
            previous[1]["result"].get("learner_backend", "native")
            if previous
            else "native"
        )
        requested_backend = getattr(args, "go1_learner", None)
        if (
            previous
            and requested_backend is not None
            and requested_backend != inherited_backend
        ):
            raise ValueError("Go1 resume cannot switch learner backend")
        args.go1_learner = requested_backend or inherited_backend
        if (
            args.command == "train"
            and args.go1_learner == "light-loco"
            and not standalone
        ):
            raise ValueError(
                "Light Loco training requires --standalone in an environment with the light-loco extra"
            )
        args.go1_learning_rate = resolve_go1_learning_rate(
            getattr(args, "go1_learning_rate", None),
            previous[1]["result"] if previous else None,
        )
        args.go1_command_profile = resolve_go1_command_profile(
            getattr(args, "go1_command_profile", None),
            previous[1]["result"] if previous else None,
        )
        args.go1_reward_profile = resolve_go1_reward_profile(
            getattr(args, "go1_reward_profile", None),
            previous[1]["result"] if previous else None,
        )
        args.go1_semantics = resolve_go1_semantics(
            getattr(args, "go1_semantics", None),
            previous[1]["result"] if previous else None,
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = {
        k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()
    }
    request.update(
        project=str(project) if project else None,
        upstream_task=recipe["task"] if project else None,
        start_iteration=0,
    )
    if previous and args.task == "go1-joystick":
        from ._go1_assets import recorded_go1_assets

        input_run = args.run if args.command == "evaluate" else args.resume_run
        request["input_assets"] = recorded_go1_assets(input_run, previous[1]["result"])
        request["input_checkpoint_iteration"] = previous[1]["result"][
            "checkpoint_iteration"
        ]
        request["input_checkpoint_sha256"] = previous[1]["result"]["checkpoint_sha256"]
        request["input_learner_backend"] = previous[1]["result"].get(
            "learner_backend", "native"
        )
        request["input_learning_rate_override"] = previous[1]["result"].get(
            "learning_rate_override"
        )
        request["input_command_profile"] = previous[1]["result"].get(
            "command_profile", "original"
        )
        request["input_reward_profile"] = previous[1]["result"].get(
            "reward_profile", "original"
        )
        request["input_task_semantics"] = previous[1]["result"].get(
            "task_semantics", "upstream-v1"
        )
    if previous and args.command == "train":
        prior_result = previous[1]["result"]
        request["start_iteration"] = prior_result["checkpoint_iteration"] + (
            recipe["source"] == "mjbatch"
        )
        request["prior_updates"] = prior_result.get(
            "cumulative_updates", prior_result["completed_updates"]
        )

    manifest = {
        "schema": 1,
        "workflow": "recipe_" + args.command,
        "run_contract": run_identity("recipes", args.command, args.task),
        "recipe": args.task,
        "source": source,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "launcher_sha256": sha256(Path(__file__)),
        "worker_sha256": sha256(WORKER),
        "adapter_sha256": {
            name: sha256(Path(__file__).parent / name)
            for name in (
                "_wuji_recipe.py",
                "_wuji_diagnostics.py",
                "_mjbatch_recipe.py",
                "runtime/implementation.py",
                "_h1_metrics.py",
                "_go1_metrics.py",
                "_go1_assets.py",
                "_go1_implementation.py",
                "_go1_resume.py",
                "_go1_checkpoint.py",
                "_go1_recovery.py",
                "_h1_motion.py",
                "locomotion/go1.py",
                "locomotion/go1_config.py",
                "locomotion/go1_ppo.py",
                "locomotion/go1_scene.xml",
            )
        },
        "request": request,
    }
    with run_lifecycle(
        manifest,
        lambda record: write_json(output / "run.json", record),
        label=output / "run.json",
    ):
        implementation = output / "implementation" / "embodiedforge"
        frozen = snapshot_implementation(Path(__file__).parent, implementation)
        expected_files = {
            **manifest["adapter_sha256"],
            "recipes.py": manifest["launcher_sha256"],
            "_recipe_worker.py": manifest["worker_sha256"],
        }
        if any(
            frozen["files"].get(name) != digest
            for name, digest in expected_files.items()
        ):
            raise ValueError("Implementation changed while preparing the run")
        manifest["implementation"] = {
            "path": str(implementation.relative_to(output)),
            **frozen,
        }
        if previous and args.task == "go1-joystick" and args.command == "evaluate":
            from ._go1_implementation import copy_go1_implementation

            recorded = copy_go1_implementation(
                args.run.resolve(), previous[1], output / "input-implementation"
            )
            request["input_implementation"] = recorded
            request["input_versions"] = (
                previous[1]["result"].get("runtime", {}).get("versions", {})
            )
            if recorded is None:
                LOGGER.warning(
                    "Run has no implementation snapshot; evaluation uses current Go1 code. "
                    "Training source consistency cannot be verified."
                )
            else:
                manifest["input_implementation"] = {
                    **recorded,
                    "path": str(Path(recorded["path"]).relative_to(output)),
                }
        if previous:
            checkpoint, data = previous
            target = output / "input.pt"
            shutil.copyfile(checkpoint, target)
            input_sha256 = sha256(target)
            if input_sha256 != data["result"]["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during copy")
            request["checkpoint"] = str(target)
            manifest["input_sha256"] = input_sha256
            if args.task == "go1-joystick" and args.command == "train":
                origin = output / "input-run.json"
                write_json(origin, data)
                request["input_run_manifest"] = str(origin)
                request["input_run_manifest_sha256"] = sha256(origin)
        write_json(output / "request.json", request)
        command = [
            sys.executable if standalone else str(project / ".venv/bin/python"),
            "-I",
            str(implementation / WORKER.name),
            str(output / "request.json"),
        ]
        manifest["command"] = command
        write_json(output / "run.json", manifest)
        environment = child_environment(project)
        if standalone:
            from ._go1_assets import reuse_asset_cache

            reuse_asset_cache(environment, request.get("input_assets"))
        run_process(command, output, environment, args.timeout)
        result = json.loads((output / "result.json").read_text())
        validate_result(request, result, output)
        validate_snapshot(implementation, frozen, label="Run implementation snapshot")
        if request.get("input_implementation"):
            recorded = request["input_implementation"]
            validate_snapshot(
                Path(recorded["path"]), recorded, label="Input implementation snapshot"
            )
        if project is not None:
            validate_snapshot(project, source)
        if args.command == "train":
            result["cumulative_updates"] = (
                request.get("prior_updates", 0) + result["completed_updates"]
            )
            write_json(output / "result.json", result)
            if args.task == "go1-joystick" and previous:
                report = json.loads((output / "resume-report.json").read_text())
                if (
                    sha256(output / "resume-report.json")
                    != result["resume_report"]["sha256"]
                ):
                    raise ValueError("Resume report SHA256 mismatch")
                if (
                    sha256(output / "input-run.json")
                    != request["input_run_manifest_sha256"]
                ):
                    raise ValueError("Resume input manifest SHA256 mismatch")
                LOGGER.info(
                    "Go1 resume code=%s; configuration changes=%s; dependency changes=%s; report: %s",
                    report["code"]["status"],
                    [
                        k
                        for k, v in report["configuration"].items()
                        if v["status"] == "changed"
                    ],
                    [
                        k
                        for k, v in report["dependencies"].items()
                        if v["status"] == "changed"
                    ],
                    output / "resume-report.json",
                )
        manifest.update(status="complete", result=result)
        if args.command == "evaluate":
            acceptance = assess(result, criteria)
            manifest["acceptance"] = acceptance
            write_json(output / "acceptance.json", acceptance)
            if acceptance["passed"] is False:
                raise EvaluationRejected(
                    "Policy did not meet the requested evaluation criteria"
                )
        LOGGER.info("%s completed: %s", args.task, output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Project-owned Go1 PPO and external Wuji/MPC recipes in isolated SDKs"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    prep = sub.add_parser("setup")
    prep.add_argument("--source", choices=SOURCES, required=True)
    prep.add_argument("--repo", type=Path)
    prep.add_argument("--python", default="3.12")
    prep.add_argument("--timeout", type=float, default=1200)
    prep.add_argument("--cache", type=Path, default=Path(".cache/external"))
    for mode in ("train", "evaluate", "solve"):
        child = sub.add_parser(mode)
        child.add_argument("--task", choices=RECIPES, required=True)
        child.add_argument("--cache", type=Path, default=Path(".cache/external"))
        child.add_argument("--output", type=Path, required=True)
        child.add_argument(
            "--timeout",
            type=float,
            default=None if mode == "train" else 600,
            help="Wall-clock limit in seconds; training has no limit by default",
        )
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--num-envs", type=int, default=32)
        child.add_argument("--threads", type=int, default=4)
        if mode in ("train", "evaluate"):
            child.add_argument(
                "--standalone",
                action="store_true",
                help="Go1 only: use installed packages in the current Python; no SDK checkout/setup",
            )
            child.add_argument(
                "--go1-semantics",
                choices=["transition-v2", "upstream-v1"],
                help="Go1 transition contract; defaults to input run, or transition-v2 for new training",
            )
        if mode == "train":
            child.add_argument(
                "--go1-learner",
                choices=("native", "light-loco"),
                help="Go1 PPO backend: defaults to native, or inherits the resumed run",
            )
            child.add_argument(
                "--headless",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Disable training preview (default); --no-headless opens a browser preview",
            )
            child.add_argument("--viewer-port", type=int, default=8083)
            child.add_argument("--viewer-fps", type=int, default=10)
            child.add_argument(
                "--go1-learning-rate",
                type=float,
                help="Fixed Go1 learning rate; inherits input override, otherwise uses original annealing",
            )
            child.add_argument(
                "--go1-command-profile",
                choices=tuple(GO1_COMMAND_PROFILES),
                help="Go1 command distribution; defaults to input run, or original for new training",
            )
            child.add_argument(
                "--go1-reward-profile",
                choices=tuple(GO1_REWARD_PROFILES),
                help="Go1 reward weights; defaults to input run, or original for new training",
            )
            child.add_argument(
                "--updates",
                type=int,
                required=True,
                help="Additional PPO updates in this invocation",
            )
            child.add_argument("--resume-run", type=Path)
            child.add_argument("--horizon", type=int, default=24)
        elif mode == "evaluate":
            child.set_defaults(num_envs=None)
            commands = child.add_mutually_exclusive_group()
            child.add_argument("--run", type=Path, required=True)
            child.add_argument(
                "--steps",
                type=int,
                help="Default: 3000 for switching suites, 500 otherwise",
            )
            child.add_argument("--num-trials", type=int, default=20)
            child.add_argument(
                "--policy", choices=["trained", "zero"], default="trained"
            )
            commands.add_argument(
                "--suite", choices=["basic", "extended", *SWITCHING_SUITES]
            )
            child.add_argument("--seeds", type=int, nargs="+")
            child.add_argument("--record-motion", action="store_true")
            child.add_argument("--record-video", action="store_true")
            commands.add_argument("--velocity", type=float, nargs=3)
            for criterion in (
                "min-success-rate",
                "max-drop-rate",
                "min-survival-fraction",
                "max-planar-rmse",
                "max-yaw-rmse",
                "min-settled-fraction",
                "min-tracking-fraction",
            ):
                child.add_argument("--" + criterion, type=float)
        else:
            child.set_defaults(num_envs=1024)
            child.add_argument("--steps", type=int, default=150)
            child.add_argument("--horizon", type=int, default=25)
            child.add_argument("--generations", type=int, default=30)
    status = sub.add_parser("status")
    status.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(args, "viewer_port"):
        if not 1 <= args.viewer_port <= 65535:
            parser.error("--viewer-port must be between 1 and 65535")
        if not 1 <= args.viewer_fps <= 30:
            parser.error("--viewer-fps must be between 1 and 30")
    for name in (
        "timeout",
        "num_envs",
        "threads",
        "updates",
        "horizon",
        "steps",
        "num_trials",
        "generations",
    ):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if getattr(args, "seed", 0) < 0:
        parser.error("--seed must be nonnegative")
    if getattr(args, "velocity", None) is not None and not all(
        math.isfinite(v) for v in args.velocity
    ):
        parser.error("--velocity must be finite")
    if getattr(args, "seeds", None) and (
        min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds)
    ):
        parser.error("--seeds must be nonnegative and unique")
    setup_logging("INFO")
    if args.command == "list":
        print(json.dumps(RECIPES, indent=2))
        return
    if args.command == "status":
        data = json.loads((args.run / "run.json").read_text())
        summary = {
            key: data.get(key)
            for key in (
                "workflow",
                "recipe",
                "status",
                "started_at",
                "finished_at",
                "error",
            )
        }
        summary["request"] = data.get("request")
        summary["checkpoint"] = data.get("result", {}).get("checkpoint")
        summary["completed_updates"] = data.get("result", {}).get("completed_updates")
        summary["cumulative_updates"] = data.get("result", {}).get("cumulative_updates")
        summary["acceptance"] = data.get("acceptance")
        progress = go1_training_progress(args.run, data)
        if progress is not None:
            summary["progress"] = progress
        if (
            data.get("recipe") == "go1-joystick"
            and data.get("status")
            in {
                "interrupted",
                "timed_out",
                "failed",
            }
            and data.get("workflow") == "recipe_train"
        ):
            try:
                _, recovered = checkpoint_input(
                    args.run.resolve(), "go1-joystick", allow_recovery=True
                )
                summary["recoverable_checkpoint"] = recovered["result"]
            except (ValueError, OSError, KeyError) as error:
                summary["recovery_error"] = str(error)
        print(json.dumps(summary, indent=2))
        return
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        setup(args) if args.command == "setup" else execute(args)
    except EvaluationRejected as exc:
        LOGGER.warning("%s", exc)
        raise SystemExit(2) from exc
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
