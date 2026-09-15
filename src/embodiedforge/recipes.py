"""Pinned external task recipes, isolated from the lightweight core runtime."""

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

from ._go1_metrics import SWITCHING_SUITES
from .locomotion.go1_config import COMMAND_PROFILES as GO1_COMMAND_PROFILES
from .locomotion.go1_config import REWARD_PROFILES as GO1_REWARD_PROFILES
from .logging import get_logger, setup_logging

SOURCES = {
    "wuji_unilab": "91ccfa0ec8c129b300865bd36c59dc9eed56a744",
    "mjbatch": "b84c0c20aedbdf048122cbc47f554e9b93cc4754",
}
RECIPES = {
    "wuji-reorient": {
        "source": "wuji_unilab",
        "task": "WujiHand_Reorient",
        "methods": ["train", "evaluate"],
        "physics": "mjwarp",
        "algorithm": "PPO",
    },
    "wuji-reorient-light": {
        "source": "wuji_unilab",
        "task": "WujiHand_Reorient_Light",
        "methods": ["train", "evaluate"],
        "physics": "mjwarp",
        "algorithm": "PPO",
    },
    "go1-joystick": {
        "source": "mjbatch",
        "task": "go1_joystick",
        "methods": ["train", "evaluate"],
        "physics": "mjbatch",
        "algorithm": "PPO",
    },
    "cartpole-mpc": {
        "source": "mjbatch",
        "task": "cartpole_mpc",
        "methods": ["solve"],
        "physics": "mjbatch",
        "algorithm": "predictive_sampling",
    },
    "arm-throw-codesign": {
        "source": "mjbatch",
        "task": "arm_throw",
        "methods": ["solve"],
        "physics": "mjbatch",
        "algorithm": "CEM",
    },
}
WORKER = Path(__file__).with_name("_recipe_worker.py")
LOGGER = get_logger("recipes")
GO1_STANDALONE_SOURCE = {
    "kind": "installed_packages",
    "implementation": "embodiedforge.locomotion.go1+go1_ppo",
    "contract_version": 1,
}


class EvaluationRejected(RuntimeError):
    """The simulation completed but did not meet the requested criteria."""


def resolve_go1_semantics(requested, previous_result=None):
    """New runs use corrected transitions; existing runs retain their contract."""
    inherited = (
        previous_result.get("task_semantics", "upstream-v1")
        if previous_result is not None
        else "transition-v2"
    )
    choices = ("transition-v2", "upstream-v1")
    if inherited not in choices or (requested is not None and requested not in choices):
        raise ValueError("Unknown Go1 task semantics")
    return inherited if requested is None else requested


def resolve_go1_reward_profile(requested, previous_result=None):
    inherited = (previous_result or {}).get("reward_profile", "original")
    choices = GO1_REWARD_PROFILES
    if inherited not in choices or (requested is not None and requested not in choices):
        raise ValueError("Unknown Go1 reward profile")
    return inherited if requested is None else requested


def resolve_go1_command_profile(requested, previous_result=None):
    inherited = (previous_result or {}).get("command_profile", "original")
    if inherited not in GO1_COMMAND_PROFILES or (
        requested is not None and requested not in GO1_COMMAND_PROFILES
    ):
        raise ValueError("Unknown Go1 command profile")
    return inherited if requested is None else requested


def resolve_go1_learning_rate(requested, previous_result=None):
    inherited = (previous_result or {}).get("learning_rate_override")
    for value in (requested, inherited):
        if value is not None and (
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        ):
            raise ValueError("Go1 learning rate must be positive and finite")
    return inherited if requested is None else requested


def criteria_for(args):
    names = (
        ("min_success_rate", "max_drop_rate")
        if args.task.startswith("wuji-")
        else ("min_survival_fraction", "max_planar_rmse", "max_yaw_rmse")
    )
    if args.task == "go1-joystick" and getattr(args, "suite", None) in SWITCHING_SUITES:
        names += ("min_settled_fraction", "min_tracking_fraction")
    criteria = {}
    for name in (
        "min_success_rate",
        "max_drop_rate",
        "min_survival_fraction",
        "max_planar_rmse",
        "max_yaw_rmse",
        "min_settled_fraction",
        "min_tracking_fraction",
    ):
        value = getattr(args, name, None)
        if value is None:
            continue
        if name not in names:
            raise ValueError(f"{name} does not apply to {args.task}")
        if (
            not math.isfinite(value)
            or value < 0
            or (("rate" in name or "fraction" in name) and value > 1)
        ):
            raise ValueError(f"Invalid acceptance threshold: {name}")
        criteria[name] = value
    return criteria


def assess(result, criteria):
    if "cases" in result:
        checks = []
        for case in result["cases"]:
            checks.extend(
                {**item, "seed": case["seed"], "case": case["case"]}
                for item in assess(case, criteria)["checks"]
            )
        return {
            "criteria": criteria,
            "checks": checks,
            "passed": all(item["passed"] for item in checks) if checks else None,
        }
    if "segments" in result:
        checks = assess({k: v for k, v in result.items() if k != "segments"}, criteria)[
            "checks"
        ]
        for segment in result["segments"]:
            checks.extend(
                {**item, "segment": segment["segment"]}
                for item in assess(segment, criteria)["checks"]
            )
        return {
            "criteria": criteria,
            "checks": checks,
            "passed": all(item["passed"] for item in checks) if checks else None,
        }
    checks = []
    for name, threshold in criteria.items():
        metric = name.split("_", 1)[1]
        actual = result[metric]
        unobserved = actual is None and result.get("observed_frames") == 0
        if not unobserved and not math.isfinite(actual):
            raise ValueError(f"Non-finite evaluation metric: {metric}")
        passed = (
            False
            if unobserved
            else actual >= threshold
            if name.startswith("min_")
            else actual <= threshold
        )
        checks.append(
            {
                "criterion": name,
                "actual": actual,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return {
        "criteria": criteria,
        "checks": checks,
        "passed": all(item["passed"] for item in checks) if checks else None,
    }


def validate_result(request, result, output):
    if request.get("task") == "go1-joystick":
        for item in result.get("cases", [result]):
            if item.get("learning_rate_override") != request.get("go1_learning_rate"):
                raise ValueError("Go1 result learning rate differs from request")
    if request.get("go1_command_profile"):
        for item in result.get("cases", [result]):
            if item.get("command_profile") != request["go1_command_profile"]:
                raise ValueError("Go1 result command profile differs from request")
    if request.get("go1_reward_profile"):
        for item in result.get("cases", [result]):
            if item.get("reward_profile") != request["go1_reward_profile"]:
                raise ValueError("Go1 result reward profile differs from request")
    if request.get("go1_semantics"):
        for item in result.get("cases", [result]):
            if item.get("task_semantics") != request["go1_semantics"]:
                raise ValueError("Go1 result task semantics differs from request")
    if request["command"] == "train":
        path = (output / result["checkpoint"]).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError("Worker checkpoint is missing or outside its run")
        if sha256(path) != result["checkpoint_sha256"]:
            raise ValueError("Worker checkpoint SHA256 mismatch")
        if (
            result["completed_updates"] != request["updates"]
            or result["checkpoint_iteration"]
            != request.get("start_iteration", 0) + request["updates"] - 1
            or result["finite_tensor_count"] <= 0
        ):
            raise ValueError("Incomplete training result")
    elif request["command"] == "evaluate":
        if request["task"].startswith("wuji-"):
            from ._wuji_diagnostics import validate_evaluation

            validate_evaluation(request, result)
            if request.get("record_video"):
                from ._wuji_diagnostics import verify_video

                write_json(
                    output / "video-validation.json", verify_video(result, output)
                )
        else:
            from ._go1_metrics import evaluation_commands
            from ._h1_metrics import validate_first_episode

            commands = evaluation_commands(
                request.get("suite"), request.get("velocity")
            )
            expected = {
                (seed, case)
                for seed in (request.get("seeds") or [request["seed"]])
                for case in commands
            }
            cases = result.get("cases", [result])
            found = [(case["seed"], case["case"]) for case in cases]
            if set(found) != expected or len(found) != len(expected):
                raise ValueError("Missing, duplicate or unexpected Go1 evaluation case")
            for case in cases:
                validate_first_episode(case)
                if (
                    case["num_envs"] != request["num_envs"]
                    or case["steps_limit"] != request["steps"]
                    or case["command"] != commands[case["case"]]
                ):
                    raise ValueError(
                        "Evaluation dimensions or command differ from request"
                    )
                if request.get("suite") in SWITCHING_SUITES:
                    from ._go1_metrics import validate_switching

                    validate_switching(case, request["steps"], request["suite"])
                if request.get("record_motion"):
                    from ._h1_motion import load_motion

                    motion = case["motion"]
                    path = (output / motion["path"]).resolve()
                    if (
                        not path.is_relative_to(output)
                        or sha256(path) != motion["sha256"]
                    ):
                        raise ValueError("Motion SHA256 mismatch")
                    trace = load_motion(path)
                    if len(trace["time"]) != case["per_env"]["observed_steps"][0]:
                        raise ValueError("Motion frame count differs from evaluation")
                    metadata = trace["metadata"]
                    if (
                        request.get("suite") in SWITCHING_SUITES
                        and metadata.get("command_schedule") != case["command_schedule"]
                    ):
                        raise ValueError(
                            "Motion command schedule differs from evaluation"
                        )
                    if (
                        metadata["case"] != case["case"]
                        or metadata["seed"] != case["seed"]
                        or metadata["env_id"] != 0
                        or metadata["quaternion_order"] != "xyzw"
                        or metadata["velocity_command"] != case["command"]
                        or not math.isclose(metadata["dt"], case["dt"], abs_tol=1e-12)
                        or trace["terminated"] != case["per_env"]["terminated"][0]
                        or trace["truncated"] != case["per_env"]["truncated"][0]
                    ):
                        raise ValueError(
                            "Motion metadata or final flags differ from evaluation"
                        )
    json.dumps(result, allow_nan=False)


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_snapshot(project: Path, source: dict, *, label="Cached source") -> None:
    for relative, expected in source["files"].items():
        path = project / relative
        if not path.is_file() or path.is_symlink() or sha256(path) != expected:
            raise ValueError(f"{label} differs: {path}")


def snapshot_implementation(package: Path, destination: Path) -> dict:
    """Freeze the native Python package and its packaged robot assets for one run."""
    package, destination = package.resolve(), destination.resolve()
    if destination.is_relative_to(package):
        raise ValueError(
            "Run implementation snapshot must be outside the source package"
        )
    files = []
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.relative_to(package).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Implementation source must not contain symlinks: {path}")
        if path.is_file() and (
            path.suffix in (".py", ".xml") or path.name.startswith("LICENSE")
        ):
            files.append(path)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in files:
        relative = path.relative_to(package)
        expected = sha256(path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != expected:
            raise ValueError(f"Implementation source changed during copy: {path}")
        hashes[str(relative)] = expected
    validate_snapshot(package, {"files": hashes}, label="Implementation source")
    return {"files": hashes}


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


def run_process(command, output: Path, env: dict, timeout: float):
    LOGGER.info("Running recipe; full output: %s", output / "console.log")
    with (
        (output / "console.log").open("a") as log,
        subprocess.Popen(
            command,
            cwd=output,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        ) as process,
    ):
        try:
            code = process.wait(timeout=timeout)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except (ProcessLookupError, KeyboardInterrupt, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


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


def checkpoint_input(run: Path, recipe: str) -> tuple[Path, dict]:
    data = json.loads((run / "run.json").read_text())
    if (
        data.get("schema") != 1
        or data.get("workflow") != "recipe_train"
        or data.get("status") != "complete"
        or data.get("recipe") != recipe
        or not (
            data.get("source", {}).get("revision") == SOURCES[RECIPES[recipe]["source"]]
            or (
                recipe == "go1-joystick" and data.get("source") == GO1_STANDALONE_SOURCE
            )
        )
    ):
        raise ValueError(
            "Input must be a completed training run for this recipe/version"
        )
    artifact = data["result"]
    path = (run / artifact["checkpoint"]).resolve()
    if not path.is_relative_to(run) or not path.is_file():
        raise ValueError("Checkpoint is missing or outside the run")
    if sha256(path) != artifact["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    return path, data


def execute(args):
    recipe = RECIPES[args.task]
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
        previous = checkpoint_input(args.resume_run.resolve(), args.task)
    if args.task == "go1-joystick":
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
                "_training_runtime.py",
                "_h1_metrics.py",
                "_go1_metrics.py",
                "_go1_assets.py",
                "_go1_implementation.py",
                "_go1_resume.py",
                "_go1_checkpoint.py",
                "_h1_motion.py",
                "locomotion/go1.py",
                "locomotion/go1_config.py",
                "locomotion/go1_ppo.py",
                "locomotion/go1_scene.xml",
            )
        },
        "request": request,
    }
    write_json(output / "run.json", manifest)
    try:
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
            if sha256(target) != data["result"]["checkpoint_sha256"]:
                raise ValueError("Checkpoint changed during copy")
            request["checkpoint"] = str(target)
            manifest["input_sha256"] = sha256(target)
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
    except EvaluationRejected as exc:
        manifest.update(status="rejected", error=str(exc))
        raise
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


def go1_training_progress(run, manifest):
    """Read the latest complete metrics row without loading the SDK or checkpoint.

    The live log may be ahead of the periodically saved checkpoint. Only a
    bounded tail is read, and an unfinished final line is left for the next poll.
    """
    request = manifest.get("request") or {}
    if request.get("task") != "go1-joystick" or request.get("command") != "train":
        return None
    try:
        with (Path(run) / "metrics.jsonl").open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            offset = max(0, size - 65536)
            stream.seek(offset)
            tail = stream.read(65536)
    except FileNotFoundError:
        return None
    if offset:
        tail = tail.partition(b"\n")[2]
    lines = [line for line in tail.split(b"\n")[:-1] if line.strip()]
    if not lines:
        return None

    def reject_constant(value):
        raise ValueError(f"Non-finite Go1 progress value: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            reject_constant(value)
        return parsed

    try:
        latest = json.loads(
            lines[-1], parse_constant=reject_constant, parse_float=finite_float
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid complete Go1 metrics row") from exc
    if not isinstance(latest, dict):
        raise ValueError("Invalid Go1 progress record")
    start = request.get("start_iteration", 0)
    counts = [request.get(key) for key in ("updates", "num_envs", "horizon")]
    iteration, transitions = latest.get("iteration"), latest.get("transitions")
    elapsed = latest.get("elapsed_seconds")
    if (
        type(start) is not int
        or start < 0
        or any(type(value) is not int or value <= 0 for value in counts)
        or type(iteration) is not int
        or type(transitions) is not int
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("Invalid Go1 progress counts or elapsed time")
    updates, num_envs, horizon = counts
    observed = iteration - start + 1
    if not 1 <= observed <= updates or transitions != observed * num_envs * horizon:
        raise ValueError("Go1 progress differs from the requested training budget")
    return {
        "observed_updates": observed,
        "requested_updates": updates,
        "cumulative_updates": iteration + 1,
        "fraction": observed / updates,
        "latest_metrics": latest,
    }


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
        child.add_argument("--timeout", type=float, default=600)
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
            child.add_argument("--updates", type=int, default=3)
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
