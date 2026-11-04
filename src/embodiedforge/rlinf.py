"""Optional, isolated RLinf ManiSkill PickCube / MLP / SAC workflow."""

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .runtime.artifacts import write_json
from .runtime.environment import child_environment
from .runtime.hashing import file_sha256 as sha256
from .runtime.process import run_process, termination_signals
from .runtime.provenance import snapshot_implementation
from .runtime.records import run_identity, run_lifecycle
from .workflows.rlinf_artifacts import check_checkout as check_checkout
from .workflows.rlinf_artifacts import checkpoint_step as _checkpoint_step
from .workflows.rlinf_artifacts import inventory as inventory
from .workflows.rlinf_artifacts import resume_settings as resume_settings
from .workflows.rlinf_catalog import CONFIG as CONFIG
from .workflows.rlinf_catalog import REPLAY_WINDOW as REPLAY_WINDOW
from .workflows.rlinf_catalog import REVISION as REVISION
from .workflows.rlinf_catalog import WEIGHTS as WEIGHTS


def checkpoint_step(path):
    """Compatibility entry retaining the launcher's configurable replay window."""
    return _checkpoint_step(path, replay_window=REPLAY_WINDOW)


def latest_checkpoint(output):
    """Find the latest complete layout; upstream directory saves are not atomic."""
    directory = output / CONFIG / "checkpoints"
    candidates = [
        path
        for path in directory.glob("global_step_*")
        if path.is_dir() and re.fullmatch(r"global_step_[0-9]+", path.name)
    ]
    for path in sorted(candidates, key=lambda p: int(p.name[12:]), reverse=True):
        try:
            step = checkpoint_step(path)
        except (OSError, ValueError):
            continue
        return {
            "latest_checkpoint": str(path),
            "last_saved_step": step,
            "checkpoint_validation": "layout_only",
        }
    return {
        "latest_checkpoint": None,
        "last_saved_step": None,
        "checkpoint_validation": None,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "export"):
        command = sub.add_parser(name)
        command.add_argument("--repo", type=Path, required=True)
        command.add_argument(
            "--python", type=Path, required=True, help="RLinf SDK Python"
        )
        command.add_argument(
            "--output", type=Path, required=True, help="New run directory"
        )
        command.add_argument(
            "--headless",
            action="store_true",
            default=True,
            help="Always headless; accepted explicitly for scripts",
        )
        command.add_argument(
            "--quiet",
            action="store_true",
            help="Write worker output only to worker.log",
        )
        if name == "export":
            command.add_argument("--checkpoint", type=Path, required=True)
            command.add_argument(
                "--weight-storage",
                choices=("float32", "float16", "int8"),
                default="float32",
                help="Matrix weight storage; computation and I/O remain float32",
            )
            command.set_defaults(gpu=None, seed=1234, num_envs=16, eval_epochs=1)
            continue
        command.add_argument(
            "--gpu", type=int, default=0, help="One physical NVIDIA GPU index"
        )
        command.add_argument(
            "--seed",
            type=int,
            default=None if name == "train" else 1234,
            help=(
                "Inherit on resume, otherwise 1234"
                if name == "train"
                else "Evaluation environment seed (default: 1234)"
            ),
        )
        command.add_argument(
            "--num-envs",
            type=int,
            default=None if name == "train" else 16,
            help="Training: inherit on resume, otherwise 32; evaluation: 16",
        )
        if name == "train":
            command.add_argument(
                "--iterations",
                type=int,
                required=True,
                help="Additional RLinf collection/update iterations",
            )
            command.add_argument(
                "--save-interval",
                type=int,
                default=200,
                help="Save every N iterations, and at the end",
            )
            command.add_argument(
                "--eval-interval",
                type=int,
                help="Evaluate every N iterations; defaults to --save-interval and must divide it",
            )
            command.add_argument(
                "--resume", type=Path, help="Full global_step_N directory"
            )
        else:
            policy = command.add_mutually_exclusive_group(required=True)
            policy.add_argument(
                "--checkpoint",
                type=Path,
                help="Full model_state_dict/full_weights.pt",
            )
            policy.add_argument(
                "--onnx",
                type=Path,
                help="Exported policy.onnx; CPU policy, GPU simulator",
            )
            command.add_argument(
                "--eval-epochs",
                type=int,
                default=1,
                help="Number of 50-step evaluation rollouts",
            )
    return result


@termination_signals()
def launch(args):
    if (args.num_envs is not None and args.num_envs < 1) or (
        args.seed is not None and not 0 <= args.seed < 2**32
    ):
        raise ValueError("--num-envs must be positive; --seed must be in [0, 2**32)")
    if args.gpu is not None and args.gpu < 0:
        raise ValueError("--gpu must be a nonnegative NVIDIA GPU index")
    if args.command == "train" and min(args.iterations, args.save_interval) < 1:
        raise ValueError("--iterations and --save-interval must be positive")
    eval_interval = None
    if args.command == "train":
        eval_interval = (
            args.save_interval if args.eval_interval is None else args.eval_interval
        )
        if eval_interval < 1 or args.save_interval % eval_interval:
            raise ValueError(
                "--eval-interval must be positive and divide --save-interval"
            )
    if args.command == "evaluate" and args.eval_epochs < 1:
        raise ValueError("--eval-epochs must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Launch once from ef; this backend manages its own workers")
    repo = args.repo.resolve()
    # Resolving a venv Python symlink would discard its environment identity.
    python = args.python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"Missing executable RLinf SDK Python: {python}")
    check_checkout(repo)
    output = args.output.resolve()
    onnx = getattr(args, "onnx", None)
    source = onnx or getattr(args, "resume", None) or getattr(args, "checkpoint", None)
    source = source.resolve() if source else None
    start = checkpoint_step(source) if source and args.command == "train" else 0
    saved = resume_settings(source) if start else {"seed": 1234, "num_envs": 32}
    settings = {}
    for name in ("seed", "num_envs"):
        selected = getattr(args, name)
        if start and selected is not None and selected != saved[name]:
            raise ValueError(
                f"--{name.replace('_', '-')} differs from the checkpoint ({saved[name]}); omit it to inherit saved settings"
            )
        settings[name] = saved[name] if selected is None else selected
    if source and (output == source or output.is_relative_to(source)):
        raise ValueError("Output must be outside the input checkpoint")
    output.mkdir(parents=True, exist_ok=False)
    request = {
        "schema": 1,
        "command": args.command,
        "repo": str(repo),
        "revision": REVISION,
        "config": CONFIG,
        "python": str(python),
        "output": str(output),
        "gpu": args.gpu,
        "quiet": args.quiet,
        **settings,
        "start_step": start,
        "iterations": getattr(args, "iterations", None),
        "save_interval": getattr(args, "save_interval", None),
        "eval_interval": eval_interval,
        "evaluation_rng": "isolated-torch",
        "eval_epochs": getattr(args, "eval_epochs", None),
        "checkpoint": None,
        "policy_format": "onnx" if onnx else "pytorch",
        "input_source": str(source) if source else None,
    }
    if args.command == "export":
        request["weight_storage"] = args.weight_storage
    status = {
        "status": "preparing",
        "command": args.command,
        "run_contract": run_identity("rlinf", args.command, CONFIG),
    }
    request_path = output / "request.json"

    def finalize(record):
        if args.command == "train":
            record.update(latest_checkpoint(output))

    with run_lifecycle(
        status,
        lambda record: write_json(output / "run.json", record),
        finalize=finalize,
        label=output / "run.json",
    ):
        write_json(request_path, request)
        if source:
            inputs = output / "inputs"
            inputs.mkdir()
            filename = "policy.onnx" if onnx else "full_weights.pt"
            checkpoint = inputs / (source.name if args.command == "train" else filename)
            if args.command == "train":
                before = inventory(source)
                shutil.copytree(source, checkpoint)
                copied = inventory(checkpoint)
                if copied != before or inventory(source) != before:
                    raise ValueError(
                        "Resume checkpoint changed during snapshot; stop its writer first"
                    )
                checkpoint_step(checkpoint)
                if resume_settings(checkpoint) != settings:
                    raise ValueError(
                        "Resume settings changed while preparing the snapshot"
                    )
                request["input_files"] = [
                    {**entry, "path": str(Path(checkpoint.name) / entry["path"])}
                    for entry in copied
                ]
            else:
                before = sha256(source)
                shutil.copyfile(source, checkpoint)
                copied = inventory(inputs)
                if copied[0]["sha256"] != before or sha256(source) != before:
                    raise ValueError(
                        "Evaluation weights changed during snapshot; stop their writer first"
                    )
                request["input_files"] = copied
            request["checkpoint"] = str(checkpoint)
        package = output / "implementation" / "embodiedforge"
        request["implementation"] = snapshot_implementation(
            Path(__file__).resolve().parent, package
        )
        write_json(request_path, request)
        env = child_environment(python.parent.parent)
        for key in list(env):
            if key.startswith(("RLINF_", "RAY_")) or key in (
                "RANK",
                "LOCAL_RANK",
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "GROUP_RANK",
                "ROLE_RANK",
                "ROLE_WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "CLUSTER_NAMESPACE",
            ):
                env.pop(key)
        env.update(
            PYTHONPATH=os.pathsep.join((str(package.parent), str(repo))),
            EMBODIED_PATH=str(repo / "examples/embodiment"),
            CUDA_VISIBLE_DEVICES="" if args.command == "export" else str(args.gpu),
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            RLINF_EXT_MODULE="embodiedforge._rlinf_worker",
            EF_RLINF_SEED=str(settings["seed"]),
        )
        status["status"] = "running"
        write_json(output / "run.json", status)
        command = [str(python), "-m", "embodiedforge._rlinf_worker", str(request_path)]
        if (python.parent.parent / "pyvenv.cfg").is_file():
            # RLinf's installer appends Vulkan/library exports to venv activation.
            # Positional arguments preserve spaces and avoid shell interpolation.
            command = [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'source "$1" || exit; shift; exec "$@"',
                "rlinf",
                str(python.parent / "activate"),
                *command,
            ]
        if args.quiet:
            print(f"RLinf worker log: {output / 'worker.log'}", flush=True)
        run_process(
            command,
            cwd=output,
            env=env,
            log_path=output / "worker.log",
            echo=not args.quiet,
        )
        report = json.loads((output / "result.json").read_text())
        if report.get("status") != "complete":
            raise RuntimeError("RLinf worker did not report completion")
        status.update(status="complete", result=report)
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(launch(args))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
