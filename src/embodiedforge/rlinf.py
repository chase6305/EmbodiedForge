"""Optional, isolated RLinf ManiSkill PickCube / MLP / SAC workflow."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ._microduck_process import run_process, termination_signals
from .h1 import child_environment, sha256, write_json
from .recipes import snapshot_implementation

REVISION = "67864d67c086a111de9921c5f4591eb5882f74ec"
CONFIG = "maniskill_sac_mlp"
WEIGHTS = "actor/model_state_dict/full_weights.pt"
REPLAY_WINDOW = 10_000  # Fixed maniskill_sac_mlp cache and sampling window.


def check_checkout(repo):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError(f"Expected RLinf {REVISION}, found {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError("RLinf has tracked changes; use the pinned clean checkout")


def checkpoint_step(path):
    """Check the pinned, single-rank SAC checkpoint layout before resuming."""
    match = re.fullmatch(r"global_step_([0-9]+)", path.name)
    if not match or int(match[1]) < 1:
        raise ValueError("Resume requires a global_step_N checkpoint directory")
    for relative in (
        WEIGHTS,
        "actor/dcp_checkpoint/.metadata",
        "actor/sac_components/alpha/dcp_checkpoint/.metadata",
        "actor/sac_components/target_model/checkpoint_rank_0.pt",
        "actor/sac_components/replay_buffer/rank_0/metadata.json",
        "actor/sac_components/replay_buffer/rank_0/trajectory_index.json",
    ):
        if not (path / relative).is_file() or not (path / relative).stat().st_size:
            raise ValueError(f"Incomplete SAC checkpoint: missing {path / relative}")
    for relative in (
        "actor/dcp_checkpoint",
        "actor/sac_components/alpha/dcp_checkpoint",
    ):
        if not any(
            p.stat().st_size for p in (path / relative).glob("*.distcp") if p.is_file()
        ):
            raise ValueError(
                f"Incomplete SAC checkpoint: no data shards in {path / relative}"
            )
    replay = path / "actor/sac_components/replay_buffer/rank_0"
    try:
        metadata = json.loads((replay / "metadata.json").read_text())
        index = json.loads((replay / "trajectory_index.json").read_text())
        ids = index["trajectory_id_list"]
        entries = index["trajectory_index"]
        if (
            metadata["trajectory_format"] != "pt"
            or not isinstance(ids, list)
            or not ids
        ):
            raise ValueError("expected a nonempty pt replay buffer")
        if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("invalid or duplicate replay trajectory IDs")
        if ids != sorted(ids) or set(entries) != {str(i) for i in ids}:
            raise ValueError("replay history and trajectory index differ")
        samples = 0
        for trajectory_id in ids:
            entry = entries[str(trajectory_id)]
            shape = entry["shape"]
            if (
                not isinstance(shape, list)
                or len(shape) < 2
                or any(type(n) is not int or n < 1 for n in shape)
                or type(entry["num_samples"]) is not int
                or entry["num_samples"] != shape[0] * shape[1]
                or type(entry["trajectory_id"]) is not int
                or entry["trajectory_id"] != trajectory_id
            ):
                raise ValueError("invalid replay trajectory shape or sample count")
            samples += entry["num_samples"]
        for name, expected in (
            ("size", len(ids)),
            ("trajectory_counter", ids[-1] + 1),
            ("total_samples", samples),
        ):
            if type(metadata[name]) is not int or metadata[name] != expected:
                raise ValueError(f"replay {name} differs from its index")
        # Upstream retains the historical index but serializes only the active
        # cache. Requiring every historical file breaks resumes after 10k entries.
        for trajectory_id in ids[-REPLAY_WINDOW:]:
            version = entries[str(trajectory_id)]["model_weights_id"]
            if not isinstance(version, str) or "/" in version or "\\" in version:
                raise ValueError("invalid replay model_weights_id")
            trajectory = replay / f"trajectory_{trajectory_id}_{version}.pt"
            if not trajectory.is_file() or not trajectory.stat().st_size:
                raise ValueError(f"missing active replay trajectory: {trajectory}")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SAC replay checkpoint {replay}: {exc}") from exc
    return int(match[1])


def inventory(root):
    return [
        {
            "path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": sha256(p),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def resume_settings(checkpoint):
    """Recover the saved collection seed and batch width for this single-rank recipe."""
    replay = checkpoint / "actor/sac_components/replay_buffer/rank_0"
    metadata = json.loads((replay / "metadata.json").read_text())
    index = json.loads((replay / "trajectory_index.json").read_text())
    try:
        seed = metadata["seed"]
        latest = str(index["trajectory_id_list"][-1])
        num_envs = index["trajectory_index"][latest]["shape"][1]
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or type(num_envs) is not int
            or num_envs < 1
        ):
            raise ValueError("invalid seed or environment batch width")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid SAC resume settings in {replay}: {exc}") from exc
    return {"seed": seed, "num_envs": num_envs}


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
    status = {"status": "preparing", "command": args.command}
    request_path = output / "request.json"
    try:
        write_json(request_path, request)
        write_json(output / "run.json", status)
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
    except BaseException as exc:
        status.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if args.command == "train":
                status.update(latest_checkpoint(output))
            write_json(output / "run.json", status)
        except Exception as exc:
            if not active_error:
                raise
            print(f"Failed to finalize RLinf run: {exc}", file=sys.stderr)
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(launch(args))
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
