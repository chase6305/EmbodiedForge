"""G1 HOI workflows in an explicitly selected, isolated Weave SDK environment."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ._microduck_process import run_process, termination_signals
from ._motion_clips import MotionClips, import_weave
from .h1 import child_environment, sha256, write_json
from .recipes import snapshot_implementation

WEAVE_REVISION = "b162a351ddabe302a7a3d50d1543eb83dfeff082"
ISAACLAB_REVISION = "e17312889676ed229b986d56c9e0b23a01cf0ab7"
TASK = "G1-Inspire-HOI-v0"
# Asset entry points from the pinned Weave objects/object_cfg.py.
OBJECT_USD = {
    name: f"{name}/{name}.{'usda' if name == 'clothesstand' else 'usd'}"
    for name in (
        "clothesstand",
        "floorlamp",
        "largebox",
        "largetable",
        "smallbox",
        "smalltable",
        "trashcan",
        "tripod",
        "whitechair",
        "woodchair",
    )
}


def check_checkout(path, revision):
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"{path}: expected revision {revision}, found {actual}")
    # Untracked motion files and downloaded assets are permitted.
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError(f"Tracked source changes in {path}; use the pinned checkout")


def check_assets(repo, objects):
    """Check the pinned task's local inputs, without requiring unused source assets."""
    package = repo / "source/g1_hoi_learning/g1_hoi_learning"
    unknown = set(objects) - OBJECT_USD.keys()
    if unknown:
        raise ValueError(f"Unknown Weave objects: {sorted(unknown)}")

    def require(path):
        if not path.is_file():
            raise ValueError(f"Missing Weave asset: {path}")
        with path.open("rb") as stream:
            if stream.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(
                    f"Unresolved Git LFS asset: {path}; fetch Weave LFS assets"
                )

    urdf = package / "assets/unitree_g1/g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf"
    require(urdf)
    try:
        meshes = {mesh.attrib["filename"] for mesh in ET.parse(urdf).iter("mesh")}
    except (ET.ParseError, KeyError) as exc:
        raise ValueError(f"Invalid Weave robot URDF: {urdf}") from exc
    for mesh in sorted(meshes):
        require(urdf.parent / mesh)
    require(package / "objects/geometry/bps_128.npy")
    for name in sorted(objects):
        for relative in (
            OBJECT_USD[name],
            f"{name}/surface.npy",
            f"{name}/sdf_128.npz",
        ):
            require(package / "objects" / relative)


def prepare_inputs(paths, layout, output, mode):
    """Snapshot validated numeric arrays before upstream's pickle-enabled loader."""
    summary = {"names": [], "objects": [], "lengths": [], "files": []}
    for index, path in enumerate(paths):
        clips = (
            import_weave(path, layout) if layout is not None else MotionClips.load(path)
        )
        source_hash = (
            clips.metadata["source"]["sha256"] if layout is not None else clips.sha256
        )
        if any(x in clips.arrays for x in ("terminated", "truncated", "clip_end")):
            raise ValueError(
                "Training references must not contain rollout termination flags"
            )
        if clips.fps != 50:
            raise ValueError(
                "Pinned Weave uses 50 Hz control; resample input motions to 50 fps"
            )
        if index and any(
            clips.metadata[key] != summary["layout"][key]
            for key in ("robot", "joint_names", "body_names")
        ):
            raise ValueError("All motion files must use the same robot layout")
        if set(summary["names"]) & set(clips.names):
            raise ValueError("Motion names must be unique across input files")
        if any(not name.replace("_", "").isalnum() for name in clips.objects):
            raise ValueError("Object names must be asset identifiers")
        # Match MotionLoader's float32 tensors before publishing a snapshot.
        # Torch cannot consume non-native-endian arrays, even with dtype=float32.
        for field in clips.frame_fields:
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.ascontiguousarray(clips.arrays[field], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {field} overflows Weave float32 inputs")
            clips.arrays[field] = values
        destination = output / f"motion_{index:03d}.npz"
        clips.save(destination)
        summary["files"].append(
            {
                "path": str(destination),
                "sha256": sha256(destination),
                "source_path": str(Path(path).resolve()),
                "source_sha256": source_hash,
            }
        )
        summary["names"].extend(clips.names)
        summary["objects"].extend(clips.objects)
        summary["lengths"].extend(clips.lengths.tolist())
        summary["layout"] = {
            key: clips.metadata[key]
            for key in (
                "robot",
                "joint_names",
                "body_names",
                "quaternion_order",
                "coordinate_frame",
            )
        }
        # Release this clip buffer before loading the next large NPZ.
        del clips, values
    if mode == "evaluate" and len(set(summary["objects"])) != 1:
        raise ValueError(
            "Weave evaluation requires one object; evaluate objects separately"
        )
    return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("train", "evaluate", "export"):
        command = sub.add_parser(name)
        command.add_argument("--weave-root", type=Path, required=True)
        command.add_argument("--isaaclab-root", type=Path, required=True)
        command.add_argument(
            "--python", type=Path, required=True, help="Weave SDK Python 3.11"
        )
        command.add_argument("--motions", type=Path, nargs="+", required=True)
        command.add_argument(
            "--layout", type=Path, help="Raw Weave NPZ layout; omit for ef-motion-v1"
        )
        command.add_argument(
            "--output", type=Path, required=True, help="New run directory"
        )
        command.add_argument("--checkpoint", type=Path, required=name != "train")
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--seed", type=int, default=42)
        rendering = command.add_mutually_exclusive_group()
        rendering.add_argument(
            "--gui", action="store_true", help="Enable viewer; default headless"
        )
        rendering.add_argument(
            "--headless",
            dest="gui",
            action="store_false",
            help="Run without a viewer (default)",
        )
        command.set_defaults(gui=False)
        if name == "train":
            command.add_argument("--num-envs", type=int, default=4096)
            command.add_argument(
                "--iterations",
                type=int,
                required=True,
                help="Additional PPO iterations",
            )
            command.add_argument("--save-interval", type=int, default=100)
    return result


@termination_signals()
def launch(args):
    # This entry point starts exactly one worker. RSL-RL otherwise detects an
    # inherited torchrun WORLD_SIZE and waits for workers we never launched.
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ValueError(
            "WORLD_SIZE and LOCAL_WORLD_SIZE must be integer 1 for weave"
        ) from exc
    if world_size != 1 or local_world_size != 1:
        raise ValueError(
            "weave supports one SDK worker; launch directly outside torchrun "
            "or another distributed process context"
        )
    repo = args.weave_root.resolve()
    lab = args.isaaclab_root.resolve()
    # Do not resolve the Python symlink: venv identity depends on its bin path.
    python = args.python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"SDK Python is not executable: {python}")
    for name in ("num_envs", "iterations", "save_interval"):
        if getattr(args, name, 1) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    check_checkout(repo, WEAVE_REVISION)
    check_checkout(lab, ISAACLAB_REVISION)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = {
        "command": args.command,
        "task": TASK,
        "weave_root": str(repo),
        "isaaclab_root": str(lab),
        "python": str(python),
        "output": str(output),
        "weave_revision": WEAVE_REVISION,
        "isaaclab_revision": ISAACLAB_REVISION,
        "device": args.device,
        "seed": args.seed,
        "headless": not args.gui,
        "num_envs": getattr(args, "num_envs", None),
        "iterations": getattr(args, "iterations", None),
        "save_interval": getattr(args, "save_interval", None),
        "checkpoint": None,
    }
    status = {"status": "preparing", "command": args.command}
    request_path = output / "request.json"
    try:
        write_json(request_path, request)
        write_json(output / "run.json", status)
        inputs = output / "inputs"
        inputs.mkdir()
        layout = json.loads(args.layout.read_text()) if args.layout else None
        request["motions"] = prepare_inputs(args.motions, layout, inputs, args.command)
        if args.command == "train" and args.num_envs < len(
            set(request["motions"]["objects"])
        ):
            raise ValueError("--num-envs must cover every distinct motion object")
        check_assets(repo, set(request["motions"]["objects"]))
        if args.checkpoint:
            checkpoint = inputs / "checkpoint.pt"
            before = sha256(args.checkpoint)
            shutil.copyfile(args.checkpoint, checkpoint)
            copied = sha256(checkpoint)
            if copied != before or sha256(args.checkpoint) != before:
                raise ValueError(
                    "Weave checkpoint changed during snapshot; stop its writer first"
                )
            request["checkpoint"] = {
                "path": str(checkpoint),
                "sha256": copied,
            }
        package = output / "implementation" / "embodiedforge"
        request["implementation"] = snapshot_implementation(
            Path(__file__).resolve().parent, package
        )
        write_json(request_path, request)
        environment = child_environment(python.parent.parent)
        for name in (
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "RANK",
            "LOCAL_RANK",
            "GROUP_RANK",
            "ROLE_RANK",
            "ROLE_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        ):
            environment.pop(name, None)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(package.parent), str(repo / "source/g1_hoi_learning"))
        )
        # Binary Isaac Sim installs need their own library and Python paths.
        command = [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'if [ -f "$1/_isaac_sim/setup_conda_env.sh" ]; then '
            'source "$1/_isaac_sim/setup_conda_env.sh" || exit; fi; '
            'exec "$2" -m embodiedforge._weave_worker "$3"',
            "weave",
            str(lab),
            str(python),
            str(request_path),
        ]
        status["status"] = "running"
        write_json(output / "run.json", status)
        run_process(
            command,
            cwd=output,
            env=environment,
            log_path=output / "worker.log",
        )
        report = json.loads((output / "result.json").read_text())
        if report.get("status") != "complete":
            raise RuntimeError(f"Weave worker did not complete: {report.get('status')}")
        status["status"] = "complete"
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
                # The worker publishes model_<iteration>.pt by atomic rename. Ignore
                # unfinished .tmp files, copied input weights, and unrelated files.
                latest = max(
                    (
                        path
                        for path in output.glob("model_*.pt")
                        if path.is_file() and path.stem[6:].isdecimal()
                    ),
                    key=lambda path: int(path.stem[6:]),
                    default=None,
                )
                status["latest_checkpoint"] = str(latest) if latest else None
                status["last_saved_iteration"] = (
                    int(latest.stem[6:]) if latest else None
                )
                status["checkpoint_validation"] = "not_performed" if latest else None
                if status["status"] == "complete" and str(latest) == report.get(
                    "checkpoint"
                ):
                    status["checkpoint_validation"] = "policy_and_optimizer"
            write_json(output / "run.json", status)
        except Exception as exc:
            if not active_error:
                raise
            print(f"Failed to finalize Weave run: {exc}", file=sys.stderr)
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(launch(args))
    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
