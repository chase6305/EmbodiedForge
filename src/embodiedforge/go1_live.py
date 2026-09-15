"""Run a managed Go1 checkpoint with live velocity controls in the shared Web UI."""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import numpy as np

from ._go1_assets import recorded_go1_assets, reuse_asset_cache
from ._live_channel import JsonChannel
from .logging import get_logger, setup_logging
from .recipes import checkpoint_input, sha256
from .viewers.frames import OrbitCamera, frame_dependencies
from .viewers.robot import RobotSceneSpec, RobotSceneUpdate
from .viewers.timing import RateClock
from .viewers.web import WebViewer


@dataclass(frozen=True)
class LiveConfig:
    num_envs: int
    task: str = "Go1 在线策略"
    physics: str = "mjbatch / MuJoCo"


class LivePolicyProcess:
    """Keep the training interpreter, checkpoint copy and compiled model private."""

    def __init__(
        self, run, *, python=None, num_envs=1, seed=0, threads=2, episode_steps=3000
    ):
        for name, value, maximum in [
            ("num_envs", num_envs, 64),
            ("threads", threads, 64),
            ("episode_steps", episode_steps, 1_000_000),
        ]:
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        run = Path(run).resolve(strict=True)
        checkpoint, manifest = checkpoint_input(run, "go1-joystick")
        # Keep the venv entry path: resolving its Python symlink would select
        # the base interpreter and lose the training environment's packages.
        executable = (
            Path(python or manifest["result"]["runtime"]["executable"])
            .expanduser()
            .absolute()
        )
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Training Python executable is missing or not executable")
        self.directory = tempfile.TemporaryDirectory(prefix="embodiedforge-go1-live-")
        self.process = self.socket = None
        self.closed = False
        self.run = run.name
        self.model_path = Path(self.directory.name) / "model.mjb"
        self.contract = dict(manifest["result"])
        self.contract["assets"] = recorded_go1_assets(run, self.contract)
        try:
            copied = Path(self.directory.name) / "checkpoint.pt"
            shutil.copyfile(checkpoint, copied)
            if sha256(copied) != self.contract["checkpoint_sha256"]:
                raise ValueError("Copied checkpoint SHA256 mismatch")
            self.socket, child = socket.socketpair()
            self.socket.settimeout(120)
            environment = dict(os.environ)
            reuse_asset_cache(environment, self.contract.get("assets"))
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            try:
                self.process = subprocess.Popen(
                    [
                        str(executable),
                        "-m",
                        "embodiedforge._go1_live_worker",
                        "--fd",
                        str(child.fileno()),
                    ],
                    pass_fds=(child.fileno(),),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                )
            finally:
                child.close()
            self.channel = JsonChannel(self.socket)
            self.channel.send(
                {
                    "checkpoint": str(copied),
                    "sha256": self.contract["checkpoint_sha256"],
                    "model_path": str(self.model_path),
                    "contract": self.contract,
                    "num_envs": num_envs,
                    "seed": seed,
                    "threads": threads,
                    "episode_steps": episode_steps,
                }
            )
            result = self._receive()
            self.metadata, self.state = result["metadata"], result["state"]
            if self.metadata["mujoco"] != metadata.version("mujoco"):
                raise ValueError(
                    "Policy and renderer require the same MuJoCo version for compiled models"
                )
            if sha256(self.model_path) != self.metadata["model_sha256"]:
                raise ValueError("Compiled model SHA256 mismatch")
            self.socket.settimeout(30)
        except BaseException:
            self.close()
            raise

    def _receive(self):
        value = self.channel.receive()
        if not value["ok"]:
            raise RuntimeError(value["error"])
        return value

    def request(self, action, **kwargs):
        if self.closed:
            raise RuntimeError("Policy session is closed")
        self.channel.send({"action": action, **kwargs})
        self.state = self._receive()["state"]
        return self.state

    @property
    def blocked(self):
        return any(self.state["done"])

    def update(self, viewer):
        s, i = self.state, viewer.selected_env
        viewer.update(
            RobotSceneUpdate(np.asarray(s["qpos"]), np.asarray(s["time"])),
            {
                "episode_id": np.asarray(s["episode"]),
                "step_id": np.asarray(s["step"]),
                "time": np.asarray(s["time"]),
            },
            np.asarray(s["reward"]),
            display_state={
                "position": s["position"][i],
                "velocity": float(np.linalg.norm(s["velocity"][i])),
                "reward": s["reward"][i],
                "live": {
                    "run": self.run,
                    "iteration": self.metadata["iteration"],
                    "checkpoint_sha256": self.metadata["checkpoint_sha256"],
                    "command": s["command"][i],
                    "command_limits": self.metadata["command_limits"],
                    "measured_velocity": [*s["velocity"][i][:2], s["yaw_velocity"][i]],
                    "blocked_envs": [j for j, v in enumerate(s["done"]) if v],
                    "done": s["done"][i],
                    "fell": s["fell"][i],
                },
            },
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process is not None:
                if self.process.poll() is None and self.socket is not None:
                    try:
                        self.socket.settimeout(0.2)
                        JsonChannel(self.socket).send({"action": "close"})
                    except OSError:
                        pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=2)
        finally:
            if self.socket is not None:
                self.socket.close()
            self.directory.cleanup()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Completed managed go1-joystick training run",
    )
    parser.add_argument(
        "--worker-python",
        type=Path,
        help="Training SDK interpreter; defaults to the run manifest",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--episode-steps", type=int, default=3000)
    parser.add_argument(
        "--render-backend", choices=("mujoco", "gl", "rtx"), default="mujoco"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args(argv)
    if (
        not 1 <= args.port <= 65535
        or not 1 <= args.fps <= 60
        or any(not 64 <= n <= 1920 for n in (args.width, args.height))
    ):
        parser.error("Invalid port, fps or image dimensions")
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative")
    report = frame_dependencies(args.render_backend, RobotSceneSpec(""))
    if not report["metadata_ok"]:
        parser.error(f"Install embodiedforge[{report['install_extra']}]")
    setup_logging("INFO")
    logger = get_logger(__name__)
    policy = LivePolicyProcess(
        args.run,
        python=args.worker_python,
        num_envs=args.num_envs,
        seed=args.seed,
        threads=args.threads,
        episode_steps=args.episode_steps,
    )
    viewer = None
    try:
        viewer = WebViewer(
            LiveConfig(args.num_envs),
            RobotSceneSpec(str(policy.model_path)),
            backend=args.render_backend,
            host=args.host,
            port=args.port,
            width=args.width,
            height=args.height,
            fps=args.fps,
            camera=OrbitCamera(distance=1.8),
            velocity_limits=tuple(policy.metadata["command_limits"]),
        )
        viewer.paused = True
        policy.update(viewer)
        logger.info(
            "Live Go1 ready: %s (paused, iteration=%s)",
            viewer.url,
            policy.metadata["iteration"],
        )
        started = time.monotonic()
        physics_clock = RateClock()
        while viewer.is_running() and (
            args.duration == 0 or time.monotonic() - started < args.duration
        ):
            tick = time.monotonic()
            while (index := viewer.consume_reset()) is not None:
                policy.request("reset", env_id=index)
            while (command := viewer.consume_velocity()) is not None:
                index, value = command
                policy.request("velocity", env_id=index, value=value)
            if policy.blocked:
                viewer.paused = True
            if (
                physics_clock.due(tick, viewer.playback_speed / policy.metadata["dt"])
                and viewer.should_step()
                and not policy.blocked
            ):
                policy.request("step")
                if policy.blocked:
                    viewer.paused = True
            if viewer.update_due(tick):
                policy.update(viewer)
            time.sleep(
                max(
                    0,
                    min(
                        0.02,
                        min(physics_clock.next, viewer.next_frame_time)
                        - time.monotonic(),
                    ),
                )
            )
    except KeyboardInterrupt:
        pass
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            try:
                if viewer is not None:
                    viewer.close()
            finally:
                policy.close()
        except Exception:
            if not active_error:
                raise
            logger.exception(
                "Live session cleanup also failed; preserving the original error"
            )
        logger.info("Live Go1 stopped")


if __name__ == "__main__":
    main()
