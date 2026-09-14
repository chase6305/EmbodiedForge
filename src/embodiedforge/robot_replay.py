"""Replay recorded Go1/H1 articulations through the shared Web viewer.

Example: python -m embodiedforge.robot_replay --model robot.xml --motion motion.npz
This reads recorded simulation states; it does not run a policy or new physics.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._h1_motion import load_motion
from .logging import get_logger, setup_logging
from .viewers.frames import OrbitCamera, frame_dependencies
from .viewers.robot import RobotSceneSpec, RobotSceneUpdate, load_robot_model
from .viewers.web import WebViewer


@dataclass(frozen=True)
class ReplayConfig:
    num_envs: int
    task: str
    physics: str = "记录回放"


class RobotMotion:
    """Map named joints and check every recorded body pose against model FK."""

    def __init__(self, path, model):
        import mujoco

        path = Path(path).resolve(strict=True)
        load_motion(path)  # Existing recording schema, time and terminal checks.
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            self.time, self.positions = data["time"].copy(), data["positions"].copy()
            quaternion, joints = data["quaternions"].copy(), data["joints"].copy()
            self.terminated = bool(data["terminated"][-1])
            self.truncated = bool(data["truncated"][-1])
        self.path = path
        meta = self.metadata
        if meta.get("schema", 1) != 1:
            raise ValueError("Unsupported motion schema")
        if meta.get("quaternion_order") != "xyzw":
            raise ValueError("Robot replay requires explicit xyzw quaternion order")
        for key in ("joint_names", "body_names"):
            names = meta[key]
            if (
                not names
                or not all(isinstance(n, str) and n for n in names)
                or len(set(names)) != len(names)
            ):
                raise ValueError(f"Motion {key} must be unique nonempty names")
        norms = np.linalg.norm(quaternion, axis=-1)
        if not np.allclose(norms, 1, atol=1e-3, rtol=0):
            raise ValueError("Motion quaternions must have unit norm")
        quaternion /= norms[..., None]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Replay requires exactly one floating root joint")
        root = int(model.jnt_bodyid[free[0]])
        if model.body(root).name != meta["body_names"][0]:
            raise ValueError(
                "Motion root body does not match the model's floating root"
            )
        ids = []
        for name in meta["joint_names"]:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0 or model.jnt_type[joint] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(f"Missing or incompatible model joint: {name}")
            ids.append(joint)
        if set(ids) != set(range(model.njnt)) - set(free):
            raise ValueError("Motion must specify every non-root model joint")
        self.qpos = np.tile(model.qpos0, (len(self.time), 1))
        start = model.jnt_qposadr[free[0]]
        self.qpos[:, start : start + 3] = self.positions[:, 0]
        self.qpos[:, start + 3 : start + 7] = np.roll(quaternion[:, 0], 1, axis=-1)
        self.qpos[:, model.jnt_qposadr[ids]] = joints
        point_types = meta.get("point_types", ["body"] * len(meta["body_names"]))
        if (
            len(point_types) != len(meta["body_names"])
            or point_types[0] != "body"
            or any(t not in ("body", "site") for t in point_types)
        ):
            raise ValueError("Invalid recorded point types")
        body_rows, body_ids = [], []
        for i, (name, kind) in enumerate(
            zip(meta["body_names"], point_types, strict=True)
        ):
            if kind != "body":
                continue
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body < 0:
                raise ValueError(f"Missing model body: {name}")
            body_rows.append(i)
            body_ids.append(body)
        data = mujoco.MjData(model)
        self.max_position_error = self.max_rotation_error = 0.0
        for i, qpos in enumerate(self.qpos):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            position_error = np.max(
                np.linalg.norm(
                    data.xpos[body_ids] - self.positions[i, body_rows], axis=-1
                )
            )
            expected = np.roll(quaternion[i, body_rows], 1, axis=-1)
            dots = np.abs(np.sum(data.xquat[body_ids] * expected, axis=-1))
            rotation_error = np.max(2 * np.arccos(np.clip(dots, -1, 1)))
            self.max_position_error = max(
                self.max_position_error, float(position_error)
            )
            self.max_rotation_error = max(
                self.max_rotation_error, float(rotation_error)
            )
            if position_error > 1e-3 or rotation_error > 2e-3:
                raise ValueError(
                    f"Motion/model pose mismatch at frame {i}: {position_error:.5g} m, {rotation_error:.5g} rad"
                )
        self.velocity = np.zeros_like(self.positions[:, 0], dtype=float)
        if len(self.time) > 1:
            self.velocity[1:] = (
                np.diff(self.positions[:, 0], axis=0) / np.diff(self.time)[:, None]
            )

    def details(self, index):
        command = self.metadata.get("velocity_command")
        segment_name = self.metadata.get("case", "")
        for segment in self.metadata.get("command_schedule", []):
            if segment["start_step"] <= index < segment["end_step"]:
                command, segment_name = segment["command"], segment["segment"]
                break
        return {
            "file": self.path.name,
            "frame": int(index),
            "frames": len(self.time),
            "duration": float(self.time[-1]),
            "segment": segment_name,
            "command": command,
            "ended": index == len(self.time) - 1,
            "terminal": "终止"
            if self.terminated
            else "截断"
            if self.truncated
            else "记录结束",
        }


class ReplaySession:
    """Per-recording cursors; scrubbing never interpolates across a terminal frame."""

    def __init__(self, motions):
        self.motions = motions
        self.indices = np.zeros(len(motions), dtype=int)
        self.playhead = np.array([m.time[0] for m in motions])

    def seek(self, env_id, frame):
        if type(env_id) is not int or not 0 <= env_id < len(self.motions):
            raise ValueError("Recording index out of range")
        if type(frame) is not int or not 0 <= frame < len(self.motions[env_id].time):
            raise ValueError("Frame index out of range")
        self.indices[env_id] = frame
        self.playhead[env_id] = self.motions[env_id].time[frame]

    def advance(self, seconds=None):
        if seconds is not None and (
            type(seconds) not in (int, float) or not np.isfinite(seconds) or seconds < 0
        ):
            raise ValueError("Playback interval must be finite and nonnegative")
        for i, motion in enumerate(self.motions):
            if seconds is None:
                self.seek(i, min(int(self.indices[i]) + 1, len(motion.time) - 1))
            else:
                self.playhead[i] = min(self.playhead[i] + seconds, motion.time[-1])
                self.indices[i] = max(
                    0, np.searchsorted(motion.time, self.playhead[i], side="right") - 1
                )

    @property
    def ended(self):
        return all(
            i == len(m.time) - 1
            for i, m in zip(self.indices, self.motions, strict=True)
        )

    def update(self, viewer):
        selected = viewer.selected_env
        motion, index = self.motions[selected], int(self.indices[selected])
        scene = RobotSceneUpdate(
            np.stack(
                [m.qpos[i] for m, i in zip(self.motions, self.indices, strict=True)]
            ),
            np.array(
                [m.time[i] for m, i in zip(self.motions, self.indices, strict=True)]
            ),
        )
        viewer.update(
            scene,
            {
                "episode_id": np.zeros(len(self.motions)),
                "step_id": self.indices,
                "time": scene.time,
            },
            np.zeros(len(self.motions)),
            display_state={
                "position": motion.positions[index, 0].tolist(),
                "velocity": float(np.linalg.norm(motion.velocity[index])),
                "reward": None,
                "replay": {**motion.details(index), "all_ended": self.ended},
            },
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Matching local MJCF robot asset"
    )
    parser.add_argument(
        "--motion",
        type=Path,
        nargs="+",
        required=True,
        help="One or more recorded NPZ trials of the same robot",
    )
    parser.add_argument(
        "--render-backend", choices=("mujoco", "gl", "rtx"), default="mujoco"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.fps <= 60:
        parser.error("port must be 1..65535 and fps must be 1..60")
    if (
        not np.isfinite(args.duration)
        or args.duration < 0
        or any(not 64 <= v <= 1920 for v in (args.width, args.height))
    ):
        parser.error("duration must be finite/nonnegative; dimensions must be 64..1920")
    setup_logging("INFO")
    logger = get_logger(__name__)
    spec = RobotSceneSpec(str(args.model.resolve()))
    report = frame_dependencies(args.render_backend, spec)
    if not report["metadata_ok"]:
        parser.error(f"Install embodiedforge[{report['install_extra']}]")
    model = load_robot_model(spec.model_path)
    motions = [RobotMotion(path, model) for path in args.motion]
    logger.info(
        "Validated %d recorded frames against MJCF; max position error %.6g m",
        sum(len(m.time) for m in motions),
        max(m.max_position_error for m in motions),
    )
    playback = ReplaySession(motions)
    config = ReplayConfig(
        len(motions), motions[0].metadata.get("title", "机器人运动回放")
    )
    viewer = WebViewer(
        config,
        spec,
        backend=args.render_backend,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        replay_frames=tuple(len(m.time) for m in motions),
        camera=OrbitCamera(distance=max(1.8, float(motions[0].positions[0, 0, 2]) * 3)),
    )
    viewer.paused = True
    try:
        logger.info(
            "Rendering robot first frame; initial RTX compilation may take longer"
        )
        playback.update(viewer)
        logger.info("Robot replay ready (paused)")
        started = previous = time.monotonic()
        while viewer.is_running() and (
            args.duration == 0 or time.monotonic() - started < args.duration
        ):
            now = time.monotonic()
            dt, previous = now - previous, now
            while (index := viewer.consume_reset()) is not None:
                playback.seek(index, 0)
            while (seek := viewer.consume_seek()) is not None:
                playback.seek(*seek)
            if viewer.should_step():
                playback.advance(
                    None if viewer.paused else min(dt, 0.1) * viewer.playback_speed
                )
            if playback.ended:
                viewer.paused = True
            if viewer.update_due(now):
                playback.update(viewer)
                # Loading a renderer is not playback time.
                if viewer.reconfigured:
                    previous = time.monotonic()
            time.sleep(max(0, min(0.005, viewer.next_frame_time - time.monotonic())))
    except KeyboardInterrupt:
        pass
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            viewer.close()
        except Exception:
            if not active_error:
                raise
            logger.exception(
                "Replay cleanup also failed; preserving the original error"
            )
        logger.info("Robot replay stopped")


if __name__ == "__main__":
    main()
