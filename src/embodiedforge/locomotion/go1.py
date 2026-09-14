# SPDX-License-Identifier: Apache-2.0
# Copyright 2026, Kevin Zakka
# Derived from mjbatch/examples/go1_joystick.py at
# b84c0c20aedbdf048122cbc47f554e9b93cc4754.
# Modified for EmbodiedForge: split task/learner, explicit runtime arguments,
# packaged scene, CPU execution, and no upstream example or viewer imports.

"""Go1 joystick task: model, reset, observations, controls and rewards."""

from pathlib import Path

import mujoco
import mujoco_menagerie as mm
import numpy as np
from mjbatch import Batch

from .go1_config import COMMAND_PROFILES, REWARD_PROFILES
from .go1_config import REWARD as REWARD

TIMESTEP, DECIMATION, KP, KD = 0.004, 5, 35.0, 0.5
ROTOR_INERTIA = 0.000111842  # kg m^2, the Go1 motor's rotor
CTRL_DT, ACTION_SCALE = TIMESTEP * DECIMATION, 0.5
FEET, FOOT_RADIUS = ("FR", "FL", "RR", "RL"), 0.023
STAND = np.array([0.1, 0.9, -1.8, -0.1, 0.9, -1.8] * 2)  # hip, thigh, calf per leg
POSE_W = np.array([1.0, 1.0, 0.1] * 4)
COMMAND_RANGE = np.array(COMMAND_PROFILES["original"]["range"])
COMMAND_ON = np.array(COMMAND_PROFILES["original"]["on"])
COMMAND_SECONDS, FRICTION = COMMAND_PROFILES["original"]["seconds"], (0.4, 1.0)
GAIT_HZ, SWING, CEILING, WIDTH = (
    2.0,
    0.06,
    0.075,
    0.035,
)  # trot Hz, swing m, ceiling m, width m
PHASE = np.array([0.0, 0.5, 0.5, 0.0])  # a trot: FR with RL, FL with RR
SIGMA = 0.25
FLOOR = 0.0
OBS_DIM, ACT_DIM = 50, 12
JOINTS, DOFS = slice(7, 7 + ACT_DIM), slice(6, 6 + ACT_DIM)
THEME = str(Path(__file__).with_name("go1_scene.xml"))
SEMANTICS = ("transition-v2", "upstream-v1")


def build_model():
    spec, robot = mujoco.MjSpec.from_file(THEME), mm.get("unitree_go1").spec("go1")
    robot.option.cone, robot.option.impratio = mujoco.mjtCone.mjCONE_PYRAMIDAL, 1.0
    spec.attach(robot, frame=spec.worldbody.add_frame(), prefix="")
    spec.delete(spec.light("spotlight"))
    spec.light("sun").mode = mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM
    spec.visual.global_.fovy = 38.0
    spec.visual.global_.bvactive = 0  # on by default in MuJoCo 3.11 and 3x slower
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.impratio = 1.0
    spec.option.solver = mujoco.mjtSolver.mjSOL_PGS
    spec.option.iterations = 10
    for geom in spec.geoms:
        if geom.name in FEET:
            geom.condim, geom.solimp = 3, [0.9, 0.95, 0.001, 0.5, 2.0]
            geom.margin = (
                0.0  # the stock 1 mm margin puts contacts outside the touch sensor
            )
        elif geom.name != "floor":
            geom.contype = geom.conaffinity = 0
    for joint in spec.joints[1:]:
        gear = 9 if joint.name.endswith("calf_joint") else 6
        joint.damping, joint.frictionloss = [0.0, 0.0, 0.0], 0.0
        joint.armature = ROTOR_INERTIA * gear**2
    for actuator in spec.actuators:
        actuator.set_to_position(kp=KP, kv=KD)
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    kind, site, body = (
        mujoco.mjtSensor,
        mujoco.mjtObj.mjOBJ_SITE,
        mujoco.mjtObj.mjOBJ_XBODY,
    )
    spec.add_sensor(name="gyro", type=kind.mjSENS_GYRO, objtype=site, objname="imu")
    spec.add_sensor(
        name="vel", type=kind.mjSENS_VELOCIMETER, objtype=site, objname="imu"
    )
    up = kind.mjSENS_FRAMEZAXIS  # the world's z axis in the trunk frame
    spec.add_sensor(
        name="up", type=up, objtype=body, objname="world", reftype=body, refname="trunk"
    )
    for foot in FEET:
        spec.add_sensor(
            name=foot, type=kind.mjSENS_FRAMELINVEL, objtype=site, objname=foot
        )
        spec.add_sensor(
            name=foot + "_touch", type=kind.mjSENS_TOUCH, objtype=site, objname=foot
        )
    model = spec.compile()
    sun = model.light("sun").id
    model.light_poscom0[sun] = -6.0 * model.light_dir0[sun]
    return model


class Go1:
    """Batched joystick task with explicit reset and no automatic episode freeze."""

    def __init__(
        self,
        num_envs,
        seed=0,
        *,
        num_threads=4,
        episode_steps=500,
        semantics="transition-v2",
        reward_profile="original",
        command_profile="original",
    ):
        if semantics not in SEMANTICS:
            raise ValueError(f"Unknown Go1 task semantics: {semantics}")
        self.semantics = semantics
        if reward_profile not in REWARD_PROFILES:
            raise ValueError(f"Unknown Go1 reward profile: {reward_profile}")
        self.reward_profile = reward_profile
        self.reward_weights = REWARD_PROFILES[reward_profile].copy()
        if command_profile not in COMMAND_PROFILES:
            raise ValueError(f"Unknown Go1 command profile: {command_profile}")
        self.command_profile = command_profile
        distribution = COMMAND_PROFILES[command_profile]
        self.command_range = np.array(distribution["range"])
        self.command_on = np.array(distribution["on"])
        self.command_seconds = distribution["seconds"]
        self.command_keep = distribution.get("keep", 0.5)
        self.command_stop_probability = distribution.get("stop_probability", 0.0)
        if (
            type(num_envs) is not int
            or num_envs <= 0
            or type(episode_steps) is not int
            or episode_steps <= 0
            or type(num_threads) is not int
            or num_threads <= 0
            or type(seed) is not int
            or seed < 0
        ):
            raise ValueError(
                "Go1 counts and threads must be positive integers; seed must be a nonnegative integer"
            )
        self.episode_steps = episode_steps
        self.batch = batch = Batch(build_model(), num_envs, num_threads=num_threads)
        self.qpos, self.qvel, self.ctrl = (
            batch.bind(f) for f in ("qpos", "qvel", "ctrl")
        )
        self.gyro, self.vel, self.up = (batch.sensor(s) for s in ("gyro", "vel", "up"))
        self.foot_vel = [batch.sensor(f) for f in FEET]
        self.foot_force = [batch.sensor(f + "_touch") for f in FEET]
        self.torque = batch.bind("actuator_force")
        self.foot_pos = [batch.site(f).xpos for f in FEET]
        self.trunk = batch.body("trunk").xpos
        self.friction = batch.expand("geom_friction")  # per env, redrawn at reset
        self.feet = [batch.model.geom(f).id for f in FEET]
        limits = batch.model.jnt_range[1 : 1 + ACT_DIM]
        self.lower, self.upper = 0.95 * limits[:, 0], 0.95 * limits[:, 1]
        self.rng = np.random.default_rng(seed)
        self.steps, self.clock = np.zeros(num_envs, np.int64), np.zeros(num_envs)
        self.action = np.zeros((num_envs, ACT_DIM), np.float32)
        self.down = np.ones((num_envs, 4), bool)
        self.command, self.until = np.zeros((num_envs, 3)), np.zeros(num_envs)
        self.reset(np.arange(num_envs))

    def _ids(self, ids):
        if ids is None:
            return np.arange(len(self.steps), dtype=np.int64)
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError("Go1 environment IDs must be one-dimensional")
        if not ids.size:
            return np.empty(0, dtype=np.int64)
        if (
            ids.dtype.kind not in "iu"
            or (ids < 0).any()
            or (ids >= len(self.steps)).any()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Go1 environment IDs must be unique integers in range")
        return np.ascontiguousarray(ids, dtype=np.int64)

    def reset(self, ids=None):
        """Reset selected rows, drawing random values in caller-supplied order.

        None resets all rows; an empty selection is a no-op. Invalid IDs are
        rejected before changing physics or consuming random numbers.
        """
        ids = self._ids(ids)
        n = ids.size
        if not n:
            return
        # Native calls require sorted IDs; NumPy writes preserve caller order.
        selected = np.sort(ids)
        self.batch.reset(selected, keyframe=0)
        self.qpos[ids, JOINTS] = STAND + self.rng.uniform(-0.1, 0.1, (n, ACT_DIM))
        self.qvel[ids, : DOFS.stop] = self.rng.uniform(-0.2, 0.2, (n, DOFS.stop))
        self.friction[ids[:, None], self.feet, 0] = self.rng.uniform(*FRICTION, (n, 1))
        self.batch.forward(selected)  # sensordata is stale until forward runs
        # The keyframe stands 2.4 cm lower than this pose needs, which buries the feet.
        self.qpos[ids, 2] -= (
            np.stack(self.foot_pos, 1)[ids, :, 2].min(1) - FOOT_RADIUS - 0.001
        )
        self.batch.forward(selected)
        self.clock[ids] = self.rng.uniform(0.0, 1.0, n)
        self.resample(ids, keep=0.0)
        self.steps[ids], self.action[ids], self.down[ids] = 0, 0.0, True

    def resample(self, ids, keep=None):
        ids = self._ids(ids)
        if keep is None:
            keep = self.command_keep
        if not np.isfinite(keep) or not 0 <= keep <= 1:
            raise ValueError("Go1 command retention probability must be in [0, 1]")
        n = ids.size
        if not n:
            return
        fresh = self.rng.uniform(-1.0, 1.0, (n, 3)) * self.command_range
        fresh *= self.rng.random((n, 3)) < self.command_on
        kept = self.rng.random((n, 3)) < keep
        self.command[ids] = np.where(kept, self.command[ids], fresh)
        # Whole-vector stops also override retained axes, so the policy sees
        # transitions into standing rather than only standing initial states.
        if self.command_stop_probability:
            stop = self.rng.random(n) < self.command_stop_probability
            self.command[ids[stop]] = 0.0
        self.until[ids] = self.rng.exponential(self.command_seconds / CTRL_DT, n)

    def moving(self):
        return (np.linalg.norm(self.command, axis=1) > 0.01)[:, None]

    def obs(self):
        angle, moving = 2 * np.pi * self.clock[:, None], self.moving()
        cols = (
            self.vel,
            self.gyro,
            self.up,
            self.qpos[:, JOINTS] - STAND,
            self.qvel[:, DOFS],
            self.action,
            self.command,
            np.sin(angle)
            * moving,  # hidden at rest, or it marches waiting for a command
            np.cos(angle) * moving,
        )
        return np.concatenate(cols, 1, dtype=np.float32)

    def step(self, action):
        action = np.asarray(action)
        if (
            action.shape != self.action.shape
            or action.dtype.kind not in "fiu"
            or not np.isfinite(action).all()
            or (np.abs(action.astype(np.float64)) > np.finfo(np.float32).max).any()
        ):
            raise ValueError("Expected finite Go1 actions with shape (num_envs, 12)")
        self.ctrl[:] = STAND + ACTION_SCALE * action
        flying = np.stack(self.foot_vel, 1).copy()
        self.batch.step(nstep=DECIMATION)
        self.steps += 1
        self.until -= 1
        if self.semantics == "upstream-v1":
            self.resample(np.flatnonzero(self.until <= 0))
        q, up, gyro, command = self.qpos[:, JOINTS], self.up, self.gyro, self.command
        force = np.concatenate(self.foot_force, 1)
        down = force > 0.0
        contact = down | self.down  # a bounce does not end the stance
        height = np.stack(self.foot_pos, 1)[:, :, 2] - FOOT_RADIUS
        slide = np.sum(np.stack(self.foot_vel, 1)[:, :, :2] ** 2, 2)
        moving = self.moving()
        phase = np.sin(2 * np.pi * (self.clock[:, None] + PHASE)) * moving
        low = np.minimum(height - SWING * phase, 0.0)
        high = np.maximum(height - CEILING, 0.0)
        swing = np.exp(-force / 10.0) * np.exp(-((low**2 + high**2) / WIDTH**2))
        blend = np.clip(0.5 + phase, 0.0, 1.0) * moving  # a ramp, not a switch
        over = np.maximum(self.lower - q, 0.0) + np.maximum(q - self.upper, 0.0)
        terms = dict(
            track=np.exp(-np.sum((self.vel[:, :2] - command[:, :2]) ** 2, 1) / SIGMA),
            turn=np.exp(-((gyro[:, 2] - command[:, 2]) ** 2) / SIGMA),
            gait=(blend * swing + (1.0 - blend) * contact * np.exp(-slide / 0.05)).mean(
                1
            ),
            pose=np.exp(-np.sum((q - STAND) ** 2 * POSE_W, 1)),
            orient=up[:, 0] ** 2 + up[:, 1] ** 2,
            bounce=self.qvel[:, 2] ** 2,
            wobble=gyro[:, 0] ** 2 + gyro[:, 1] ** 2,
            limits=np.sum(over, 1),
            rate=np.sum((action - self.action) ** 2, 1),
            land=np.sum((down & ~self.down) * np.sum(flying**2, 2), 1),
        )
        reward = np.maximum(
            np.asarray(
                sum(self.reward_weights[k] * v for k, v in terms.items()), np.float32
            ),
            FLOOR,
        )
        self.clock = (self.clock + GAIT_HZ * CTRL_DT) % 1.0
        self.action, self.down = action.astype(np.float32), down
        fell = up[:, 2] < 0.0
        if self.semantics == "transition-v2":
            # Credit the action against the command in its input observation.
            # Publish a new command only for the next observation/action pair.
            self.resample(np.flatnonzero(self.until <= 0))
        return reward, fell | (self.steps >= self.episode_steps), fell, terms
