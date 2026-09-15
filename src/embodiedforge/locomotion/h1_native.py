# SPDX-License-Identifier: BSD-3-Clause
# H1 configuration/reward concepts adapted from IsaacLab (2022-2026 contributors).
# See LICENSE.IsaacLab. The runtime below is an independent MuJoCo implementation.
"""Native H1 flat-ground task, with explicit transitions and selective reset."""

from pathlib import Path

import mujoco
import numpy as np

from .articulation import ArticulationBatch, finite_array, selected_ids

OBS_DIM, ACT_DIM = 69, 19
TIMESTEP, DECIMATION, ACTION_SCALE = 0.005, 4, 0.5
CTRL_DT = TIMESTEP * DECIMATION
VERSION = "h1-mjbatch-flat-v1"


def joint_defaults(name):
    if name.endswith("hip_pitch"):
        return -0.28, 200, 5, 300
    if name.endswith("knee"):
        return 0.79, 200, 5, 300
    if name.endswith("ankle"):
        return -0.52, 20, 4, 100
    if "hip_" in name:
        return 0, 150, 5, 300
    if name == "torso":
        return 0, 200, 5, 300
    if "shoulder_" in name or name.endswith("elbow"):
        return (
            (
                0.28
                if name.endswith("shoulder_pitch")
                else 0.52
                if name.endswith("elbow")
                else 0
            ),
            40,
            10,
            300,
        )
    raise ValueError(f"Unknown H1 joint: {name}")


def build_model(path):
    spec = mujoco.MjSpec.from_file(str(Path(path).resolve(strict=True)))
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in spec.geoms):
        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(0, 0, 0.1),
            contype=1,
            conaffinity=2,
        )
    # Separate robot/floor collision bits, disabling robot self-collision.
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            geom.contype, geom.conaffinity = 1, 2
        elif geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 2, 1
    if len(list(spec.actuators)) != ACT_DIM:
        raise ValueError("H1 requires exactly 19 joint actuators")
    for actuator in spec.actuators:
        _, kp, kd, limit = joint_defaults(actuator.target)
        actuator.set_to_position(kp=kp, kv=kd)
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        actuator.forcerange = (-limit, limit)
        actuator.gear = (1, 0, 0, 0, 0, 0)
    for joint in spec.joints:
        if joint.type != mujoco.mjtJoint.mjJNT_FREE:
            joint.damping = (0, 0, 0)
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.visual.global_.bvactive = 0
    for name, kind in (
        ("native_velocity", mujoco.mjtSensor.mjSENS_VELOCIMETER),
        ("native_gyro", mujoco.mjtSensor.mjSENS_GYRO),
    ):
        spec.add_sensor(
            name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu"
        )
    for body_name in ("left_ankle_link", "right_ankle_link", "torso_link"):
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"Missing H1 body: {body_name}")
        site = body.add_site(
            name="native_touch_" + body_name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(0.22, 0.12, 0.10) if "ankle" in body_name else (0.2, 0.2, 0.45),
            pos=(0.06, 0, -0.05) if "ankle" in body_name else (0, 0, 0.3),
            group=5,
        )
        spec.add_sensor(
            name=site.name,
            type=mujoco.mjtSensor.mjSENS_TOUCH,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site.name,
        )
        spec.add_sensor(
            name="native_vel_" + body_name,
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=site.name,
        )
    model = spec.compile()
    if (
        model.nq != 26
        or model.nv != 25
        or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
    ):
        raise ValueError("H1 model must have one floating root and 19 scalar joints")
    return model


class H1:
    def __init__(
        self,
        model,
        num_envs=64,
        *,
        seed=0,
        threads=4,
        episode_steps=1000,
        training=True,
    ):
        if type(episode_steps) is not int or episode_steps <= 0:
            raise ValueError("Episode length must be a positive integer")
        self.robot = ArticulationBatch(model, num_envs, seed=seed, threads=threads)
        self.model, self.batch = model, self.robot.batch
        self.num_envs, self.training, self.episode_steps = (
            num_envs,
            training,
            episode_steps,
        )
        # qpos0 is MuJoCo's kinematic reference, not a configurable standing pose.
        self.default = np.array([joint_defaults(name)[0] for name in self.robot.names])
        self.groups = {
            "legs": self.robot.group(".*_hip_.*", ".*_knee", "torso"),
            "feet": self.robot.group(".*_ankle"),
            "arms": self.robot.group(".*_shoulder_.*", ".*_elbow"),
            "hip": self.robot.group(".*_hip_yaw", ".*_hip_roll"),
            "torso": self.robot.group("torso"),
        }
        self.velocity, self.gyro = [
            self.batch.sensor(n) for n in ("native_velocity", "native_gyro")
        ]
        self.rotation = self.batch.bind("xmat")[:, model.body("torso_link").id].reshape(
            num_envs, 3, 3
        )
        self.commands = np.zeros((num_envs, 3))
        self.fixed_commands = np.zeros(num_envs, bool)
        self.steps = np.zeros(num_envs, np.int64)
        self.action = np.zeros((num_envs, ACT_DIM))
        self.air = np.zeros((num_envs, 2))
        self.contact_time = np.zeros_like(self.air)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def set_commands(self, values, env_ids=None):
        ids = selected_ids(env_ids, self.num_envs)
        commands = finite_array(values, (len(ids), 3), "Velocity commands")
        if (np.abs(commands) > (1, 1, 1)).any():
            raise ValueError("Velocity commands must lie in [-1, 1]")
        self.commands[ids], self.fixed_commands[ids] = commands, True

    def resample(self, ids):
        self.commands[ids] = self.rng.uniform((0, 0, -1), (1, 0, 1), (len(ids), 3))
        self.commands[ids[self.rng.random(len(ids)) < 0.02]] = 0

    def reset(self, env_ids=None):
        ids = selected_ids(env_ids, self.num_envs)
        if not len(ids):
            return self.obs()
        qpos = np.tile(self.model.qpos0, (len(ids), 1))
        qpos[:, self.robot.qadr] = self.default
        qpos[:, :3] = (0, 0, 1.05)
        qpos[:, 3:7] = (1, 0, 0, 0)
        qvel = np.zeros((len(ids), self.model.nv))
        if self.training:
            self.robot.randomize(ids, body_ids=[self.model.body("torso_link").id])
            qpos[:, :2] = self.rng.uniform(-0.5, 0.5, (len(ids), 2))
            yaw = self.rng.uniform(-np.pi, np.pi, len(ids))
            qpos[:, 3], qpos[:, 6] = np.cos(yaw / 2), np.sin(yaw / 2)
            qvel[:, :6] = self.rng.uniform(-0.5, 0.5, (len(ids), 6))
        self.robot.reset(ids, qpos=qpos, qvel=qvel)
        self.steps[ids], self.action[ids], self.air[ids], self.contact_time[ids] = (
            0,
            0,
            0,
            0,
        )
        self.resample(ids[~self.fixed_commands[ids]])
        return self.obs()

    def obs(self):
        observation = np.concatenate(
            (
                self.velocity,
                self.gyro,
                -self.rotation[:, 2, :],
                self.commands,
                self.robot.qpos[:, self.robot.qadr] - self.default,
                self.robot.qvel[:, self.robot.vadr],
                self.action,
            ),
            axis=1,
        )
        if self.training:
            scale = np.r_[
                np.full(3, 0.1),
                np.full(3, 0.2),
                np.full(3, 0.05),
                np.zeros(3),
                np.full(19, 0.01),
                np.full(19, 1.5),
                np.zeros(19),
            ]
            observation += self.rng.uniform(-1, 1, observation.shape) * scale
        return observation.astype(np.float32)

    def step(self, actions):
        actions = np.clip(
            finite_array(actions, (self.num_envs, ACT_DIM), "Actions"), -5, 5
        )
        previous_velocity = self.robot.qvel[:, self.robot.vadr].copy()
        self.robot.set_joint_position_targets(
            self.default + ACTION_SCALE * np.clip(actions, -5, 5)
        )
        self.robot.step(DECIMATION)
        self.steps += 1
        foot_force = np.concatenate(
            [
                self.batch.sensor("native_touch_" + n)
                for n in ("left_ankle_link", "right_ankle_link")
            ],
            axis=1,
        )
        contact = foot_force > 1
        self.air = np.where(contact, 0, self.air + CTRL_DT)
        self.contact_time = np.where(contact, self.contact_time + CTRL_DT, 0)
        single = contact.sum(1) == 1
        air_reward = (
            np.minimum(np.where(contact, self.contact_time, self.air).min(1), 0.6)
            * single
            * (np.linalg.norm(self.commands[:, :2], axis=1) > 0.1)
        )
        yaw = np.arctan2(self.rotation[:, 1, 0], self.rotation[:, 0, 0])
        # MuJoCo root linear velocity is world-aligned; rewards use a yaw frame.
        world_vel = self.robot.qvel[:, :3]
        velocity = np.column_stack(
            (
                np.cos(yaw) * world_vel[:, 0] + np.sin(yaw) * world_vel[:, 1],
                -np.sin(yaw) * world_vel[:, 0] + np.cos(yaw) * world_vel[:, 1],
            )
        )
        world_gyro = np.einsum("nij,nj->ni", self.rotation, self.gyro)
        self.measured_velocity = np.column_stack((velocity, world_gyro[:, 2]))
        fell = self.batch.sensor("native_touch_torso_link")[:, 0] > 1
        # Native safeguard also terminates a tipped/sub-floor base, including a
        # fall whose torso contact occurred between the sampled control ticks.
        fell |= (self.robot.qpos[:, 2] < 0.45) | (self.rotation[:, 2, 2] < 0.2)
        q = self.robot.qpos[:, self.robot.qadr]
        joint_ids = self.model.actuator_trnid[:, 0]
        limits = self.model.jnt_range[joint_ids]
        middle, half = limits.mean(1), np.diff(limits, axis=1)[:, 0] * 0.45
        excess = np.maximum(abs(q - middle) - half, 0)[:, self.groups["feet"]].sum(1)
        slide = sum(
            np.linalg.norm(self.batch.sensor("native_vel_" + n)[:, :2], axis=1)
            * contact[:, i]
            for i, n in enumerate(("left_ankle_link", "right_ankle_link"))
        )
        terms = {
            "track_linear": np.exp(
                -np.sum((velocity - self.commands[:, :2]) ** 2, 1) / 0.25
            ),
            "track_yaw": np.exp(
                -((world_gyro[:, 2] - self.commands[:, 2]) ** 2) / 0.25
            ),
            "air_time": air_reward,
            "slide": -0.25 * slide,
            "orientation": -np.sum(self.rotation[:, 2, :2] ** 2, 1),
            "angular_xy": -0.05 * np.sum(self.gyro[:, :2] ** 2, 1),
            "acceleration": -1.25e-7
            * np.sum(
                ((self.robot.qvel[:, self.robot.vadr] - previous_velocity) / CTRL_DT)
                ** 2,
                1,
            ),
            "action_rate": -0.005 * np.sum((actions - self.action) ** 2, 1),
            "ankle_limits": -excess,
            "termination": -200 * fell.astype(float),
        }
        for name, weight in (("hip", 0.2), ("arms", 0.2), ("torso", 0.1)):
            terms[name + "_deviation"] = -weight * np.abs(q - self.default)[
                :, self.groups[name]
            ].sum(1)
        reward = (sum(terms.values()) * CTRL_DT).astype(np.float32)
        self.action[:] = actions
        # Changing the command happens after crediting the completed transition.
        if self.training:
            self.resample(
                np.flatnonzero((self.steps % 500 == 0) & ~self.fixed_commands)
            )
        truncated = (self.steps >= self.episode_steps) & ~fell
        return self.obs(), reward, fell, truncated, terms
