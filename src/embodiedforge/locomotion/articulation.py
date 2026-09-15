"""Named, batched MJCF articulation operations without IsaacLab imports."""

import re

import mujoco
import numpy as np
from mjbatch import Batch


def selected_ids(values, count):
    if values is None:
        return np.arange(count, dtype=np.int64)
    ids = np.asarray(values)
    if ids.ndim != 1 or (
        ids.size
        and (
            ids.dtype.kind not in "iu"
            or (ids < 0).any()
            or (ids >= count).any()
            or len(np.unique(ids)) != len(ids)
        )
    ):
        raise ValueError(
            "IDs must be a one-dimensional collection of unique valid integers"
        )
    return np.ascontiguousarray(ids, dtype=np.int64)


def finite_array(value, shape, name):
    array = np.asarray(value)
    if (
        array.shape != shape
        or array.dtype.kind not in "fiu"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.astype(np.float64)


class ArticulationBatch:
    """One MJCF topology, independent simulation rows, scalar joint actuators.

    snapshot() returns owned arrays; internal integration buffers are live views.
    Joint IDs index actuator order, never
    assume qpos order. Raw state uses MuJoCo qpos/qvel conventions (wxyz quats).
    """

    def __init__(self, model, num_envs, *, threads=4, seed=0):
        if any(type(v) is not int or v <= 0 for v in (num_envs, threads)):
            raise ValueError("Environment and thread counts must be positive integers")
        if type(seed) is not int or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        joint_ids = model.actuator_trnid[:, 0]
        if not model.nu or (model.actuator_trntype != mujoco.mjtTrn.mjTRN_JOINT).any():
            raise ValueError("Articulations require scalar joint actuators")
        if (
            len(set(joint_ids)) != model.nu
            or not np.isin(
                model.jnt_type[joint_ids],
                [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE],
            ).all()
        ):
            raise ValueError("Each actuator must address a distinct hinge/slide joint")
        if (
            (model.actuator_gaintype != mujoco.mjtGain.mjGAIN_FIXED).any()
            or (model.actuator_biastype != mujoco.mjtBias.mjBIAS_AFFINE).any()
            or not np.allclose(
                model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 1]
            )
            or (model.actuator_gainprm[:, 0] <= 0).any()
            or not np.allclose(
                model.actuator_gear, np.tile((1, 0, 0, 0, 0, 0), (model.nu, 1))
            )
        ):
            raise ValueError(
                "Position control requires positive-gain, unit-gear position actuators"
            )
        self.model, self.num_envs = model, num_envs
        self.names = tuple(model.joint(int(i)).name for i in joint_ids)
        self.qadr, self.vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
        self.batch = Batch(model, num_envs, num_threads=threads)
        self.qpos, self.qvel, self.ctrl = [
            self.batch.bind(f) for f in ("qpos", "qvel", "ctrl")
        ]
        self._actuator_force = self.batch.bind("actuator_force")
        self._joint_force = self.batch.bind("qfrc_actuator")
        # mjbatch only refreshes fields bound before forward/step. Bind poses
        # eagerly so a first recording/snapshot cannot read zero-filled buffers.
        self.body_position = self.batch.bind("xpos")
        self.body_quaternion = self.batch.bind("xquat")
        self.actuator_force = np.zeros((num_envs, model.nu))
        self.joint_force = np.zeros_like(self.actuator_force)
        self.rng = np.random.default_rng(seed)
        self.batch.reset()

    def group(self, *patterns):
        """Union of full regular-expression matches, in stable actuator order."""
        if not patterns:
            raise ValueError("At least one joint pattern is required")
        selected = set()
        for pattern in patterns:
            found = {
                i for i, name in enumerate(self.names) if re.fullmatch(pattern, name)
            }
            if not found:
                raise ValueError(f"Joint pattern matches nothing: {pattern}")
            selected.update(found)
        return np.array(sorted(selected), dtype=np.int64)

    def set_joint_position_targets(self, values, *, env_ids=None, joint_ids=None):
        ids = selected_ids(env_ids, self.num_envs)
        joints = selected_ids(joint_ids, self.model.nu)
        targets = finite_array(values, (len(ids), len(joints)), "Position targets")
        self.ctrl[np.ix_(ids, joints)] = targets

    def reset(self, env_ids=None, *, qpos=None, qvel=None):
        """Validate all inputs before resetting; selected row order is respected."""
        ids = selected_ids(env_ids, self.num_envs)
        position = (
            np.tile(self.model.qpos0, (len(ids), 1))
            if qpos is None
            else finite_array(qpos, (len(ids), self.model.nq), "qpos")
        )
        velocity = (
            np.zeros((len(ids), self.model.nv))
            if qvel is None
            else finite_array(qvel, (len(ids), self.model.nv), "qvel")
        )
        for j in range(self.model.njnt):
            kind, adr = self.model.jnt_type[j], self.model.jnt_qposadr[j]
            if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
                start = adr + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
                if not np.allclose(
                    np.linalg.norm(position[:, start : start + 4], axis=1), 1, atol=1e-5
                ):
                    raise ValueError("Reset quaternions must have unit length")
        if not len(ids):
            return
        self.batch.reset(np.sort(ids))
        self.qpos[ids], self.qvel[ids] = position, velocity
        self.ctrl[ids] = position[:, self.qadr]
        self.actuator_force[ids] = self.joint_force[ids] = 0
        self.batch.forward(np.sort(ids))

    def randomize(
        self,
        env_ids=None,
        *,
        friction=(0.6, 0.9),
        mass_scale=(0.8, 1.25),
        body_ids=None,
    ):
        """Uniform sliding friction, log-uniform mass/inertia scaling from nominal.

        Derived constants are recomputed per selected simulation. This operation
        is intended at reset, not while a solver integration step is in flight.
        """
        ids = selected_ids(env_ids, self.num_envs)
        bodies = selected_ids(body_ids, self.model.nbody)
        for limits in (friction, mass_scale):
            limits = finite_array(limits, (2,), "Randomization range")
            if not 0 < limits[0] <= limits[1]:
                raise ValueError("Randomization ranges must be positive and ordered")
        if not len(ids):
            return
        mu = self.rng.uniform(*friction, (len(ids), 1))
        scale = np.exp(self.rng.uniform(*np.log(mass_scale), (len(ids), 1)))
        self.batch.expand("geom_friction")[ids, :, 0] = mu
        self.batch.expand("body_mass")[np.ix_(ids, bodies)] = (
            self.model.body_mass[bodies] * scale
        )
        self.batch.expand("body_inertia")[np.ix_(ids, bodies)] = (
            self.model.body_inertia[bodies] * scale[..., None]
        )
        self.batch.set_const(np.sort(ids))
        self.batch.forward(np.sort(ids))

    def step(self, substeps=1):
        if type(substeps) is not int or substeps <= 0:
            raise ValueError("Substeps must be a positive integer")
        self.batch.step(nstep=substeps)
        # Keep forces used by the last integration substep before refreshing FK.
        self.actuator_force[:] = self._actuator_force
        self.joint_force[:] = self._joint_force[:, self.vadr]
        self.batch.forward()
        if not np.isfinite(self.qpos).all() or not np.isfinite(self.qvel).all():
            raise RuntimeError("Non-finite articulation state")

    def snapshot(self):
        return {
            "joint_names": self.names,
            "qpos": self.qpos.copy(),
            "qvel": self.qvel.copy(),
            "joint_position": self.qpos[:, self.qadr].copy(),
            "joint_velocity": self.qvel[:, self.vadr].copy(),
            "position_target": self.ctrl.copy(),
            "actuator_force": self.actuator_force.copy(),
            "joint_torque": self.joint_force.copy(),
            "body_position": self.body_position.copy(),
            "body_quaternion_wxyz": self.body_quaternion.copy(),
        }
