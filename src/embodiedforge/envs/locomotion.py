"""Borrow Go1/H1 environments without changing their native training APIs."""

from copy import deepcopy

import numpy as np

from embodiedforge.core import Observation, StepResult

from .specs import ENVIRONMENT_SPEC_VERSION


class _NativeEnvironment:
    """Cache native observations and enforce reset before advancing a done batch."""

    def __init__(self, native, *, task, action_dim, proprio_dim, dt, bounds):
        self.native = native
        self.failure: BaseException | None = None
        steps = np.asarray(native.steps)
        if steps.ndim != 1 or not len(steps) or np.any(steps != 0):
            raise ValueError("Attach a native environment adapter after a full reset")
        self._spec = {
            "schema_version": ENVIRONMENT_SPEC_VERSION,
            "task": task,
            "num_envs": len(steps),
            "action_shape": [action_dim],
            "proprio_shape": [proprio_dim],
            "control_dt": dt,
            "action_units": "scaled_joint_position_offset",
            "action_range": bounds,
            "device": "cpu",
            "done_handling": "reset_before_next_step",
        }
        self._steps = np.zeros(len(steps), dtype=np.int64)
        self._episodes = np.zeros(len(steps), dtype=np.int64)
        self._versions = np.zeros(len(steps), dtype=np.int64)
        self._done = np.zeros(len(steps), dtype=bool)
        self._proprio = self._features(native.obs()).copy()

    @property
    def spec(self) -> dict:
        return deepcopy(self._spec)

    def _check(self):
        if self.failure is not None:
            raise RuntimeError(
                "Native adapter failed; reconstruct its environment"
            ) from self.failure

    def _features(self, value):
        """Validate native features; cache owners copy the retained array."""
        value = np.asarray(value)
        expected = (self._spec["num_envs"], self._spec["proprio_shape"][0])
        if (
            value.shape != expected
            or value.dtype != np.float32
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "Native observation must be finite float32 matching its spec"
            )
        return value

    def _ids(self, env_ids):
        if env_ids is None:
            return np.arange(self._spec["num_envs"], dtype=np.int64)
        ids = np.asarray(env_ids)
        if ids.ndim != 1 or (
            ids.size
            and (
                ids.dtype.kind not in "iu"
                or np.any(ids < 0)
                or np.any(ids >= self._spec["num_envs"])
                or len(np.unique(ids)) != len(ids)
            )
        ):
            raise ValueError("env_ids must be unique one-dimensional integers in range")
        return ids.astype(np.int64)

    def _observe(self, ids) -> Observation:
        return {
            "env_id": ids.copy(),
            "episode_id": self._episodes[ids],
            "step_id": self._steps[ids],
            "state_version": self._versions[ids],
            "time": self._steps[ids] * self._spec["control_dt"],
            "proprio": self._proprio[ids],
        }

    def observe(self) -> Observation:
        self._check()
        return self._observe(np.arange(self._spec["num_envs"]))

    def reset(self, env_ids=None) -> Observation:
        self._check()
        ids = self._ids(env_ids)
        if not len(ids):
            return self._observe(ids)
        try:
            value = self.native.reset(ids)
            # H1 reset already sampled its observation (including training
            # noise); Go1 reset returns None. Never sample H1 a second time.
            if value is None:
                value = self.native.obs()
            value = self._features(value)
            self._proprio[ids] = value[ids]
            self._steps[ids] = 0
            self._episodes[ids] += 1
            self._versions[ids] += 1
            self._done[ids] = False
            return self._observe(ids)
        except BaseException as error:
            self.failure = error
            raise

    def _transition(self, actions):
        raise NotImplementedError

    def step(self, actions) -> StepResult:
        self._check()
        if self._done.any():
            raise RuntimeError("Reset all completed native rows before the next step")
        actions = np.asarray(actions)
        expected = (self._spec["num_envs"], self._spec["action_shape"][0])
        if (
            actions.shape != expected
            or actions.dtype.kind not in "fiu"
            or not np.isfinite(actions).all()
        ):
            raise ValueError(
                f"Actions must be finite numeric values with shape {expected}"
            )
        bounds = self._spec["action_range"]
        if bounds is None:
            limit = np.finfo(np.float32).max
            if np.any(actions < -limit) or np.any(actions > limit):
                raise ValueError(
                    "Native policy actions must be representable as float32"
                )
            executed = actions.copy()
        else:
            executed = np.clip(actions, *bounds)
        try:
            observation, reward, terminated, truncated, terms = self._transition(
                executed
            )
            self._proprio = self._features(observation).copy()
            count = self._spec["num_envs"]
            reward = np.asarray(reward)
            terminated, truncated = np.asarray(terminated), np.asarray(truncated)
            if (
                reward.shape != (count,)
                or reward.dtype.kind not in "fiu"
                or not np.isfinite(reward).all()
            ):
                raise ValueError("Native reward must be finite with shape [N]")
            limit = np.finfo(np.float32).max
            if np.any(reward < -limit) or np.any(reward > limit):
                raise ValueError("Native reward must be representable as float32")
            if any(
                value.shape != (count,) or value.dtype != bool
                for value in (terminated, truncated)
            ):
                raise ValueError("Native done flags must be boolean with shape [N]")
            if np.any(terminated & truncated):
                raise ValueError("Native transition cannot be terminated and truncated")
            info = {
                "active": np.ones(count, dtype=bool),
                "executed_action": executed.astype(np.float32),
            }
            for name, value in terms.items():
                value = np.asarray(value)
                if (
                    value.shape != (count,)
                    or value.dtype.kind not in "biuf"
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        "Native reward terms must be finite with shape [N]"
                    )
                info[f"reward/{name}"] = value.copy()
            self._steps += 1
            self._versions += 1
            self._done = terminated | truncated
            return StepResult(
                self.observe(),
                reward.astype(np.float32),
                terminated.copy(),
                truncated.copy(),
                info,
            )
        except BaseException as error:
            self.failure = error
            raise


class Go1Environment(_NativeEnvironment):
    """Normalize Go1's (reward, done, fell, terms) without bounding its policy."""

    def __init__(self, native):
        from embodiedforge.locomotion.go1_config import ACT_DIM, CTRL_DT, OBS_DIM

        super().__init__(
            native,
            task="go1-joystick",
            action_dim=ACT_DIM,
            proprio_dim=OBS_DIM,
            dt=CTRL_DT,
            bounds=None,
        )

    def _transition(self, actions):
        reward, done, fell, terms = self.native.step(actions)
        return self.native.obs(), reward, fell, done & ~fell, terms


class H1Environment(_NativeEnvironment):
    """Normalize native H1's full-batch reset and clipped joint policy actions."""

    def __init__(self, native):
        from embodiedforge.locomotion.h1_config import (
            ACT_DIM,
            ACTION_LIMIT,
            CTRL_DT,
            OBS_DIM,
            VERSION,
        )

        super().__init__(
            native,
            task=VERSION,
            action_dim=ACT_DIM,
            proprio_dim=OBS_DIM,
            dt=CTRL_DT,
            bounds=[-ACTION_LIMIT, ACTION_LIMIT],
        )

    def _transition(self, actions):
        return self.native.step(actions)
