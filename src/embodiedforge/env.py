"""Orchestrate task, physics and sensors with explicit per-environment reset."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from .backends import resolve
from .core import (
    Array,
    Config,
    Observation,
    PhysicsBackend,
    RenderBackend,
    StateSnapshot,
    StepResult,
    Task,
)
from .sensors import SensorPipeline
from .tasks import make_task


class VectorEnv:
    """Synchronous CPU environment; owns one task and both backend lifetimes.

    step returns post-action observations before reset. Completed environments
    freeze until explicitly reset. Returned observations own their arrays.
    This class coordinates execution; it does not define task geometry/rewards.
    """

    def __init__(self, config: Config | None = None, task: Task | None = None) -> None:
        self.config = config or Config()
        self.task = task if task is not None else make_task(self.config.task)
        physics, render = resolve(self.config, self.task.scene, self.task.spec)
        self.physics: PhysicsBackend | None = None
        self.renderer: RenderBackend | None = None
        self.closed = False
        self.failure: BaseException | None = None
        self.cleanup_errors: list[Exception] = []
        try:
            self.task.build(self.config.num_envs)
            self.physics = physics.factory()
            self.physics.build(self.task.scene, self.config)
            self.renderer = render.factory()
            self.renderer.build(self.task.scene, self.config)
            self.sensors = SensorPipeline(self.config, self.renderer)
            count = self.config.num_envs
            self.steps = np.zeros(count, dtype=np.int64)
            self.episodes = np.full(count, -1, dtype=np.int64)
            self.terminated = np.zeros(count, dtype=bool)
            self.truncated = np.zeros(count, dtype=bool)
            self.seed = self.config.seed
            self.reset()
        except BaseException:
            self._close_preserving_error()
            raise

    @property
    def spec(self) -> dict:
        """Serializable environment contract; dimensions and units come from TaskSpec."""
        task_spec = self.task.spec
        return {
            "task": task_spec.id,
            "scene": asdict(self.task.scene),
            "num_envs": self.config.num_envs,
            "action_shape": [task_spec.action_dim],
            "action_range": [task_spec.action_low, task_spec.action_high],
            "action_units": task_spec.action_units,
            "proprio_shape": [task_spec.proprio_dim],
            "device": "cpu",
            "control_dt": 1 / self.config.control_hz,
            "instruction": task_spec.instruction,
        }

    @property
    def calibration(self) -> dict:
        """Return an owned copy of camera metadata for run manifests."""
        return deepcopy(self.sensors.calibration)

    def _check_open(self) -> None:
        if self.failure is not None:
            raise RuntimeError(
                "Environment failed; construct a new instance"
            ) from self.failure
        if self.closed:
            raise RuntimeError("Environment is closed")

    @contextmanager
    def _operation(self) -> Iterator[None]:
        """A backend/task error poisons the runtime; preserve its original cause."""
        try:
            yield
        except BaseException as error:
            self.failure = error
            self._close_preserving_error()
            raise

    def _close_preserving_error(self) -> None:
        """Cleanup diagnostics must not mask an error already being propagated."""
        try:
            self.close()
        except Exception:
            # close retains every backend exception in cleanup_errors.
            pass

    def _ids(self, env_ids: Array | list[int] | None) -> Array:
        if env_ids is None:
            return np.arange(self.config.num_envs)
        ids = np.asarray(env_ids)
        if ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional integer sequence")
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        if ids.dtype.kind not in "iu":
            raise ValueError("env_ids must be a one-dimensional integer sequence")
        if (
            (ids < 0).any()
            or (ids >= self.config.num_envs).any()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("env_ids must be unique and in range")
        return ids.astype(np.int64)

    def reset(
        self, env_ids: Array | list[int] | None = None, *, seed: int | None = None
    ) -> Observation:
        """Reset selected IDs and return only those rows, preserving their order.

        A seed may be supplied only for full reset. Each environment/episode gets
        an independent generator, so resetting one row cannot alter other RNGs.
        """
        self._check_open()
        ids = self._ids(env_ids)
        if seed is not None:
            if env_ids is not None:
                raise ValueError("Reseeding requires a full reset")
            if type(seed) is not int or seed < 0:
                raise ValueError("seed must be a nonnegative integer")
        with self._operation():
            return self._reset(ids, seed)

    def _reset(self, ids: Array, seed: int | None) -> Observation:
        """Execute a validated reset inside the failure boundary."""
        if seed is not None:
            self.seed = seed
            self.episodes[:] = -1
        self.episodes[ids] += 1
        rngs = [
            np.random.default_rng(
                np.random.SeedSequence([self.seed, int(i), int(self.episodes[i])])
            )
            for i in ids
        ]
        position = self.task.reset(ids, rngs)
        self.physics.reset(ids, position)
        self.steps[ids] = 0
        self.terminated[ids] = self.truncated[ids] = False
        state = self.physics.snapshot()
        self._sample(ids, state)
        return {key: value[ids].copy() for key, value in self._observe(state).items()}

    def _sample(self, ids: Array, state: StateSnapshot) -> None:
        if len(ids) and self.config.channels:
            self.sensors.sample(ids, self.task.scene_update(state))

    def _observe(self, state: StateSnapshot) -> Observation:
        metadata = {
            "env_id": np.arange(self.config.num_envs),
            "episode_id": self.episodes.copy(),
            "step_id": self.steps.copy(),
            "time": state.time.copy(),
            "state_version": state.version.copy(),
        }
        sensors = self.sensors.observe(state.time)
        features = self.task.observe(state)
        reserved = set(metadata) | set(sensors)
        if features.keys() & reserved:
            raise ValueError(
                "Task observation must not overwrite runtime/sensor fields"
            )
        proprio = features.get("proprio")
        if (
            proprio is None
            or not isinstance(proprio, np.ndarray)
            or proprio.shape != (self.config.num_envs, self.task.spec.proprio_dim)
            or proprio.dtype != np.float32
        ):
            raise ValueError("Task proprio does not match TaskSpec shape/float32 dtype")
        if not np.isfinite(proprio).all():
            raise ValueError("Task proprio must be finite")
        for name, value in features.items():
            if (
                not isinstance(value, np.ndarray)
                or value.ndim < 1
                or len(value) != self.config.num_envs
            ):
                raise ValueError(
                    f"Task observation {name} must have an environment batch axis"
                )
        # A task may return borrowed feature arrays; the public result never does.
        return {
            **metadata,
            **sensors,
            **{name: value.copy() for name, value in features.items()},
        }

    def observe(self) -> Observation:
        """Read current state and cached frames without advancing or resampling."""
        self._check_open()
        with self._operation():
            return self._observe(self.physics.snapshot())

    def step(self, actions: Array) -> StepResult:
        """Advance one control period; invalid input is rejected before mutation."""
        self._check_open()
        actions = np.asarray(actions, dtype=np.float64)
        task_spec = self.task.spec
        expected = (self.config.num_envs, task_spec.action_dim)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"actions must be finite with shape {expected}")
        if (
            (actions < task_spec.action_low - 1e-7)
            | (actions > task_spec.action_high + 1e-7)
        ).any():
            raise ValueError(
                f"actions must lie in [{task_spec.action_low}, {task_spec.action_high}]"
            )
        actions = np.clip(actions, task_spec.action_low, task_spec.action_high)
        with self._operation():
            return self._step(actions)

    def _step(self, actions: Array) -> StepResult:
        """Execute a validated action inside the failure boundary."""
        active = ~(self.terminated | self.truncated)
        previous = self.physics.snapshot()
        self.physics.apply_control(np.where(active[:, None], actions, 0))
        self.physics.step(
            1 / self.config.control_hz,
            self.config.physics_hz // self.config.control_hz,
            active,
        )
        self.steps[active] += 1
        state = self.physics.snapshot()
        reward, terminated, success = self.task.evaluate(state, actions, previous)
        for name, value, dtype in (
            ("reward", reward, np.float32),
            ("terminated", terminated, bool),
            ("success", success, bool),
        ):
            if value.shape != (self.config.num_envs,) or value.dtype != dtype:
                raise ValueError(f"Task {name} must have shape [N] and dtype {dtype}")
        if not np.isfinite(reward).all():
            raise ValueError("Task reward must be finite")
        self.terminated[active] = terminated[active]
        self.truncated[active] = (
            (self.steps >= self.config.max_steps) & ~self.terminated
        )[active]
        due = self.sensors.due(self.steps, active, self.terminated | self.truncated)
        self._sample(due, state)
        return StepResult(
            self._observe(state),
            np.where(active, reward, 0),
            self.terminated.copy(),
            self.truncated.copy(),
            {
                "active": active,
                "success": success & active,
                "executed_action": np.where(active[:, None], actions, 0).astype(
                    np.float32
                ),
            },
        )

    def close(self) -> None:
        """Release render then physics resources, once; safe after partial build."""
        if self.closed:
            return
        self.closed = True
        for backend in (self.renderer, self.physics):
            if backend is not None:
                try:
                    backend.close()
                except Exception as error:
                    self.cleanup_errors.append(error)
        if self.cleanup_errors:
            raise RuntimeError(
                "Backend cleanup failed: "
                + "; ".join(str(e) for e in self.cleanup_errors)
            ) from self.cleanup_errors[0]

    def __enter__(self) -> "VectorEnv":
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is None:
            self.close()
        else:
            self._close_preserving_error()
