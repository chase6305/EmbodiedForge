"""Task state, reset distributions, observations and rewards, independent of engines."""

from collections.abc import Callable

import numpy as np

from .core import (
    Array,
    Observation,
    SceneSpec,
    SceneUpdate,
    StateSnapshot,
    Task,
    TaskSpec,
)


class ReachTask:
    """Planar force-controlled reach task; each environment owns one target XY."""

    scene = SceneSpec()
    spec = TaskSpec(
        "reach",
        "Move the blue agent to the green target and stop.",
        action_dim=2,
        proprio_dim=6,
        action_units="newtons_xy",
    )

    def build(self, num_envs: int) -> None:
        self.target = np.zeros((num_envs, 2))

    def initial(self, rng: np.random.Generator) -> tuple[Array, Array]:
        """Sample agent and target positions using this episode's private RNG."""
        return rng.uniform(-0.8, 0.8, 2), rng.uniform(-0.8, 0.8, 2)

    def reset(self, ids: Array, rngs: list[np.random.Generator]) -> Array:
        """Modify only selected targets and return matching initial positions."""
        position = np.empty((len(ids), 2))
        for k, (i, rng) in enumerate(zip(ids, rngs, strict=True)):
            position[k], self.target[i] = self.initial(rng)
        return position

    def observe(self, state: StateSnapshot) -> Observation:
        """Features: XY position, XY velocity, relative target XY (six floats)."""
        return {
            "proprio": np.concatenate(
                (state.position, state.velocity, self.target - state.position), axis=1
            ).astype(np.float32)
        }

    def scene_update(self, state: StateSnapshot) -> SceneUpdate:
        return SceneUpdate(state, {"agent": state.position, "target": self.target})

    def evaluate(
        self, state: StateSnapshot, actions: Array, previous: StateSnapshot
    ) -> tuple[Array, Array, Array]:
        """Reward progress; terminate at a settled target or outside the workspace."""
        distance = np.linalg.norm(self.target - state.position, axis=1)
        speed = np.linalg.norm(state.velocity, axis=1)
        success = (distance < 0.08) & (speed < 0.15)
        escaped = np.linalg.norm(state.position, axis=1) > 3
        progress = np.linalg.norm(self.target - previous.position, axis=1) - distance
        reward = (
            10 * progress
            - 0.02 * distance
            - 0.001 * (actions**2).sum(axis=1)
            + 2 * success
            - escaped
        )
        return reward.astype(np.float32), success | escaped, success

    def expert_action(self, observation: Observation) -> Array:
        """Optional PD demonstrator; not required by the Task protocol."""
        proprio = observation["proprio"]
        return np.clip(6 * proprio[:, 4:6] - 4 * proprio[:, 2:4], -1, 1).astype(
            np.float32
        )


class HoldTask(ReachTask):
    """Return to a fixed origin with position/velocity only (four features)."""

    spec = TaskSpec(
        "hold",
        "Move the blue agent to the origin and stop.",
        action_dim=2,
        proprio_dim=4,
        action_units="newtons_xy",
    )

    def initial(self, rng: np.random.Generator) -> tuple[Array, Array]:
        return rng.uniform(-0.8, 0.8, 2), np.zeros(2)

    def observe(self, state: StateSnapshot) -> Observation:
        return {
            "proprio": np.concatenate((state.position, state.velocity), axis=1).astype(
                np.float32
            )
        }

    def expert_action(self, observation: Observation) -> Array:
        proprio = observation["proprio"]
        return np.clip(-6 * proprio[:, :2] - 4 * proprio[:, 2:4], -1, 1).astype(
            np.float32
        )


TASKS: dict[str, Callable[[], Task]] = {"reach": ReachTask, "hold": HoldTask}


def register_task(name: str, factory: Callable[[], Task]) -> None:
    """Register explicitly at application startup; duplicate names are errors."""
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("Task registration requires a name and callable factory")
    if name in TASKS:
        raise ValueError(f"Task already registered: {name}")
    TASKS[name] = factory


def make_task(name: str) -> Task:
    """Create a fresh task instance without initializing a simulation backend."""
    if name not in TASKS:
        raise ValueError(f"Unknown task {name!r}; available: {sorted(TASKS)}")
    return TASKS[name]()


def expert_action(observation: Observation) -> Array:
    """Compatibility helper for the original reach demonstration."""
    return ReachTask().expert_action(observation)
