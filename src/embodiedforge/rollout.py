"""Policy evaluation and demonstration collection without training dependencies."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .core import Array, Observation, Policy, StepResult
from .env import VectorEnv
from .policies import ActionExecutor


class TransitionWriter(Protocol):
    """Borrow one transition synchronously; copy any arrays retained after append."""

    def append(self, observation: Observation, result: StepResult) -> None: ...


@dataclass(frozen=True)
class RolloutSummary:
    """Aggregate metrics over this call; unfinished episodes are not successes."""

    completed_episodes: int
    success_rate: float | None
    mean_step_reward: float


class DemonstrationPolicy:
    """Adapt a task's single-action demonstrator to the [B,H,A] policy contract."""

    def __init__(self, action: Callable[[Observation], Array]) -> None:
        self.action = action

    def act(self, observation: Observation) -> Array:
        """Return a one-step chunk; ActionExecutor validates dimensions and bounds."""
        return np.asarray(self.action(observation))[:, None, :]


def run_rollout(
    env: VectorEnv,
    policy: Policy,
    *,
    steps: int,
    chunk_horizon: int = 1,
    writer: TransitionWriter | None = None,
    on_reset: Callable[[Array], None] | None = None,
) -> RolloutSummary:
    """Start fresh episodes and consume bounded chunks for a fixed step budget.

    Borrows env, policy and writer; the caller owns their lifetimes. Do not pass
    a writer with pending transitions from an earlier rollout. Records terminal
    observations before reset. on_reset receives global environment IDs at the
    start and after each reset, before further inference; use it to clear policy
    history/KV caches. Policies receive only rows needing a new chunk, with their
    original env_id values. Errors propagate; no automatic retry of inference or
    storage side effects is attempted.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("rollout steps must be a positive integer")
    spec = env.task.spec
    executor = ActionExecutor(
        env.config.num_envs,
        spec.action_dim,
        chunk_horizon,
        action_low=spec.action_low,
        action_high=spec.action_high,
    )
    observation = env.reset()
    if on_reset is not None:
        on_reset(np.arange(env.config.num_envs, dtype=np.int64))
    completed = successes = 0
    total_reward = 0.0
    for index in range(steps):
        actions = executor.act(observation, policy)
        result = env.step(actions)
        if writer is not None:
            writer.append(observation, result)
        done = result.terminated | result.truncated
        completed += int(done.sum())
        successes += int((result.info["success"] & done).sum())
        total_reward += float(result.reward.sum())
        # Keep the terminal result owned by any consumer intact when patching
        # reset rows into the next policy input.
        observation = result.observation
        if done.any() and index + 1 < steps:
            ids = np.flatnonzero(done)
            reset_observation = env.reset(ids)
            executor.reset(ids)
            if on_reset is not None:
                on_reset(ids.copy())
            observation = {key: value.copy() for key, value in observation.items()}
            for key in observation:
                observation[key][ids] = reset_observation[key]
    return RolloutSummary(
        completed_episodes=completed,
        success_rate=successes / completed if completed else None,
        mean_step_reward=total_reward / (steps * env.config.num_envs),
    )
