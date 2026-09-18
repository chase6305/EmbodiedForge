"""Policy evaluation and demonstration collection without training dependencies."""

import hashlib
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


@dataclass(frozen=True)
class EpisodeBatch:
    """One complete episode per environment, with no replacement episodes."""

    success: Array
    returns: Array
    lengths: Array
    initial_observation_sha256: str


def run_episode_batch(
    env: VectorEnv,
    policy: Policy,
    *,
    chunk_horizon: int = 1,
    on_reset: Callable[[Array], None] | None = None,
) -> EpisodeBatch:
    """Reseed to env.seed, finish exactly one episode per row, and freeze done rows.

    Finished rows never reach policy inference again. The policy receives original
    env_id values for active rows; on_reset clears caller-owned policy state.
    """
    spec = env.task.spec
    executor = ActionExecutor(
        env.config.num_envs,
        spec.action_dim,
        chunk_horizon,
        action_low=spec.action_low,
        action_high=spec.action_high,
    )
    observation = env.reset(seed=env.seed)
    if on_reset is not None:
        on_reset(np.arange(env.config.num_envs, dtype=np.int64))
    digest = hashlib.sha256()
    for name, value in sorted(observation.items()):
        digest.update(f"{name}:{value.dtype}:{value.shape}:".encode())
        digest.update(value.tobytes())
    done = np.zeros(env.config.num_envs, dtype=bool)
    success = np.zeros_like(done)
    returns = np.zeros(env.config.num_envs, dtype=np.float64)
    lengths = np.zeros(env.config.num_envs, dtype=np.int64)

    class ActivePolicy:
        def act(self, batch):
            active = ~done[batch["env_id"]]
            if active.any():
                predicted = np.asarray(
                    policy.act({k: v[active] for k, v in batch.items()})
                )
                if (
                    predicted.ndim != 3
                    or predicted.shape[0] != int(active.sum())
                    or predicted.shape[2] != spec.action_dim
                ):
                    raise ValueError("policy must return [active_rows,H,action_dim]")
                horizon = predicted.shape[1]
            else:
                horizon = 1
            chunks = np.full(
                (len(active), horizon, spec.action_dim),
                np.clip(0.0, spec.action_low, spec.action_high),
                dtype=np.float32,
            )
            if active.any():
                chunks[active] = predicted
            return chunks

    active_policy = ActivePolicy()
    for _ in range(env.config.max_steps):
        active = ~done
        result = env.step(executor.act(observation, active_policy))
        returns[active] += result.reward[active]
        lengths[active] += 1
        newly_done = active & (result.terminated | result.truncated)
        success[newly_done] = result.info["success"][newly_done]
        done |= newly_done
        observation = result.observation
        if done.all():
            break
    if not done.all():
        raise RuntimeError("Environment did not finish every episode within max_steps")
    return EpisodeBatch(success, returns, lengths, digest.hexdigest())


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
