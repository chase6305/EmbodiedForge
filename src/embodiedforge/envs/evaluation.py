"""Fixed-budget evaluation over the shared borrowed CPU environment contract."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from embodiedforge.core import Array, Observation

from .base import BatchEnvironment, TransitionWriter
from .observations import replace_reset_rows
from .specs import validate_environment_spec


@dataclass(frozen=True)
class EvaluationSummary:
    """Only complete episodes contribute to episode-return and success metrics."""

    transitions: int
    completed_episodes: int
    terminated_episodes: int
    truncated_episodes: int
    mean_step_reward: float
    mean_episode_return: float | None
    success_rate: float | None


def evaluate_policy(
    env: BatchEnvironment,
    action_fn: Callable[[Observation], Array],
    *,
    steps: int,
    writer: TransitionWriter | None = None,
    on_reset: Callable[[Array], None] | None = None,
) -> EvaluationSummary:
    """Start fresh episodes, record final observations, and explicitly reset done.

    action_fn receives all rows and returns one [N,A] action batch. The caller
    owns policy state and may clear history through on_reset. Resources and
    reseeding stay with the caller. A sink must copy retained arrays before
    returning; it receives post-action transitions before any reset occurs.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError("Evaluation steps must be a positive integer")
    spec = env.spec
    validate_environment_spec(spec)
    environment_count = spec["num_envs"]
    observation = env.reset()
    if on_reset is not None:
        on_reset(np.arange(environment_count, dtype=np.int64))
    episode_returns = np.zeros(environment_count, dtype=np.float64)
    completed_episodes = 0
    terminated_episodes = 0
    truncated_episodes = 0
    successful_episodes = 0
    episodes_with_success_metric = 0
    total_step_reward = 0.0
    total_completed_episode_return = 0.0
    for step_index in range(steps):
        result = env.step(action_fn(observation))
        if (
            not isinstance(result.reward, np.ndarray)
            or result.reward.shape != (environment_count,)
            or result.reward.dtype.kind not in "fiu"
            or not np.isfinite(result.reward).all()
        ):
            raise ValueError("Evaluation reward must be finite numeric[N]")
        if any(
            not isinstance(value, np.ndarray)
            or value.shape != (environment_count,)
            or value.dtype != bool
            for value in (result.terminated, result.truncated)
        ):
            raise ValueError("Evaluation done flags must be boolean[N]")
        if np.any(result.terminated & result.truncated):
            raise ValueError("Evaluation transition cannot be terminated and truncated")
        success_mask = result.info.get("success")
        if "success" in result.info and (
            not isinstance(success_mask, np.ndarray)
            or success_mask.shape != (environment_count,)
            or success_mask.dtype != bool
        ):
            raise ValueError("Evaluation success flags must be boolean[N]")
        if writer is not None:
            writer.append(observation, result)
        completed_mask = result.terminated | result.truncated
        episode_returns += result.reward
        total_step_reward += float(np.sum(result.reward, dtype=np.float64))
        completed_this_step = int(completed_mask.sum())
        completed_episodes += completed_this_step
        terminated_episodes += int(result.terminated.sum())
        truncated_episodes += int((result.truncated & ~result.terminated).sum())
        total_completed_episode_return += float(episode_returns[completed_mask].sum())
        # Success is optional: only completed episodes with a reported final
        # success flag enter its denominator. Unfinished returns stay per row.
        if success_mask is not None:
            episodes_with_success_metric += completed_this_step
            successful_episodes += int((success_mask & completed_mask).sum())
        observation = result.observation
        if completed_mask.any() and step_index + 1 < steps:
            reset_ids = np.flatnonzero(completed_mask)
            reset_observation = env.reset(reset_ids)
            # Reset returns selected rows in reset_ids order. Validate their
            # schema before replacing rows in the next policy observation.
            if reset_observation.keys() != observation.keys():
                raise ValueError("Reset observation fields differ from step fields")
            for name, terminal_values in observation.items():
                reset_values = reset_observation[name]
                if (
                    reset_values.shape != (len(reset_ids), *terminal_values.shape[1:])
                    or reset_values.dtype != terminal_values.dtype
                ):
                    raise ValueError(
                        "Reset observation shape/dtype differs from step fields"
                    )
            # Replacement preserves terminal arrays already delivered to writer.
            observation = replace_reset_rows(observation, reset_observation, reset_ids)
            episode_returns[reset_ids] = 0
            if on_reset is not None:
                on_reset(reset_ids.copy())
    return EvaluationSummary(
        transitions=steps * environment_count,
        completed_episodes=completed_episodes,
        terminated_episodes=terminated_episodes,
        truncated_episodes=truncated_episodes,
        mean_step_reward=total_step_reward / (steps * environment_count),
        mean_episode_return=(
            total_completed_episode_return / completed_episodes
            if completed_episodes
            else None
        ),
        success_rate=(
            successful_episodes / episodes_with_success_metric
            if episodes_with_success_metric
            else None
        ),
    )
