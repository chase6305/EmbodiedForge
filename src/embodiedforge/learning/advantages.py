"""CPU advantage estimators with explicit episode-boundary semantics."""

import numpy as np

from embodiedforge.core import Array


def generalized_advantage(
    reward: Array,
    value: Array,
    next_value: Array,
    terminated: Array,
    truncated: Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[Array, Array]:
    """Bootstrap at time limits, but never propagate GAE through reset."""
    advantage = np.empty_like(reward)
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    # Only the carry depends on time; batch the independent transition terms.
    temporal_difference = reward + gamma * next_value * ~terminated - value
    continuation_discount = gamma * gae_lambda * ~(terminated | truncated)
    for step in reversed(range(len(reward))):
        carry = temporal_difference[step] + continuation_discount[step] * carry
        advantage[step] = carry
    return advantage, advantage + value
