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
    delta = reward + gamma * next_value * ~terminated - value
    discount = gamma * gae_lambda * ~(terminated | truncated)
    for t in reversed(range(len(reward))):
        carry = delta[t] + discount[t] * carry
        advantage[t] = carry
    return advantage, advantage + value
