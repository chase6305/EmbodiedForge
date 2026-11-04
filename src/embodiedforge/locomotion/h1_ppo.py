"""Project-owned H1 PPO; no upstream trainer or IsaacLab runtime imports."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from embodiedforge.learning.advantages import generalized_advantage
from embodiedforge.learning.parameters import bounded_log_std, mlp

from .h1_config import ACT_DIM, OBS_DIM


class Policy(nn.Module):
    """Gaussian actor and value network for the fixed native H1 observation layout."""

    def __init__(self):
        super().__init__()

        self.actor = mlp((OBS_DIM, 128, 128, 128, ACT_DIM), nn.ELU)
        self.critic = mlp((OBS_DIM, 128, 128, 128, 1), nn.ELU)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, observation):
        return Normal(self.actor(observation), bounded_log_std(self.log_std).exp())

    def value(self, observation):
        return self.critic(observation).squeeze(-1)


def collect(policy, env, observation, horizon):
    """Store pre-reset transitions and reuse bootstrap values for unchanged rows."""
    if horizon <= 0:
        raise ValueError("Rollout horizon must be positive")
    rollout_buffer = {}
    current_value = None  # Reuse only within a rollout, while the policy is unchanged.
    value_reset_ids = None
    for step_index in range(horizon):
        with torch.no_grad():
            observation_tensor = torch.as_tensor(observation)
            distribution = policy.distribution(observation_tensor)
            actions = distribution.sample()
            if current_value is None:
                current_value = policy.value(observation_tensor)
            elif value_reset_ids is not None:
                # Continuing observations still match the previous bootstrap;
                # recompute only rows changed by the preceding reset.
                current_value = current_value.index_copy(
                    0,
                    torch.as_tensor(value_reset_ids),
                    policy.value(observation_tensor[value_reset_ids]),
                )
            log_prob = distribution.log_prob(actions).sum(-1)
        next_observation, reward, terminated, truncated, _ = env.step(actions.numpy())
        with torch.no_grad():
            next_value = policy.value(torch.as_tensor(next_observation))
        transition = {
            "obs": observation,
            "action": actions.numpy(),
            "logp": log_prob.numpy(),
            "value": current_value.numpy(),
            "next_value": next_value.numpy(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        if step_index == 0:
            rollout_buffer = {
                field_name: np.empty(
                    (horizon, *field_values.shape), dtype=field_values.dtype
                )
                for field_name, field_values in transition.items()
            }
        for field_name, field_values in transition.items():
            rollout_buffer[field_name][step_index] = field_values
        observation = next_observation
        current_value = next_value
        value_reset_ids = None
        done_ids = np.flatnonzero(terminated | truncated)
        if done_ids.size:
            if done_ids.size == len(current_value):
                current_value = None
            else:
                value_reset_ids = done_ids
            # Native H1 reset returns the full batch; replace only finished rows.
            observation[done_ids] = env.reset(done_ids)[done_ids]
    return rollout_buffer, observation


def optimize(policy, optimizer, rollout, rng, *, epochs=5):
    """Apply clipped policy/value updates using stored pre-reset bootstrap values."""
    advantages, returns = generalized_advantage(
        rollout["reward"],
        rollout["value"],
        rollout["next_value"],
        rollout["terminated"],
        rollout["truncated"],
        0.99,
        0.95,
    )
    sample_count = advantages.size
    flat_rollout = {
        field_name: torch.as_tensor(
            rollout[field_name].reshape(sample_count, *rollout[field_name].shape[2:])
        )
        for field_name in ("obs", "action", "logp", "value")
    }
    advantages = torch.as_tensor(advantages.ravel())
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    returns = torch.as_tensor(returns.ravel())
    losses = []
    for _ in range(epochs):
        for minibatch_ids in np.array_split(
            rng.permutation(sample_count), min(4, sample_count)
        ):
            observations = flat_rollout["obs"][minibatch_ids]
            minibatch_advantages = advantages[minibatch_ids]
            old_values = flat_rollout["value"][minibatch_ids]
            minibatch_returns = returns[minibatch_ids]
            distribution = policy.distribution(observations)
            log_prob = distribution.log_prob(flat_rollout["action"][minibatch_ids]).sum(
                -1
            )
            probability_ratio = torch.exp(
                log_prob - flat_rollout["logp"][minibatch_ids]
            )
            actor_loss = -torch.minimum(
                probability_ratio * minibatch_advantages,
                probability_ratio.clamp(0.8, 1.2) * minibatch_advantages,
            ).mean()
            predicted_values = policy.value(observations)
            clipped_values = old_values + (predicted_values - old_values).clamp(
                -0.2, 0.2
            )
            critic_loss = torch.maximum(
                (predicted_values - minibatch_returns) ** 2,
                (clipped_values - minibatch_returns) ** 2,
            ).mean()
            loss = (
                actor_loss + critic_loss - 0.01 * distribution.entropy().sum(-1).mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite native H1 PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            # Project the parameter too: a forward-only clamp gives zero
            # gradient after Adam overshoots a boundary, freezing exploration.
            with torch.no_grad():
                if not torch.isfinite(policy.log_std).all():
                    raise FloatingPointError("Non-finite native H1 policy log_std")
                policy.log_std.clamp_(-5, 2)
            losses.append(loss.item())
    if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise FloatingPointError("Non-finite native H1 policy")
    return float(np.mean(losses))
