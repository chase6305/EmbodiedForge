"""Project-owned H1 PPO; no upstream trainer or IsaacLab runtime imports."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from embodiedforge.training import generalized_advantage

from .h1_native import ACT_DIM, OBS_DIM


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        def network(output):
            return nn.Sequential(
                nn.Linear(OBS_DIM, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, output),
            )

        self.actor, self.critic = network(ACT_DIM), network(1)
        self.log_std = nn.Parameter(torch.zeros(ACT_DIM))
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, observation):
        return Normal(self.actor(observation), self.log_std.clamp(-5, 2).exp())

    def value(self, observation):
        return self.critic(observation).squeeze(-1)


def collect(policy, env, observation, horizon):
    if horizon <= 0:
        raise ValueError("Rollout horizon must be positive")
    buffer = {}
    value = None  # Reuse only within a rollout, while the policy is unchanged.
    reset_ids = None
    for step in range(horizon):
        with torch.no_grad():
            tensor = torch.as_tensor(observation)
            dist = policy.distribution(tensor)
            action = dist.sample()
            if value is None:
                value = policy.value(tensor)
            elif reset_ids is not None:
                value = value.index_copy(
                    0, torch.as_tensor(reset_ids), policy.value(tensor[reset_ids])
                )
            logp = dist.log_prob(action).sum(-1)
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        with torch.no_grad():
            next_value = policy.value(torch.as_tensor(next_obs))
        transition = {
            "obs": observation,
            "action": action.numpy(),
            "logp": logp.numpy(),
            "value": value.numpy(),
            "next_value": next_value.numpy(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }
        if step == 0:
            buffer = {
                key: np.empty((horizon, *data.shape), dtype=data.dtype)
                for key, data in transition.items()
            }
        for key, data in transition.items():
            buffer[key][step] = data
        observation = next_obs
        value = next_value
        reset_ids = None
        ids = np.flatnonzero(terminated | truncated)
        if ids.size:
            if ids.size == len(value):
                value = None
            else:
                reset_ids = ids
            observation[ids] = env.reset(ids)[ids]
    return buffer, observation


def optimize(policy, optimizer, rollout, rng, *, epochs=5):
    advantage, returns = generalized_advantage(
        rollout["reward"],
        rollout["value"],
        rollout["next_value"],
        rollout["terminated"],
        rollout["truncated"],
        0.99,
        0.95,
    )
    count = advantage.size
    flat = {
        k: torch.as_tensor(v.reshape(count, *v.shape[2:])) for k, v in rollout.items()
    }
    advantage = torch.as_tensor(advantage.ravel())
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    returns = torch.as_tensor(returns.ravel())
    losses = []
    for _ in range(epochs):
        for indices in np.array_split(rng.permutation(count), min(4, count)):
            observations = flat["obs"][indices]
            minibatch_advantages = advantage[indices]
            old_value = flat["value"][indices]
            targets = returns[indices]
            dist = policy.distribution(observations)
            logp = dist.log_prob(flat["action"][indices]).sum(-1)
            ratio = torch.exp(logp - flat["logp"][indices])
            actor = -torch.minimum(
                ratio * minibatch_advantages,
                ratio.clamp(0.8, 1.2) * minibatch_advantages,
            ).mean()
            value = policy.value(observations)
            clipped = old_value + (value - old_value).clamp(-0.2, 0.2)
            critic = torch.maximum(
                (value - targets) ** 2, (clipped - targets) ** 2
            ).mean()
            loss = actor + critic - 0.01 * dist.entropy().sum(-1).mean()
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
            losses.append(float(loss.detach()))
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError("Non-finite native H1 policy")
    return float(np.mean(losses))
