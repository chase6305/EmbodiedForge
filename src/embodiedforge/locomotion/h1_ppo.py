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
    buffer = {
        k: []
        for k in (
            "obs",
            "action",
            "logp",
            "value",
            "next_value",
            "reward",
            "terminated",
            "truncated",
        )
    }
    for _ in range(horizon):
        with torch.no_grad():
            tensor = torch.as_tensor(observation)
            dist = policy.distribution(tensor)
            action = dist.sample()
            value, logp = policy.value(tensor), dist.log_prob(action).sum(-1)
        next_obs, reward, terminated, truncated, _ = env.step(action.numpy())
        with torch.no_grad():
            next_value = policy.value(torch.as_tensor(next_obs)).numpy()
        for key, data in {
            "obs": observation,
            "action": action.numpy(),
            "logp": logp.numpy(),
            "value": value.numpy(),
            "next_value": next_value,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }.items():
            buffer[key].append(data.copy())
        observation = next_obs
        ids = np.flatnonzero(terminated | truncated)
        if ids.size:
            observation[ids] = env.reset(ids)[ids]
    return {k: np.stack(v) for k, v in buffer.items()}, observation


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
            dist = policy.distribution(flat["obs"][indices])
            logp = dist.log_prob(flat["action"][indices]).sum(-1)
            ratio = torch.exp(logp - flat["logp"][indices])
            actor = -torch.minimum(
                ratio * advantage[indices], ratio.clamp(0.8, 1.2) * advantage[indices]
            ).mean()
            value = policy.value(flat["obs"][indices])
            clipped = flat["value"][indices] + (value - flat["value"][indices]).clamp(
                -0.2, 0.2
            )
            critic = torch.maximum(
                (value - returns[indices]) ** 2, (clipped - returns[indices]) ** 2
            ).mean()
            loss = actor + critic - 0.01 * dist.entropy().sum(-1).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite native H1 PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError("Non-finite native H1 policy")
    return float(np.mean(losses))
