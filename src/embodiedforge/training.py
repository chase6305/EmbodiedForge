"""Small reference PPO trainer; imported only when training is requested."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .core import Array, Config, Observation
from .env import VectorEnv


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for the synchronous, proprio-only reference trainer."""

    updates: int = 30
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    seed: int = 0

    def __post_init__(self):
        if min(self.updates, self.rollout_steps, self.epochs, self.minibatch_size) <= 0:
            raise ValueError("PPO counts must be positive")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Invalid discount or GAE lambda")
        if not self.learning_rate > 0 or not 0 < self.clip < 1:
            raise ValueError("Invalid learning rate or clipping range")


class ActorCritic(nn.Module):
    """Gaussian actor and value network with task-defined dimensions/action units."""

    def __init__(
        self,
        proprio_dim: int = 6,
        action_dim: int = 2,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_spec = {
            "proprio_dim": proprio_dim,
            "action_dim": action_dim,
            "action_low": action_low,
            "action_high": action_high,
        }
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2
        self.actor = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def distribution(self, obs: torch.Tensor) -> Normal:
        return Normal(self.actor(obs), self.log_std.clamp(-5, 2).exp())

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, observation: Observation) -> Array:
        """Deterministic evaluation action, returned as a one-step chunk."""
        obs = torch.as_tensor(observation["proprio"], dtype=torch.float32)
        return self.to_action(self.actor(obs)).numpy()[:, None, :]

    def to_action(self, raw: torch.Tensor) -> torch.Tensor:
        """Map unconstrained Gaussian samples to the task's physical action range."""
        return raw.tanh() * self.action_scale + self.action_bias


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
    advantage = np.zeros_like(reward)
    carry = np.zeros(reward.shape[1], dtype=np.float32)
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * next_value[t] * ~terminated[t] - value[t]
        carry = delta + gamma * gae_lambda * ~(terminated[t] | truncated[t]) * carry
        advantage[t] = carry
    return advantage, advantage + value


def _collect_rollout(
    env: VectorEnv, model: ActorCritic, observation: Observation, rollout_steps: int
) -> tuple[dict[str, Array], Observation, int, int]:
    """Collect pre-reset transitions and separately refresh reset rows for acting."""
    buffer = {
        k: []
        for k in (
            "obs",
            "raw",
            "logprob",
            "reward",
            "value",
            "next_value",
            "terminated",
            "truncated",
        )
    }
    successes = completed = 0
    for _ in range(rollout_steps):
        obs = torch.as_tensor(observation["proprio"])
        with torch.no_grad():
            distribution = model.distribution(obs)
            raw = distribution.sample()
            value = model.value(obs)
            logprob = distribution.log_prob(raw).sum(-1)
        result = env.step(model.to_action(raw).numpy())
        with torch.no_grad():
            next_value = model.value(torch.as_tensor(result.observation["proprio"]))
        for key, item in {
            "obs": obs.numpy(),
            "raw": raw.numpy(),
            "logprob": logprob.numpy(),
            "reward": result.reward,
            "value": value.numpy(),
            "next_value": next_value.numpy(),
            "terminated": result.terminated,
            "truncated": result.truncated,
        }.items():
            buffer[key].append(item)
        done = result.terminated | result.truncated
        completed += int(done.sum())
        successes += int(result.info["success"].sum())
        observation = result.observation
        if done.any():
            ids = np.flatnonzero(done)
            reset_observation = env.reset(ids)
            for key in observation:
                observation[key][ids] = reset_observation[key]
    rollout = {k: np.stack(v) for k, v in buffer.items()}
    return rollout, observation, successes, completed


def _optimize_policy(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, Array],
    config: PPOConfig,
    rng: np.random.Generator,
) -> float:
    """Compute GAE and apply clipped PPO updates; return mean optimization loss."""
    advantage, returns = generalized_advantage(
        rollout["reward"],
        rollout["value"],
        rollout["next_value"],
        rollout["terminated"],
        rollout["truncated"],
        config.gamma,
        config.gae_lambda,
    )
    count = advantage.size
    flat = {
        k: torch.as_tensor(v.reshape(count, *v.shape[2:])) for k, v in rollout.items()
    }
    advantages = torch.as_tensor(advantage.reshape(-1))
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    targets = torch.as_tensor(returns.reshape(-1))
    losses = []
    for _ in range(config.epochs):
        order = rng.permutation(count)
        for start in range(0, count, config.minibatch_size):
            ids = order[start : start + config.minibatch_size]
            distribution = model.distribution(flat["obs"][ids])
            # The tanh Jacobian cancels in the old/new policy ratio.
            logprob = distribution.log_prob(flat["raw"][ids]).sum(-1)
            ratio = (logprob - flat["logprob"][ids]).exp()
            actor_loss = -torch.minimum(
                ratio * advantages[ids],
                ratio.clamp(1 - config.clip, 1 + config.clip) * advantages[ids],
            ).mean()
            value_loss = (model.value(flat["obs"][ids]) - targets[ids]).square().mean()
            loss = actor_loss + 0.5 * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            losses.append(float(loss.detach()))
    return float(np.mean(losses))


def train_ppo(
    env_config: Config, config: PPOConfig, directory: str | Path
) -> ActorCritic:
    """Train from registered task specs and save a self-describing checkpoint."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    (root / "config.json").write_text(
        json.dumps({"env": env_config.to_dict(), "ppo": asdict(config)}, indent=2)
        + "\n"
    )
    with VectorEnv(env_config) as env, (root / "metrics.jsonl").open("w") as metrics:
        spec = env.spec
        model = ActorCritic(
            proprio_dim=spec["proprio_shape"][0],
            action_dim=spec["action_shape"][0],
            action_low=spec["action_range"][0],
            action_high=spec["action_range"][1],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        observation = env.observe()
        for update in range(config.updates):
            rollout, observation, successes, completed = _collect_rollout(
                env, model, observation, config.rollout_steps
            )
            count = config.rollout_steps * env_config.num_envs
            loss = _optimize_policy(model, optimizer, rollout, config, rng)
            metric = {
                "update": update + 1,
                "transitions": (update + 1) * count,
                "mean_step_reward": float(rollout["reward"].mean()),
                "loss": loss,
                "completed_episodes": completed,
                "success_rate": successes / completed if completed else None,
            }
            metrics.write(json.dumps(metric) + "\n")
            metrics.flush()
            print(json.dumps(metric), flush=True)
        temporary = root / "checkpoint.pt.tmp"
        torch.save(
            {
                "schema_version": 2,
                "model_spec": model.model_spec,
                "env_spec": env.spec,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "env": env_config.to_dict(),
                "ppo": asdict(config),
            },
            temporary,
        )
        temporary.replace(root / "checkpoint.pt")
    return model


@dataclass
class PolicyCheckpoint:
    """Evaluation bundle; configuration and dimensions travel with model weights."""

    policy: ActorCritic
    env_config: Config
    env_spec: dict

    def validate_environment(self, spec: dict) -> None:
        """Reject semantic/dimension changes; allow device count/backend changes."""
        for key in (
            "task",
            "scene",
            "control_dt",
            "action_shape",
            "action_range",
            "action_units",
            "proprio_shape",
        ):
            if key in self.env_spec and self.env_spec[key] != spec[key]:
                raise ValueError(f"Checkpoint/environment mismatch for {key}")
        if spec["proprio_shape"] != [self.policy.model_spec["proprio_dim"]] or spec[
            "action_shape"
        ] != [self.policy.model_spec["action_dim"]]:
            raise ValueError("Checkpoint/environment dimension mismatch")


def load_checkpoint(path: str | Path) -> PolicyCheckpoint:
    """Read v1/v2 checkpoints on CPU without constructing simulation resources."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] not in (1, 2):
        raise ValueError("Unsupported checkpoint")
    model = ActorCritic(**checkpoint.get("model_spec", {}))
    model.load_state_dict(checkpoint["model"])
    model.eval()
    env_config = Config(**checkpoint["env"])
    return PolicyCheckpoint(model, env_config, checkpoint.get("env_spec", {}))


def load_policy(path: str | Path) -> ActorCritic:
    """Compatibility convenience API returning just the evaluation policy."""
    return load_checkpoint(path).policy
