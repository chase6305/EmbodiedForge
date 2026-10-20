"""Small reference PPO trainer; imported only when training is requested."""

import hashlib
import io
import json
import logging
import math
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from ._adam_checkpoint import validate_adam_checkpoint
from ._core_run import publish_checkpoint, read_resume, write_json
from ._microduck_process import termination_signals
from ._ppo_parameters import bounded_log_std
from ._training_runtime import record_training_runtime
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
        for name in ("updates", "rollout_steps", "epochs", "minibatch_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"PPO {name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("PPO seed must be an integer in [0, 2**64 - 1]")
        if not (0 <= self.gamma <= 1 and 0 <= self.gae_lambda <= 1):
            raise ValueError("Invalid discount or GAE lambda")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("PPO learning rate must be finite and positive")
        if not 0 < self.clip < 1:
            raise ValueError("Invalid PPO clipping range")


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
        if not (
            math.isfinite(action_low)
            and math.isfinite(action_high)
            and action_low < action_high
        ):
            raise ValueError("Policy action range must be finite and increasing")
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
        return Normal(self.actor(obs), bounded_log_std(self.log_std).exp())

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
    # Only the carry depends on time; batch the independent transition terms.
    delta = reward + gamma * next_value * ~terminated - value
    discount = gamma * gae_lambda * ~(terminated | truncated)
    for t in reversed(range(len(reward))):
        carry = delta[t] + discount[t] * carry
        advantage[t] = carry
    return advantage, advantage + value


def _collect_rollout(
    env: VectorEnv, model: ActorCritic, observation: Observation, rollout_steps: int
) -> tuple[dict[str, Array], Observation, int, int]:
    """Collect pre-reset transitions and separately refresh reset rows for acting."""
    if rollout_steps <= 0:
        raise ValueError("Rollout steps must be positive")
    buffer = {}
    successes = completed = 0
    value = None  # Local to this rollout: optimizer updates invalidate old values.
    reset_ids = None
    for step in range(rollout_steps):
        obs = torch.as_tensor(observation["proprio"])
        with torch.no_grad():
            distribution = model.distribution(obs)
            raw = distribution.sample()
            if value is None:
                value = model.value(obs)
            elif reset_ids is not None:
                # Keep continuing rows' bootstrap values; only reset rows changed.
                value = value.index_copy(
                    0, torch.as_tensor(reset_ids), model.value(obs[reset_ids])
                )
            logprob = distribution.log_prob(raw).sum(-1)
        result = env.step(model.to_action(raw).numpy())
        with torch.no_grad():
            next_value = model.value(torch.as_tensor(result.observation["proprio"]))
        transition = {
            "obs": obs.numpy(),
            "raw": raw.numpy(),
            "logprob": logprob.numpy(),
            "reward": result.reward,
            "value": value.numpy(),
            "next_value": next_value.numpy(),
            "terminated": result.terminated,
            "truncated": result.truncated,
        }
        if step == 0:
            buffer = {
                key: np.empty((rollout_steps, *item.shape), dtype=item.dtype)
                for key, item in transition.items()
            }
        for key, item in transition.items():
            buffer[key][step] = item
        done = result.terminated | result.truncated
        completed += int(done.sum())
        successes += int(result.info["success"].sum())
        observation = result.observation
        value = next_value
        reset_ids = None
        if done.any():
            ids = np.flatnonzero(done)
            if ids.size == len(value):
                value = None
            else:
                reset_ids = ids
            reset_observation = env.reset(ids)
            for key in observation:
                observation[key][ids] = reset_observation[key]
    return buffer, observation, successes, completed


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
            observations = flat["obs"][ids]
            minibatch_advantages = advantages[ids]
            distribution = model.distribution(observations)
            # The tanh Jacobian cancels in the old/new policy ratio.
            logprob = distribution.log_prob(flat["raw"][ids]).sum(-1)
            ratio = (logprob - flat["logprob"][ids]).exp()
            actor_loss = -torch.minimum(
                ratio * minibatch_advantages,
                ratio.clamp(1 - config.clip, 1 + config.clip) * minibatch_advantages,
            ).mean()
            value_loss = (model.value(observations) - targets[ids]).square().mean()
            loss = actor_loss + 0.5 * value_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
            optimizer.step()
            # Keep exploration trainable when Adam crosses the forward clamp.
            with torch.no_grad():
                if not torch.isfinite(model.log_std).all():
                    raise FloatingPointError("Non-finite PPO policy log_std")
                model.log_std.clamp_(-5, 2)
            losses.append(float(loss.detach()))
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite PPO policy")
    return float(np.mean(losses))


@termination_signals()
def train_ppo(
    env_config: Config,
    config: PPOConfig,
    directory: str | Path,
    *,
    resume: "TrainingCheckpoint | str | Path | None" = None,
    checkpoint_interval: int = 10,
) -> ActorCritic:
    """Train additional updates, periodically saving policy and Adam state.

    Resume restores policy/Adam, then starts new environments and RNG streams.
    The supplied PPOConfig defines this run's budget and hyperparameters; its
    learning rate replaces the saved Adam learning rate.
    """
    if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")
    root = Path(directory)
    restored = None
    if resume is not None:
        restored = (
            resume
            if isinstance(resume, TrainingCheckpoint)
            else load_training_checkpoint(resume)
        )
    if restored and (type(restored.updates) is not int or restored.updates <= 0):
        raise ValueError("Invalid core PPO checkpoint update count")
    start = restored.updates if restored else 0
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    metadata = None
    began = time.monotonic()
    try:
        with VectorEnv(env_config) as env:
            spec = env.spec
            if restored:
                restored.bundle.validate_environment(spec)
                model = deepcopy(restored.bundle.policy)
            else:
                model = ActorCritic(
                    proprio_dim=spec["proprio_shape"][0],
                    action_dim=spec["action_shape"][0],
                    action_low=spec["action_range"][0],
                    action_high=spec["action_range"][1],
                )
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            if restored:
                validate_adam_checkpoint(
                    optimizer, restored.optimizer_state, label="core PPO optimizer"
                )
                optimizer.load_state_dict(deepcopy(restored.optimizer_state))
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate
            root.mkdir(parents=True, exist_ok=False)
            metadata = {
                "schema_version": 1,
                "kind": "core-ppo",
                "status": "running",
                "initial_updates": start,
                "requested_updates": config.updates,
                "completed_updates": start,
                "saved_updates": 0,
                "checkpoint_interval": checkpoint_interval,
                "env": env_config.to_dict(),
                "ppo": asdict(config),
                "resume": (
                    {"source": restored.source, "sha256": restored.bundle.sha256}
                    if restored
                    else None
                ),
                "resume_semantics": "policy+optimizer; environment and RNG reset",
            }
            write_json(root / "run.json", metadata)
            write_json(
                root / "config.json",
                {"env": env_config.to_dict(), "ppo": asdict(config)},
            )
            record_training_runtime(
                root,
                environment=env,
                task=env.task,
                learner=train_ppo,
                physics_adapter=env.physics,
                physics=env_config.physics,
                core_vector_env=True,
                packages=[
                    "numpy",
                    "torch",
                    *{
                        "numpy": [],
                        "mujoco": ["mujoco"],
                        "mjbatch": ["mujoco", "mjbatch"],
                        "newton": ["newton", "warp-lang"],
                    }[env_config.physics],
                ],
            )
            observation = env.observe()
            with (root / "metrics.jsonl").open("x") as metrics:
                for update in range(start + 1, start + config.updates + 1):
                    rollout, observation, successes, completed = _collect_rollout(
                        env, model, observation, config.rollout_steps
                    )
                    count = config.rollout_steps * env_config.num_envs
                    loss = _optimize_policy(model, optimizer, rollout, config, rng)
                    metadata["completed_updates"] = update
                    metric = {
                        "update": update,
                        "transitions": (update - start) * count,
                        "mean_step_reward": float(rollout["reward"].mean()),
                        "loss": loss,
                        "completed_episodes": completed,
                        "success_rate": successes / completed if completed else None,
                    }
                    metrics.write(json.dumps(metric, allow_nan=False) + "\n")
                    metrics.flush()
                    print(json.dumps(metric, allow_nan=False), flush=True)
                    if (
                        update == start + 1
                        or update % checkpoint_interval == 0
                        or update == start + config.updates
                    ):
                        publish_checkpoint(
                            root,
                            {
                                "schema_version": 2,
                                "updates": update,
                                "model_spec": model.model_spec,
                                "env_spec": spec,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "env": env_config.to_dict(),
                                "ppo": asdict(config),
                            },
                            optimizer,
                            metadata,
                        )
        metadata["status"] = "complete"
    except BaseException as error:
        if metadata is not None:
            metadata["status"] = (
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if metadata is not None:
            metadata["seconds"] = time.monotonic() - began
            active_error = sys.exc_info()[0] is not None
            try:
                write_json(root / "run.json", metadata)
            except Exception:
                if not active_error:
                    raise
                logging.getLogger(__name__).exception(
                    "Could not finalize core PPO run %s; preserving the training error",
                    root,
                )
    return model


@dataclass
class PolicyCheckpoint:
    """Evaluation bundle; configuration and dimensions travel with model weights."""

    policy: ActorCritic
    env_config: Config
    env_spec: dict
    sha256: str | None = None

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
        if spec["action_range"] != [
            self.policy.model_spec["action_low"],
            self.policy.model_spec["action_high"],
        ]:
            raise ValueError("Checkpoint/environment mismatch for policy action range")


def _policy_checkpoint(checkpoint: dict, checksum: str) -> PolicyCheckpoint:
    """Build the evaluation bundle from already-read and hashed checkpoint data."""
    if (
        not isinstance(checkpoint, dict)
        or type(checkpoint.get("schema_version")) is not int
        or checkpoint["schema_version"] not in (1, 2)
    ):
        raise ValueError("Unsupported checkpoint")
    env_spec = checkpoint.get("env_spec", {})
    if not isinstance(env_spec, dict):
        raise ValueError("Checkpoint environment specification must be a mapping")
    model = ActorCritic(**checkpoint.get("model_spec", {}))
    model.load_state_dict(checkpoint["model"])
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise ValueError(f"Non-finite checkpoint policy parameter: {name}")
    model.eval()
    env_config = Config(**checkpoint["env"])
    bundle = PolicyCheckpoint(
        model,
        env_config,
        env_spec,
        sha256=checksum,
    )
    if bundle.env_spec:
        bundle.validate_environment(bundle.env_spec)
    return bundle


def load_checkpoint(path: str | Path) -> PolicyCheckpoint:
    """Read v1/v2 checkpoints on CPU without constructing simulation resources."""
    # Deserialize and hash the same bytes even if the source file is replaced.
    payload = Path(path).read_bytes()
    checkpoint = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    return _policy_checkpoint(checkpoint, hashlib.sha256(payload).hexdigest())


@dataclass
class TrainingCheckpoint:
    """Verified policy/Adam input shared by CLI configuration and training."""

    bundle: PolicyCheckpoint
    ppo_config: PPOConfig
    updates: int
    optimizer_state: dict
    source: str


def load_training_checkpoint(path: str | Path) -> TrainingCheckpoint:
    """Load a committed core run or a trained v2 checkpoint exactly once."""
    source = read_resume(path)
    checkpoint = torch.load(
        io.BytesIO(source["payload"]), map_location="cpu", weights_only=True
    )
    bundle = _policy_checkpoint(checkpoint, source["sha256"])
    if checkpoint["schema_version"] != 2 or not isinstance(checkpoint.get("ppo"), dict):
        raise ValueError("Resume requires a trained core PPO v2 checkpoint")
    semantic_fields = {
        "task",
        "scene",
        "control_dt",
        "action_shape",
        "action_range",
        "action_units",
        "proprio_shape",
    }
    if not semantic_fields <= bundle.env_spec.keys():
        raise ValueError(
            "Resume requires a complete core PPO environment specification"
        )
    config = PPOConfig(**checkpoint["ppo"])
    updates = checkpoint.get("updates", config.updates)
    if type(updates) is not int or updates <= 0:
        raise ValueError("Invalid core PPO checkpoint update count")
    if source["saved_updates"] is not None and source["saved_updates"] != updates:
        raise ValueError("Core PPO checkpoint update count does not match run")
    optimizer = torch.optim.Adam(bundle.policy.parameters(), lr=config.learning_rate)
    state = checkpoint.get("optimizer")
    validate_adam_checkpoint(optimizer, state, label="core PPO optimizer")
    return TrainingCheckpoint(bundle, config, updates, state, source["source"])


def load_policy(path: str | Path) -> ActorCritic:
    """Compatibility convenience API returning just the evaluation policy."""
    return load_checkpoint(path).policy
