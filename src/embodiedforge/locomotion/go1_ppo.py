# SPDX-License-Identifier: Apache-2.0
# Copyright 2026, Kevin Zakka
# Derived from mjbatch/examples/go1_joystick.py at
# b84c0c20aedbdf048122cbc47f554e9b93cc4754.
# Modified for EmbodiedForge: split task/learner, explicit runtime arguments,
# packaged scene, CPU execution, and no upstream example or viewer imports.

"""CPU PPO for the Go1 task, with mirrored policy and running normalization."""

import numpy as np
import torch
from torch import nn

from .go1_config import ACT_DIM as ACT_DIM
from .go1_config import OBS_DIM as OBS_DIM
from .go1_config import REWARD

LEGS = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
LEG_SIGN = np.array([-1, 1, 1] * 4, np.float32)
MIRROR = np.concatenate(
    [np.arange(9), 9 + LEGS, 21 + LEGS, 33 + LEGS, np.arange(45, 50)]
)
SIGN = np.r_[
    [1, -1, 1, -1, 1, -1, 1, -1, 1], np.tile(LEG_SIGN, 3), [1, -1, -1, -1, -1]
].astype(np.float32)
GAMMA, LAMBDA, CLIP = 0.99, 0.95, 0.2
LR, LR_END, LR_TO = 1e-3, 5e-4, 400
EPOCHS, MINIBATCHES, ENT_COEF, LOG_STD, HIDDEN = 5, 4, 0.005, np.log(0.5), 128


def log_density(z, log_std):
    return -0.5 * (z * z).sum(-1) - log_std.sum() - 0.5 * ACT_DIM * np.log(2 * np.pi)


def mlp(out_dim):
    hidden = (nn.Linear(OBS_DIM, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, HIDDEN), nn.ELU())
    return nn.Sequential(*hidden, nn.Linear(HIDDEN, out_dim))


class ActorCritic(nn.Module):
    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self):
        super().__init__()
        self.actor, self.critic = mlp(ACT_DIM), mlp(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), LOG_STD))
        self.register_buffer("mean", torch.zeros(OBS_DIM))
        self.register_buffer("var", torch.ones(OBS_DIM))
        self.register_buffer("count", torch.full((), 1e-4))
        for name, x in (
            ("mirror", MIRROR),
            ("legs", LEGS),
            ("sign", SIGN),
            ("leg_sign", LEG_SIGN),
        ):
            self.register_buffer(name, torch.as_tensor(x), persistent=False)

    @torch.no_grad()
    def absorb(self, obs):  # Chan's parallel update of the running statistics
        n, delta = obs.shape[0], obs.mean(0) - self.mean
        total = self.count + n
        var = self.var * self.count + obs.var(0, correction=0) * n
        self.var.copy_((var + delta**2 * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.add_(n)

    def _action_and_normalized_obs(self, obs):
        """Average direct and mirrored actor means to enforce mirror equivariance.

        Return that mean with the normalized observations used by the critic.
        """
        scale = self.var.sqrt() + 1e-5
        flip = (obs[:, self.mirror] * self.sign - self.mean) / scale
        obs = (obs - self.mean) / scale
        mean = self.actor(obs) + self.actor(flip)[:, self.legs] * self.leg_sign
        return 0.5 * mean, obs

    def action_mean(self, obs):
        """Compute policy actions without evaluating the critic."""
        return self._action_and_normalized_obs(obs)[0]

    def value(self, obs):
        """Compute bootstrap values without evaluating the mirrored actor."""
        normalized = (obs - self.mean) / (self.var.sqrt() + 1e-5)
        return self.critic(normalized).squeeze(-1)

    def forward(self, obs):
        mean, normalized = self._action_and_normalized_obs(obs)
        return mean, self.critic(normalized).squeeze(-1)


@torch.no_grad()
def rollout(actor, env, *, horizon=24, record_policy=False):
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Rollout horizon must be a positive integer")

    def evaluate_actor(observation):
        return tuple(
            value.cpu().numpy() for value in actor(torch.as_tensor(observation))
        )

    field_shapes = dict(
        obs=(OBS_DIM,), act=(ACT_DIM,), logp=(), val=(), rew=(), alive=()
    )
    if record_policy:
        field_shapes["policy_mean"] = (ACT_DIM,)
    buffers = {
        name: np.empty((horizon, len(env.steps), *field_shape), np.float32)
        for name, field_shape in field_shapes.items()
    }
    reward_means = []
    fallen_episodes = 0
    completed_episodes = 0
    observation = env.obs()
    # The actor stays fixed during collection; sample from its current variance.
    policy_log_std = actor.log_std.cpu().numpy()
    action_std = np.exp(policy_log_std)
    for step in range(horizon):
        action_mean, value = evaluate_actor(observation)
        standard_noise = env.rng.standard_normal(action_mean.shape, np.float32)
        actions = action_mean + action_std * standard_noise
        log_prob = log_density(standard_noise, policy_log_std)
        reward, done, fell, reward_terms = env.step(actions)
        next_observation = env.obs()
        timed_out = done & ~fell
        # Time limits bootstrap from the terminal observation; falls do not.
        if timed_out.any():
            reward[timed_out] += (
                GAMMA
                * actor.value(torch.as_tensor(next_observation[timed_out]))
                .cpu()
                .numpy()
            )
        for name, data in dict(
            obs=observation,
            act=actions,
            logp=log_prob,
            val=value,
            rew=reward,
            alive=~done,
        ).items():
            buffers[name][step] = data
        if record_policy:
            buffers["policy_mean"][step] = action_mean
        reward_means.append([reward_terms[name].mean() for name in REWARD])
        done_ids = np.flatnonzero(done)
        if done_ids.size:
            completed_episodes = completed_episodes + done_ids.size
            fallen_episodes = fallen_episodes + int(fell[done_ids].sum())
            env.reset(done_ids)
            next_observation = env.obs()
        observation = next_observation
    batch = {name: torch.as_tensor(data) for name, data in buffers.items()}
    if record_policy:
        batch["policy_log_std"] = actor.log_std.detach().clone()
    batch["last_val"] = actor.value(torch.as_tensor(observation)).cpu()
    stats = dict(zip(REWARD, np.mean(reward_means, 0), strict=True))
    stats["falls"] = fallen_episodes / max(completed_episodes, 1)
    return batch, stats


def validate_gae_batch(batch):
    """Check the shared rollout contract without computing training targets."""
    reward = batch["rew"]
    if (
        reward.ndim != 2
        or 0 in reward.shape
        or not reward.is_floating_point()
        or reward.device.type != "cpu"
    ):
        raise ValueError("Go1 GAE requires a nonempty (time, environments) batch")
    shape = reward.shape
    for name, expected in (
        ("rew", shape),
        ("val", shape),
        ("alive", shape),
        ("last_val", shape[1:]),
    ):
        value = batch[name]
        if (
            value.shape != expected
            or value.device != reward.device
            or (name != "alive" and value.dtype != reward.dtype)
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid Go1 GAE {name} shape or values")
    if not ((batch["alive"] == 0) | (batch["alive"] == 1)).all():
        raise ValueError("Go1 GAE alive must contain only zero or one")


def gae(batch):
    validate_gae_batch(batch)
    values = torch.cat([batch["val"], batch["last_val"][None]])
    advantages = torch.empty_like(batch["rew"])
    carry = 0.0
    # Rollout already bootstrapped time limits. Both kinds of done stop the carry.
    temporal_difference = (
        batch["rew"] + GAMMA * batch["alive"] * values[1:] - values[:-1]
    )
    continuation_discount = GAMMA * LAMBDA * batch["alive"]
    for step in reversed(range(len(batch["rew"]))):
        next_carry = temporal_difference[step] + continuation_discount[step] * carry
        advantages[step] = next_carry
        carry = next_carry
    return advantages, advantages + batch["val"]


@torch.no_grad()
def policy_kl(old_mean, old_log_std, new_mean, new_log_std):
    """Exact KL(old || new) for each diagonal Gaussian action distribution."""
    log_variance_ratio = 2 * (old_log_std - new_log_std)
    return 0.5 * (
        log_variance_ratio.expm1()
        - log_variance_ratio
        + (old_mean - new_mean).square() * (-2 * new_log_std).exp()
    ).sum(-1)


def update(net, opt, batch, adv, ret, *, diagnostics=False, loss_function=None):
    observation_shape = batch["obs"].shape
    if (
        len(observation_shape) != 3
        or observation_shape[2] != OBS_DIM
        or observation_shape[0] * observation_shape[1] < 2 * MINIBATCHES
        or (observation_shape[0] * observation_shape[1]) % MINIBATCHES
    ):
        raise ValueError(
            "Go1 PPO requires complete minibatches with at least two samples each"
        )
    for name, value, expected_shape in (
        ("obs", batch["obs"], observation_shape),
        ("act", batch["act"], (*observation_shape[:2], ACT_DIM)),
        ("logp", batch["logp"], observation_shape[:2]),
        ("advantage", adv, observation_shape[:2]),
        ("return", ret, observation_shape[:2]),
    ):
        if (
            value.shape != expected_shape
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid Go1 PPO {name}: expected finite CPU float32 with shape {tuple(expected_shape)}"
            )
    has_recorded_policy = "policy_mean" in batch or "policy_log_std" in batch
    if diagnostics and has_recorded_policy:
        for name, expected_shape in (
            ("policy_mean", (*observation_shape[:2], ACT_DIM)),
            ("policy_log_std", (ACT_DIM,)),
        ):
            value = batch.get(name)
            if (
                value is None
                or value.shape != expected_shape
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid Go1 recorded policy distribution: {name}")
    observations = batch["obs"].reshape(-1, OBS_DIM)
    actions = batch["act"].reshape(-1, ACT_DIM)
    old_log_prob = batch["logp"].reshape(-1)
    advantages = adv.reshape(-1)
    returns = ret.reshape(-1)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    if diagnostics:
        with torch.no_grad():
            rollout_mean = (
                batch["policy_mean"].reshape(-1, ACT_DIM)
                if has_recorded_policy
                else net.action_mean(observations)
            )
            rollout_log_std = (
                batch["policy_log_std"] if has_recorded_policy else net.log_std.clone()
            )
        gradient_norms = []
    for _ in range(EPOCHS):
        for minibatch_ids in torch.randperm(observations.shape[0]).chunk(MINIBATCHES):
            if loss_function is None:
                action_mean, predicted_value = net(observations[minibatch_ids])
                log_prob = log_density(
                    (actions[minibatch_ids] - action_mean) / net.log_std.exp(),
                    net.log_std,
                )
                probability_ratio = (log_prob - old_log_prob[minibatch_ids]).exp()
                surrogate = torch.min(
                    probability_ratio * advantages[minibatch_ids],
                    probability_ratio.clamp(1 - CLIP, 1 + CLIP)
                    * advantages[minibatch_ids],
                )
                loss = (
                    -surrogate.mean()
                    + 0.5 * (predicted_value - returns[minibatch_ids]).pow(2).mean()
                    - ENT_COEF * net.log_std.sum()
                )
            else:
                loss = loss_function(
                    observations[minibatch_ids],
                    actions[minibatch_ids],
                    old_log_prob[minibatch_ids],
                    advantages[minibatch_ids],
                    returns[minibatch_ids],
                )
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Go1 PPO loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            if diagnostics:
                gradient_norms.append(float(gradient_norm))
            opt.step()
    if diagnostics:
        with torch.no_grad():
            # Isolate optimizer drift while rollout normalization is still fixed.
            trained_mean = net.action_mean(observations)
            training_kl = policy_kl(
                rollout_mean, rollout_log_std, trained_mean, net.log_std
            )
    # Rollout probabilities used the old statistics throughout every PPO epoch.
    net.absorb(observations)
    if diagnostics:
        with torch.no_grad():
            final_mean = net.action_mean(observations)
            final_kl = policy_kl(rollout_mean, rollout_log_std, final_mean, net.log_std)
            # Holding the trained weights fixed isolates normalization drift.
            normalizer_kl = policy_kl(
                trained_mean, net.log_std, final_mean, net.log_std
            )
            final_log_prob = log_density(
                (actions - final_mean) / net.log_std.exp(), net.log_std
            )
            probability_ratio = (final_log_prob - old_log_prob).exp()
            result = {
                "ppo_kl_before_normalization": float(training_kl.mean()),
                "ppo_kl": float(final_kl.mean()),
                "ppo_max_kl": float(final_kl.max()),
                "ppo_normalizer_kl": float(normalizer_kl.mean()),
                "ppo_clip_fraction": float(
                    ((probability_ratio - 1).abs() > CLIP).float().mean()
                ),
                "ppo_mean_gradient_norm": float(np.mean(gradient_norms)),
                "ppo_action_std": float(net.log_std.exp().mean()),
            }
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Non-finite Go1 PPO diagnostic")
        return result
