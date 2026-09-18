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

from .go1 import REWARD

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
OBS_DIM, ACT_DIM = 50, 12


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
        """Average the net's answer with its answer on the mirrored observation mirrored back, so
        the policy commutes with the mirror exactly and the gait cannot limp."""
        flip = (obs[:, self.mirror] * self.sign - self.mean) / (self.var.sqrt() + 1e-5)
        obs = (obs - self.mean) / (self.var.sqrt() + 1e-5)
        mean = self.actor(obs) + self.actor(flip)[:, self.legs] * self.leg_sign
        return 0.5 * mean, obs

    def action_mean(self, obs):
        """Compute policy actions without evaluating the critic."""
        return self._action_and_normalized_obs(obs)[0]

    def forward(self, obs):
        mean, normalized = self._action_and_normalized_obs(obs)
        return mean, self.critic(normalized).squeeze(-1)


@torch.no_grad()
def rollout(actor, env, *, horizon=24, record_policy=False):
    if type(horizon) is not int or horizon <= 0:
        raise ValueError("Rollout horizon must be a positive integer")

    def policy(obs):
        return tuple(x.cpu().numpy() for x in actor(torch.as_tensor(obs)))

    shapes = dict(obs=(OBS_DIM,), act=(ACT_DIM,), logp=(), val=(), rew=(), alive=())
    if record_policy:
        shapes["policy_mean"] = (ACT_DIM,)
    buf = {
        k: np.empty((horizon, len(env.steps), *v), np.float32)
        for k, v in shapes.items()
    }
    means, falls, episodes = [], 0, 0
    obs = env.obs()
    for t in range(horizon):
        mean, val = policy(obs)
        log_std = actor.log_std.cpu().numpy()
        noise = env.rng.standard_normal(mean.shape, np.float32)
        act, logp = mean + np.exp(log_std) * noise, log_density(noise, log_std)
        reward, done, fell, terms = env.step(act)
        next_obs = env.obs()
        timeout = done & ~fell
        if timeout.any():  # bootstrap
            reward[timeout] += GAMMA * policy(next_obs[timeout])[1]
        for k, v in dict(
            obs=obs, act=act, logp=logp, val=val, rew=reward, alive=~done
        ).items():
            buf[k][t] = v
        if record_policy:
            buf["policy_mean"][t] = mean
        means.append([terms[k].mean() for k in REWARD])
        ids = np.flatnonzero(done)
        if ids.size:
            episodes, falls = episodes + ids.size, falls + int(fell[ids].sum())
            env.reset(ids)
            next_obs = env.obs()
        obs = next_obs
    batch = {k: torch.as_tensor(v) for k, v in buf.items()}
    if record_policy:
        batch["policy_log_std"] = actor.log_std.detach().clone()
    batch["last_val"] = torch.as_tensor(policy(obs)[1])
    stats = dict(zip(REWARD, np.mean(means, 0), strict=True))
    stats["falls"] = falls / max(episodes, 1)
    return batch, stats


def gae(batch):
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
    vals = torch.cat([batch["val"], batch["last_val"][None]])
    adv, carry = torch.zeros_like(batch["rew"]), 0.0
    for t in reversed(range(len(batch["rew"]))):
        alive = batch["alive"][t]
        delta = batch["rew"][t] + GAMMA * alive * vals[t + 1] - vals[t]
        adv[t] = carry = delta + GAMMA * LAMBDA * alive * carry
    return adv, adv + batch["val"]


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
    shape = batch["obs"].shape
    if (
        len(shape) != 3
        or shape[2] != OBS_DIM
        or shape[0] * shape[1] < 2 * MINIBATCHES
        or (shape[0] * shape[1]) % MINIBATCHES
    ):
        raise ValueError(
            "Go1 PPO requires complete minibatches with at least two samples each"
        )
    for name, value, expected in (
        ("obs", batch["obs"], shape),
        ("act", batch["act"], (*shape[:2], ACT_DIM)),
        ("logp", batch["logp"], shape[:2]),
        ("advantage", adv, shape[:2]),
        ("return", ret, shape[:2]),
    ):
        if (
            value.shape != expected
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                f"Invalid Go1 PPO {name}: expected finite CPU float32 with shape {tuple(expected)}"
            )
    recorded = "policy_mean" in batch or "policy_log_std" in batch
    if diagnostics and recorded:
        for name, expected in (
            ("policy_mean", (*shape[:2], ACT_DIM)),
            ("policy_log_std", (ACT_DIM,)),
        ):
            value = batch.get(name)
            if (
                value is None
                or value.shape != expected
                or value.dtype != torch.float32
                or value.device.type != "cpu"
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid Go1 recorded policy distribution: {name}")
    obs, act = batch["obs"].reshape(-1, OBS_DIM), batch["act"].reshape(-1, ACT_DIM)
    logp_old, adv, ret = batch["logp"].reshape(-1), adv.reshape(-1), ret.reshape(-1)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    if diagnostics:
        with torch.no_grad():
            old_mean = (
                batch["policy_mean"].reshape(-1, ACT_DIM)
                if recorded
                else net.action_mean(obs)
            )
            old_log_std = batch["policy_log_std"] if recorded else net.log_std.clone()
        gradient_norms = []
    for _ in range(EPOCHS):
        for i in torch.randperm(obs.shape[0]).chunk(MINIBATCHES):
            if loss_function is None:
                mean, val = net(obs[i])
                logp = log_density((act[i] - mean) / net.log_std.exp(), net.log_std)
                ratio = (logp - logp_old[i]).exp()
                surrogate = torch.min(
                    ratio * adv[i], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[i]
                )
                loss = (
                    -surrogate.mean()
                    + 0.5 * (val - ret[i]).pow(2).mean()
                    - ENT_COEF * net.log_std.sum()
                )
            else:
                loss = loss_function(obs[i], act[i], logp_old[i], adv[i], ret[i])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite Go1 PPO loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(
                net.parameters(), 1.0, error_if_nonfinite=True
            )
            if diagnostics:
                gradient_norms.append(float(norm))
            opt.step()
    if diagnostics:
        with torch.no_grad():
            learned_mean = net.action_mean(obs)
            learned_kl = policy_kl(old_mean, old_log_std, learned_mean, net.log_std)
    net.absorb(
        obs
    )  # after the epochs: the batch was collected under the old statistics
    if diagnostics:
        with torch.no_grad():
            final_mean = net.action_mean(obs)
            final_kl = policy_kl(old_mean, old_log_std, final_mean, net.log_std)
            normalizer_kl = policy_kl(
                learned_mean, net.log_std, final_mean, net.log_std
            )
            final_logp = log_density(
                (act - final_mean) / net.log_std.exp(), net.log_std
            )
            ratio = (final_logp - logp_old).exp()
            result = {
                "ppo_kl_before_normalization": float(learned_kl.mean()),
                "ppo_kl": float(final_kl.mean()),
                "ppo_max_kl": float(final_kl.max()),
                "ppo_normalizer_kl": float(normalizer_kl.mean()),
                "ppo_clip_fraction": float(((ratio - 1).abs() > CLIP).float().mean()),
                "ppo_mean_gradient_norm": float(np.mean(gradient_norms)),
                "ppo_action_std": float(net.log_std.exp().mean()),
            }
        if not all(np.isfinite(value) for value in result.values()):
            raise ValueError("Non-finite Go1 PPO diagnostic")
        return result
