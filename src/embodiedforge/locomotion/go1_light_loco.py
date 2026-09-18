"""Experimental upstream Agent PPO/GAE adapter for the native Go1 policy.

Keeps Go1's mirrored network, action scaling, normalization, and checkpoint
format. Multi-skill teachers, perception, and distillation are not enabled.
"""

import hashlib
from importlib import metadata
from pathlib import Path

import torch
from torch import nn

from . import go1_ppo as native

REVISION = "963a6ec3b8b42eb29dd6b9dfed34ededd3c64c7b"
SOURCE_HASHES = {
    "__init__.py": "bd5a0a5bb2ed51d61e5c8834a66385b3044db1962b14f4bc3f7393a9d67db002",
    "light_loco_parkour.py": "6549948530162ccd54ea969f3fbfcb7b80bc69c590e9549504895e6dedc9e5d5",
}


def verify_source():
    """Verify loaded source, not just a mutable package version string."""
    import light_loco_parkour

    root = Path(light_loco_parkour.__file__).resolve().parent
    for name, expected in SOURCE_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(
                f"Light Loco Parkour source differs from {REVISION}: {name}"
            )
    return {
        "repository": "https://github.com/lucidrains/light-loco-parkour",
        "revision": REVISION,
        "version": metadata.version("light-loco-parkour"),
        "path": str(root),
        "files": SOURCE_HASHES.copy(),
    }


class ActorAdapter(nn.Module):
    use_last_action = False

    def __init__(self, net):
        super().__init__()
        from light_loco_parkour import Gaussian

        self.net = net
        self.action_distr = Gaussian(raw_bounds=None, min_std=0.0)

    @property
    def next_latent_prediction_loss(self):
        return self.net.log_std.new_zeros(())

    def forward(self, states, **kwargs):
        mean = self.net.action_mean(states.flatten(0, 1)).reshape(
            *states.shape[:2], native.ACT_DIM
        )
        params = torch.stack((mean, self.net.log_std.expand_as(mean)), dim=-1)
        return self.action_distr(params), None


class CriticAdapter(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, states, *, target=None, **kwargs):
        _, values = self.net(states.flatten(0, 1))
        values = values.reshape(states.shape[:2])
        return (values if target is None else (values - target).square().mean()), None


def make_loss(net):
    from light_loco_parkour import Agent

    # Advantages are normalized once by the shared update loop. No history or
    # recurrent state: each minibatch row is a one-step sequence [batch, 1, dim].
    agent = Agent(
        ActorAdapter(net),
        CriticAdapter(net),
        clip=native.CLIP,
        entropy_weight=native.ENT_COEF,
        norm_advantages=False,
    )

    def loss(obs, actions, old_logp, advantages, returns):
        states = obs[:, None, :]
        return agent.actor_loss(
            states,
            actions[:, None, :],
            old_logp[:, None],
            advantages[:, None],
        ) + 0.5 * agent.critic_loss(states, returns[:, None])

    return loss


def gae(batch):
    from light_loco_parkour import Agent

    # Reuse validation and the rollout's timeout bootstrap correction. Upstream
    # consumes [environment, time], while the native collector is time-major.
    native.gae(batch)
    values = torch.cat((batch["val"], batch["last_val"][None]), dim=0).T
    returns = Agent.calc_gae(
        batch["rew"].T,
        values,
        batch["alive"].T,
        gamma=native.GAMMA,
        lam=native.LAMBDA,
        use_accelerated=False,
    ).T.contiguous()
    return returns - batch["val"], returns


def update(net, opt, batch, adv, ret, *, diagnostics=False):
    return native.update(
        net,
        opt,
        batch,
        adv,
        ret,
        diagnostics=diagnostics,
        loss_function=make_loss(net),
    )
