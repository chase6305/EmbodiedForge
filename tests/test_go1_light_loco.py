"""Numerical contracts for the optional upstream PPO objective and GAE."""

from copy import deepcopy
from importlib import import_module

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("light_loco_parkour")
pytest.importorskip("mjbatch")

llp = import_module("embodiedforge.locomotion.go1_light_loco")
native = import_module("embodiedforge.locomotion.go1_ppo")


def test_source_is_pinned_and_modified_source_is_rejected(monkeypatch):
    assert llp.verify_source()["revision"] == llp.REVISION
    monkeypatch.setitem(llp.SOURCE_HASHES, "__init__.py", "0" * 64)
    with pytest.raises(ValueError, match="source differs"):
        llp.verify_source()


@pytest.mark.parametrize("length", [1, 4, 13])
def test_gae_matches_reference_with_episode_boundaries(length):
    torch.manual_seed(4)
    batch = {
        "rew": torch.randn(length, 3),
        "val": torch.randn(length, 3),
        "last_val": torch.randn(3),
        "alive": torch.randint(0, 2, (length, 3)).float(),
    }
    expected = native.gae(batch)
    actual = llp.gae(batch)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_external_objective_has_equivalent_native_gradients():
    torch.manual_seed(11)
    torch.set_num_threads(1)
    net = native.ActorCritic()
    obs, actions = torch.randn(16, 50), torch.randn(16, 12)
    mean, values = net(obs)
    old_logp = native.log_density(
        (actions - mean) / net.log_std.exp(), net.log_std
    ).detach()
    advantage, returns = torch.randn(16), torch.randn(16)
    logp = native.log_density((actions - mean) / net.log_std.exp(), net.log_std)
    ratios = (logp - old_logp).exp()
    loss = -torch.minimum(ratios * advantage, ratios.clamp(0.8, 1.2) * advantage).mean()
    loss = (
        loss
        + 0.5 * (values - returns).square().mean()
        - native.ENT_COEF * net.log_std.sum()
    )
    expected = torch.autograd.grad(loss, tuple(net.parameters()))
    external = llp.make_loss(net)(obs, actions, old_logp, advantage, returns)
    actual = torch.autograd.grad(external, tuple(net.parameters()))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)


def test_update_changes_weights_and_checkpoint_remains_native_compatible():
    torch.manual_seed(3)
    net = native.ActorCritic()
    original = deepcopy(net.state_dict())
    obs = torch.randn(4, 4, 50)
    with torch.no_grad():
        mean, values = net(obs.flatten(0, 1))
        actions = mean + net.log_std.exp() * torch.randn_like(mean)
        logp = native.log_density((actions - mean) / net.log_std.exp(), net.log_std)
    batch = dict(
        obs=obs,
        act=actions.reshape(4, 4, 12),
        logp=logp.reshape(4, 4),
        val=values.reshape(4, 4),
        rew=torch.randn(4, 4),
        alive=torch.ones(4, 4),
        last_val=torch.zeros(4),
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    metrics = llp.update(net, opt, batch, *llp.gae(batch), diagnostics=True)
    assert metrics["ppo_mean_gradient_norm"] > 0
    assert not torch.equal(
        original["actor.0.weight"], net.state_dict()["actor.0.weight"]
    )
    restored = native.ActorCritic().eval()
    restored.load_state_dict(net.state_dict(), strict=True)
    torch.testing.assert_close(restored.action_mean(obs[0]), net.action_mean(obs[0]))
