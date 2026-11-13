"""Protect temporal foot snapshots and avoid materializing unused PPO inputs."""

import numpy as np
import pytest


@pytest.mark.parametrize("family", ["core", "h1"])
@pytest.mark.parametrize("transposed", [False, True])
def test_ppo_leaves_unused_rollout_fields_unmaterialized(
    monkeypatch, family, transposed
):
    torch = pytest.importorskip("torch")
    if family == "core":
        from embodiedforge import training as module

        model = module.ActorCritic()
        obs_dim, act_dim = 6, 2
        action_key, logp_key = "raw", "logprob"
        used = ("obs", "raw", "logprob")
    else:
        from embodiedforge.locomotion import h1_ppo as module

        model = module.Policy()
        obs_dim, act_dim = module.OBS_DIM, module.ACT_DIM
        action_key, logp_key = "action", "logp"
        used = ("obs", "action", "logp", "value")
    rng = np.random.default_rng(19)
    shapes = {"obs": (obs_dim,), action_key: (act_dim,), logp_key: ()}
    shapes.update(
        {
            key: ()
            for key in ("reward", "value", "next_value", "terminated", "truncated")
        }
    )
    rollout = {}
    for key, shape in shapes.items():
        value = rng.normal(0, 0.1, (4, 2, *shape)).astype(np.float32)
        if key in ("terminated", "truncated"):
            value = np.zeros(value.shape, bool)
        rollout[key] = (
            value.swapaxes(0, 1) if transposed else value.reshape(2, 4, *shape)
        )
    saved = {key: value.copy() for key, value in rollout.items()}
    convert = torch.as_tensor
    touched = []

    def track(value, *args, **kwargs):
        if isinstance(value, np.ndarray):
            touched.extend(
                key
                for key, source in rollout.items()
                if np.shares_memory(value, source)
            )
        return convert(value, *args, **kwargs)

    monkeypatch.setattr(torch, "as_tensor", track)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    if family == "core":
        loss = module._optimize_policy(
            model, optimizer, rollout, module.PPOConfig(epochs=1, minibatch_size=4), rng
        )
    else:
        loss = module.optimize(model, optimizer, rollout, rng, epochs=1)
    assert np.isfinite(loss)
    # Contiguous reshapes share storage; transposed reshapes can own their copy.
    assert set(touched) <= set(used)
    if not transposed:
        assert set(touched) == set(used)
    for key, value in saved.items():
        np.testing.assert_array_equal(rollout[key], value)
    assert optimizer.state
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("semantics", ["transition-v2", "upstream-v1"])
def test_go1_landing_uses_owned_pre_integration_velocity(monkeypatch, semantics):
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    pytest.importorskip("mujoco_menagerie")
    from embodiedforge.locomotion import go1

    env = go1.Go1(4, seed=83, num_threads=1, semantics=semantics)
    env.down[:] = False
    velocities = [value.copy() for value in env.foot_vel]
    for index, value in enumerate(env.foot_vel):
        value[:] = index + 1
    expected = np.stack(env.foot_vel, 1)
    captured = []
    stack = np.stack

    def track(values, *args, **kwargs):
        result = stack(values, *args, **kwargs)
        if values is env.foot_vel:
            captured.append(result)
        return result

    monkeypatch.setattr(go1.np, "stack", track)
    _, _, _, terms = env.step(np.zeros((4, 12), np.float32))
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], expected)
    assert all(not np.shares_memory(captured[0], value) for value in env.foot_vel)
    np.testing.assert_array_equal(
        terms["land"], np.sum(env.down * np.sum(expected**2, 2), 1)
    )
    # Physics advances the borrowed sensor views while the saved snapshot stays fixed.
    assert any(
        not np.array_equal(value, old)
        for value, old in zip(env.foot_vel, velocities, strict=True)
    )
