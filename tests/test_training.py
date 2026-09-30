import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodiedforge import Config
from embodiedforge.training import (
    ActorCritic,
    PPOConfig,
    _optimize_policy,
    generalized_advantage,
    load_checkpoint,
    load_policy,
    train_ppo,
)


@pytest.mark.parametrize(
    "options",
    [
        {"updates": 1.5},
        {"rollout_steps": True},
        {"epochs": 0},
        {"minibatch_size": float("nan")},
        {"learning_rate": float("inf")},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": False},
    ],
)
def test_invalid_ppo_config_is_rejected_before_training(options):
    with pytest.raises(ValueError):
        PPOConfig(**options)


def test_nonfinite_gradient_does_not_update_policy_or_adam():
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters())
    observation = torch.zeros(2, 6)
    action = torch.zeros(2, 2)
    with torch.no_grad():
        logprob = model.distribution(observation).log_prob(action).sum(-1)
    rollout = {
        "obs": observation.numpy()[None],
        "raw": action.numpy()[None],
        "logprob": logprob.numpy()[None],
        "reward": np.array([[1, 2]], dtype=np.float32),
        "value": np.zeros((1, 2), dtype=np.float32),
        "next_value": np.zeros((1, 2), dtype=np.float32),
        "terminated": np.ones((1, 2), dtype=bool),
        "truncated": np.zeros((1, 2), dtype=bool),
    }
    before = {key: value.clone() for key, value in model.state_dict().items()}
    # A finite forward loss does not guarantee finite derivatives.
    handle = model.log_std.register_hook(lambda grad: grad * float("nan"))
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            _optimize_policy(
                model, optimizer, rollout, PPOConfig(epochs=1), np.random.default_rng(0)
            )
    finally:
        handle.remove()
    assert not optimizer.state
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)


def test_gae_bootstraps_timeout_but_does_not_cross_reset():
    reward = np.array([[1.0, 1.0], [100.0, 100.0]], dtype=np.float32)
    value = np.zeros_like(reward)
    next_value = np.array([[10.0, 10.0], [0.0, 0.0]], dtype=np.float32)
    terminated = np.array([[True, False], [True, True]])
    truncated = np.array([[False, True], [False, False]])
    adv, returns = generalized_advantage(
        reward, value, next_value, terminated, truncated, 0.9, 0.95
    )
    np.testing.assert_allclose(adv[0], [1.0, 10.0])
    np.testing.assert_allclose(returns, adv)


@pytest.mark.parametrize("backend", ["core", "h1"])
def test_ppo_exploration_recovers_from_overshot_bounds(backend):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if backend == "h1":
        pytest.importorskip("mujoco")
        pytest.importorskip("mjbatch")
        from embodiedforge.locomotion.h1_ppo import Policy, optimize

        model, observation = Policy(), torch.zeros(8, 69)
    else:
        model, observation = ActorCritic(), torch.zeros(8, 6)
    with torch.no_grad():
        model.log_std[::2] = 2.01
        model.log_std[1::2] = -5.01
        distribution = model.distribution(observation)
        offset = torch.zeros_like(distribution.loc)
        # Positive advantages favor lower variance on even dimensions and
        # higher variance on odd dimensions; negative samples do the reverse.
        offset[:4, 1::2] = 3
        offset[4:, ::2] = 3
        action = distribution.loc + offset * distribution.scale
        logprob = distribution.log_prob(action).sum(-1)
    model.distribution(observation).log_prob(action).sum().backward()
    assert torch.count_nonzero(model.log_std.grad) == 0
    optimizer = torch.optim.Adam(
        [
            {"params": [model.log_std], "lr": 0.0001},
            {
                "params": [p for p in model.parameters() if p is not model.log_std],
                "lr": 0,
            },
        ]
    )
    rollout = {
        "obs": observation.numpy()[None],
        "action" if backend == "h1" else "raw": action.numpy()[None],
        "logp" if backend == "h1" else "logprob": logprob.numpy()[None],
        "reward": np.array([[1] * 4 + [-1] * 4], dtype=np.float32),
        "value": np.zeros((1, 8), dtype=np.float32),
        "next_value": np.zeros((1, 8), dtype=np.float32),
        "terminated": np.ones((1, 8), dtype=bool),
        "truncated": np.zeros((1, 8), dtype=bool),
    }
    rng = np.random.default_rng(0)
    if backend == "h1":
        optimize(model, optimizer, rollout, rng, epochs=2)
    else:
        _optimize_policy(
            model, optimizer, rollout, PPOConfig(epochs=2, minibatch_size=2), rng
        )
    assert torch.all(model.log_std[::2] < 2)
    assert torch.all(model.log_std[1::2] > -5)
    assert torch.all((model.log_std >= -5) & (model.log_std <= 2))


@pytest.mark.parametrize("task,features", [("reach", 6), ("hold", 4)])
def test_ppo_checkpoint_roundtrip(tmp_path, task, features):
    torch.set_num_threads(1)
    model = train_ppo(
        Config(task=task, num_envs=4, max_steps=8),
        PPOConfig(updates=2, rollout_steps=16, epochs=1, minibatch_size=32),
        tmp_path / "train",
    )
    restored = load_policy(tmp_path / "train/checkpoint.pt")
    runtime = json.loads((tmp_path / "train/training-runtime.json").read_text())
    assert runtime["core_vector_env"] is True
    assert runtime["environment"]["module"] == "embodiedforge.env"
    assert runtime["learner"]["symbol"] == "train_ppo"
    assert runtime["task"]["project_namespace"]
    assert runtime["physics"] == "numpy"
    obs = {"proprio": np.ones((2, features), dtype=np.float32)}
    np.testing.assert_allclose(model.act(obs), restored.act(obs))
    assert np.isfinite(restored.act(obs)).all()
    bundle = load_checkpoint(tmp_path / "train/checkpoint.pt")
    assert bundle.env_config.task == task
    assert bundle.env_spec["proprio_shape"] == [features]
    wrong_spec = {**bundle.env_spec, "action_units": "radians"}
    with pytest.raises(ValueError, match="action_units"):
        bundle.validate_environment(wrong_spec)

    checkpoint = torch.load(tmp_path / "train/checkpoint.pt", weights_only=True)
    damaged = tmp_path / "damaged.pt"
    for name, nonfinite in (
        ("actor.0.weight", float("nan")),
        ("log_std", float("inf")),
    ):
        original = checkpoint["model"][name].clone()
        checkpoint["model"][name].flatten()[0] = nonfinite
        torch.save(checkpoint, damaged)
        with pytest.raises(
            ValueError, match=f"Non-finite checkpoint policy parameter: {name}"
        ):
            load_policy(damaged)
        checkpoint["model"][name] = original
    for low, high in ((-1, float("inf")), (1, -1), (0, 0), (-0.5, 0.5)):
        checkpoint["model_spec"].update(action_low=low, action_high=high)
        torch.save(checkpoint, damaged)
        with pytest.raises(ValueError, match="action range"):
            load_policy(damaged)


def test_legacy_checkpoint_load(tmp_path):
    from embodiedforge.training import ActorCritic

    policy = ActorCritic()
    config = Config().to_dict()
    del config["task"]
    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": 1, "model": policy.state_dict(), "env": config}, path)
    bundle = load_checkpoint(path)
    restored = bundle.policy
    obs = {"proprio": np.zeros((1, 6), dtype=np.float32)}
    np.testing.assert_array_equal(restored.act(obs), policy.act(obs))
    spec = {"proprio_shape": [6], "action_shape": [2], "action_range": [-1.0, 1.0]}
    bundle.validate_environment(spec)
    with pytest.raises(ValueError, match="action range"):
        bundle.validate_environment({**spec, "action_range": [-2.0, 2.0]})


@pytest.mark.parametrize("task,physics", [("reach", "mujoco"), ("hold", "numpy")])
def test_learning_comparison_uses_checkpoint_task_and_physics(
    tmp_path, monkeypatch, capsys, task, physics
):
    if physics == "mujoco":
        pytest.importorskip("mujoco")
    import embodiedforge

    torch.set_num_threads(1)
    config = Config(task=task, physics=physics, num_envs=4, max_steps=8)
    train_ppo(
        config,
        PPOConfig(updates=1, rollout_steps=8, epochs=1, minibatch_size=16),
        tmp_path / "train",
    )
    capsys.readouterr()
    actual_configs = []
    original_env = embodiedforge.VectorEnv

    def environment(config):
        actual_configs.append(config)
        return original_env(config)

    monkeypatch.setattr(embodiedforge, "VectorEnv", environment)
    checkpoint = tmp_path / "train/checkpoint.pt"
    expected_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    replacement = tmp_path / "replacement.pt"
    replaced_state = torch.load(checkpoint, weights_only=True)
    replaced_state["env"]["seed"] += 1
    torch.save(replaced_state, replacement)
    original_load = torch.load

    def load_and_replace(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        replacement.replace(checkpoint)
        return loaded

    monkeypatch.setattr(torch, "load", load_and_replace)
    script = Path(__file__).resolve().parents[1] / "benchmarks/check_learning.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), str(checkpoint), "--num-envs", "3", "--seeds", "7"]
    )
    runpy.run_path(str(script), run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report["metadata"]["checkpoint_sha256"] == expected_digest
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_digest
    assert len(actual_configs) == 2
    for actual in actual_configs:
        assert (actual.task, actual.physics, actual.max_steps) == (task, physics, 8)
        assert (actual.num_envs, actual.seed) == (3, 7)
    assert report["metadata"]["training_environment"]["task"] == task
    assert report["metadata"]["training_environment"]["seed"] == config.seed
    for policy in ("untrained_seed_0", "trained"):
        assert report[policy][0]["seed"] == 7
        assert np.isfinite(report[policy][0]["mean_episode_return"])
