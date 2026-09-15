import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodiedforge import Config
from embodiedforge.training import (
    PPOConfig,
    generalized_advantage,
    load_checkpoint,
    load_policy,
    train_ppo,
)


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


def test_legacy_checkpoint_load(tmp_path):
    from embodiedforge.training import ActorCritic

    policy = ActorCritic()
    config = Config().to_dict()
    del config["task"]
    path = tmp_path / "legacy.pt"
    torch.save({"schema_version": 1, "model": policy.state_dict(), "env": config}, path)
    restored = load_policy(path)
    obs = {"proprio": np.zeros((1, 6), dtype=np.float32)}
    np.testing.assert_array_equal(restored.act(obs), policy.act(obs))


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
    script = Path(__file__).resolve().parents[1] / "benchmarks/check_learning.py"
    monkeypatch.setattr(
        sys, "argv", [str(script), str(checkpoint), "--num-envs", "3", "--seeds", "7"]
    )
    runpy.run_path(str(script), run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert len(actual_configs) == 2
    for actual in actual_configs:
        assert (actual.task, actual.physics, actual.max_steps) == (task, physics, 8)
        assert (actual.num_envs, actual.seed) == (3, 7)
    assert report["metadata"]["training_environment"]["task"] == task
    for policy in ("untrained_seed_0", "trained"):
        assert report[policy][0]["seed"] == 7
        assert np.isfinite(report[policy][0]["mean_episode_return"])
