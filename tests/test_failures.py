"""Failure boundaries preserve causes and never continue with partial state."""

import json

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes
from embodiedforge.policies import ActionExecutor


def test_backend_error_closes_both_resources_and_preserves_cause(monkeypatch):
    env = VectorEnv(Config(num_envs=1))
    closed = []

    def broken_step(*args):
        raise ValueError("solver failed after mutation")

    def close_render():
        closed.append("render")
        raise OSError("render cleanup failed")

    def close_physics():
        closed.append("physics")
        raise OSError("physics cleanup failed")

    monkeypatch.setattr(env.physics, "step", broken_step)
    monkeypatch.setattr(env.renderer, "close", close_render)
    monkeypatch.setattr(env.physics, "close", close_physics)
    with pytest.raises(ValueError, match="solver failed"):
        env.step(np.zeros((1, 2)))
    assert closed == ["render", "physics"]
    assert len(env.cleanup_errors) == 2
    for operation in (env.observe, env.reset, lambda: env.step(np.zeros((1, 2)))):
        with pytest.raises(RuntimeError, match="Environment failed") as error:
            operation()
        assert error.value.__cause__ is env.failure
    env.close()
    assert closed == ["render", "physics"]


def test_invalid_input_does_not_poison_environment():
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(ValueError):
            env.step(np.full((1, 2), np.nan))
        with pytest.raises(ValueError):
            env.reset([-1])
        assert env.failure is None
        assert env.step(np.zeros((1, 2))).observation["step_id"].tolist() == [1]


def test_context_cleanup_does_not_hide_user_exception(monkeypatch):
    env = VectorEnv()

    def broken_close():
        raise OSError("cleanup failure")

    monkeypatch.setattr(env.renderer, "close", broken_close)
    with pytest.raises(KeyError, match="original"):
        with env:
            raise KeyError("original")
    assert isinstance(env.cleanup_errors[0], OSError)


def test_recorder_validates_all_rows_before_publishing(tmp_path):
    root = tmp_path / "episodes"
    with (
        VectorEnv(Config(num_envs=2, max_steps=1)) as env,
        EpisodeRecorder(root, env) as writer,
    ):
        obs = env.observe()
        result = env.step(np.zeros((2, 2)))
        result.observation["step_id"][1] = 8
        with pytest.raises(ValueError, match="advance exactly one step"):
            writer.append(obs, result)
        assert writer.pending == {}
        assert not list(root.glob("*.npz"))
        result.observation["step_id"][1] = 1
        writer.append(obs, result)
    assert len(list(read_episodes(root))) == 2


def test_recorder_rejects_changed_observation_without_appending(tmp_path):
    with (
        VectorEnv(Config(num_envs=1)) as env,
        EpisodeRecorder(tmp_path / "run", env) as writer,
    ):
        obs = env.observe()
        writer.append(obs, env.step(np.zeros((1, 2))))
        obs = env.observe()
        obs["proprio"][:] = 10
        with pytest.raises(ValueError, match="continuity"):
            writer.append(obs, env.step(np.zeros((1, 2))))
        assert len(writer.pending[(0, 0)]) == 1


def test_storage_failure_disables_append_and_does_not_publish(tmp_path, monkeypatch):
    root = tmp_path / "run"

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    with (
        VectorEnv(Config(num_envs=1, max_steps=1)) as env,
        EpisodeRecorder(root, env) as writer,
    ):
        obs = env.observe()
        result = env.step(np.zeros((1, 2)))
        monkeypatch.setattr(np, "savez_compressed", fail_write)
        with pytest.raises(OSError, match="disk full"):
            writer.append(obs, result)
        with pytest.raises(RuntimeError, match="Recorder failed"):
            writer.append(obs, result)
    assert json.loads((root / "run.json").read_text())["status"] == "failed"
    assert list(read_episodes(root)) == []


def test_chunks_replan_only_reset_or_discontinuous_rows():
    class Policy:
        def __init__(self):
            self.calls = []

        def act(self, obs):
            self.calls.append(obs["env_id"].tolist())
            return np.tile(
                np.array([[[0.1], [0.2], [0.3]]]), (len(obs["env_id"]), 1, 1)
            )

    policy = Policy()
    executor = ActionExecutor(2, 1, 3)
    obs = {
        "env_id": np.arange(2),
        "episode_id": np.zeros(2, dtype=int),
        "step_id": np.zeros(2, dtype=int),
    }
    executor.act(obs, policy)
    obs["step_id"][:] = 1
    obs["episode_id"][1] = 1
    obs["step_id"][1] = 0
    np.testing.assert_allclose(executor.act(obs, policy), [[0.2], [0.1]])
    assert policy.calls == [[0, 1], [1]]
    obs["step_id"][:] = [4, 1]
    np.testing.assert_allclose(executor.act(obs, policy), [[0.1], [0.2]])
    assert policy.calls[-1] == [0]


@pytest.mark.parametrize("field", ["next/episode_id", "next/time", "obs/proprio"])
def test_reader_rejects_corrupt_transition_identity_or_continuity(tmp_path, field):
    root = tmp_path / "run"
    with (
        VectorEnv(Config(num_envs=1, max_steps=3)) as env,
        EpisodeRecorder(root, env) as writer,
    ):
        for _ in range(3):
            obs = env.observe()
            writer.append(obs, env.step(np.zeros((1, 2))))
    path = next(root.glob("*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    data[field][1] += 10
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        list(read_episodes(root))
