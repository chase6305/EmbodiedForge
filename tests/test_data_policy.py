import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, action_windows, read_episodes
from embodiedforge.policies import ActionExecutor


def test_episode_roundtrip_and_windows(tmp_path):
    root = tmp_path / "run"
    with VectorEnv(
        Config(num_envs=2, max_steps=5, render="raster", channels=("rgb",))
    ) as env:
        with EpisodeRecorder(root, env) as recorder:
            for _ in range(2):
                for _ in range(5):
                    obs = env.observe()
                    result = env.step(np.zeros((2, 2)))
                    recorder.append(obs, result)
                env.reset()
    episodes = list(read_episodes(root))
    assert len(episodes) == 4
    for e in episodes:
        assert e["truncated"].tolist() == [False] * 4 + [True]
        np.testing.assert_array_equal(e["next/step_id"], np.arange(1, 6))
        np.testing.assert_array_equal(e["obs/proprio"][1:], e["next/proprio"][:-1])
        assert e["next/frame_age"][-1] == 0
    windows = list(action_windows(root, history=2, horizon=3))
    assert len(windows) == 8
    assert windows[0]["action"].shape == (3, 2)
    assert windows[0]["observation"]["rgb"].shape == (2, 1, 64, 64, 3)
    assert windows[0]["instruction"]
    assert not list(action_windows(root, history=6))


def test_incomplete_not_exposed_and_snapshot_not_aliased(tmp_path):
    root = tmp_path / "run"
    with VectorEnv(Config(num_envs=1, max_steps=3)) as env:
        with EpisodeRecorder(root, env) as recorder:
            obs = env.observe()
            result = env.step(np.zeros((1, 2)))
            recorder.append(obs, result)
            expected = recorder.pending[(0, 0)][0]["obs/proprio"].copy()
            obs["proprio"][:] = 99
            np.testing.assert_array_equal(
                recorder.pending[(0, 0)][0]["obs/proprio"], expected
            )
    assert not list(read_episodes(root))


def test_chunk_reset_only_replans_selected_rows():
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
    obs = {"env_id": np.arange(2)}
    np.testing.assert_allclose(executor.act(obs, policy), [[0.1], [0.1]])
    executor.reset([1])
    np.testing.assert_allclose(executor.act(obs, policy), [[0.2], [0.1]])
    assert policy.calls == [[0, 1], [1]]


def test_recorder_rejects_missing_transition(tmp_path):
    with (
        VectorEnv(Config(num_envs=1)) as env,
        EpisodeRecorder(tmp_path / "run", env) as writer,
    ):
        env.step(np.zeros((1, 2)))
        obs = env.observe()
        result = env.step(np.zeros((1, 2)))
        with pytest.raises(ValueError, match="consecutive"):
            writer.append(obs, result)


def test_recorder_rejects_early_reset_instead_of_accumulating_buffers(tmp_path):
    with (
        VectorEnv(Config(num_envs=1)) as env,
        EpisodeRecorder(tmp_path / "run", env) as writer,
    ):
        obs = env.observe()
        writer.append(obs, env.step(np.zeros((1, 2))))
        env.reset()
        obs = env.observe()
        with pytest.raises(ValueError, match="before its recorded episode ended"):
            writer.append(obs, env.step(np.zeros((1, 2))))
        assert len(writer.pending) == 1


def test_executor_uses_task_bounds_and_rejects_invalid_reset_atomically():
    class ForcePolicy:
        def act(self, observation):
            return np.full((len(observation["env_id"]), 2, 1), 1.5, dtype=np.float32)

    executor = ActionExecutor(2, 1, 2, action_low=-2, action_high=2)
    observation = {"env_id": np.arange(2)}
    np.testing.assert_allclose(executor.act(observation, ForcePolicy()), [[1.5], [1.5]])
    cursor = executor.cursor.copy()
    for ids in ([-1], [2], [0, 0], [[0]]):
        with pytest.raises(ValueError):
            executor.reset(ids)
        np.testing.assert_array_equal(executor.cursor, cursor)
