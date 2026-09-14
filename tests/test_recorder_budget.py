"""Bound retained payloads before copying an incoming transition batch."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, action_windows, read_episodes


def payload_bytes(writer):
    return sum(
        v.nbytes
        for rows in writer.pending.values()
        for row in rows
        for v in row.values()
    )


def test_budget_rejection_is_atomic_and_complete_episodes_release_bytes(tmp_path):
    with VectorEnv(Config(num_envs=2, max_steps=3)) as env:
        observation = env.observe()
        result = env.step(np.zeros((2, 2)))
        arrays = (
            *observation.values(),
            *result.observation.values(),
            result.info["executed_action"],
            result.reward,
            result.terminated,
            result.truncated,
        )
        per_batch = sum(value.nbytes for value in arrays)
        with EpisodeRecorder(
            tmp_path / "limited", env, max_buffer_bytes=2 * per_batch - 1
        ) as writer:
            writer.append(observation, result)
            assert writer.buffered_bytes == payload_bytes(writer) == per_batch
            observation = result.observation
            result = env.step(np.zeros((2, 2)))
            with pytest.raises(BufferError, match="buffer budget exceeded"):
                writer.append(observation, result)
            assert writer.buffered_bytes == per_batch
            assert all(len(rows) == 1 for rows in writer.pending.values())
            assert writer.failure is None
        env.reset()
        with EpisodeRecorder(
            tmp_path / "run", env, max_buffer_bytes=3 * per_batch
        ) as writer:
            for index in range(3):
                observation = env.observe()
                writer.append(observation, env.step(np.zeros((2, 2))))
                assert writer.buffered_bytes == payload_bytes(writer)
                assert writer.buffered_bytes == (
                    (index + 1) * per_batch if index < 2 else 0
                )
            assert not writer.pending
        assert len(list(read_episodes(tmp_path / "run"))) == 2


def test_image_payloads_and_incomplete_close_are_accounted(tmp_path):
    with VectorEnv(
        Config(num_envs=2, render="raster", channels=("rgb", "depth"))
    ) as env:
        writer = EpisodeRecorder(tmp_path / "run", env)
        observation = env.observe()
        writer.append(observation, env.step(np.zeros((2, 2))))
        assert writer.buffered_bytes == payload_bytes(writer)
        assert writer.buffered_bytes >= 2 * observation["rgb"].nbytes
        writer.close()
        assert writer.buffered_bytes == 0
        assert not writer.pending


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget_does_not_create_run(tmp_path, budget):
    root = tmp_path / "run"
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(ValueError, match="positive integer"):
            EpisodeRecorder(root, env, max_buffer_bytes=budget)
    assert not root.exists()


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_window_lengths_rejected_before_reading_files(tmp_path, value):
    with pytest.raises(ValueError, match="positive integers"):
        next(action_windows(tmp_path, history=value))
    with pytest.raises(ValueError, match="positive integers"):
        next(action_windows(tmp_path, horizon=value))
