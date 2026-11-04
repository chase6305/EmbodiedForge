"""Reject schema drift before publishing or changing a buffered episode."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize(
    "change", ["drop_field", "dtype", "shape", "reward_dtype", "action_dtype"]
)
def test_episode_schema_drift_is_retryable(tmp_path, episode_format, workers, change):
    with VectorEnv(Config(num_envs=2, max_steps=2)) as env:
        with EpisodeRecorder(
            tmp_path / "run", env, episode_format=episode_format, write_workers=workers
        ) as writer:
            observation = env.observe()
            first = env.step(np.zeros((2, 2)))
            observation["extra"] = np.ones((2, 2), dtype=np.float64)
            first.observation["extra"] = observation["extra"].copy()
            writer.append(observation, first)
            second = env.step(np.zeros((2, 2)))
            second.observation["extra"] = first.observation["extra"].copy()
            before = {name: value.copy() for name, value in first.observation.items()}
            if change == "drop_field":
                del before["extra"], second.observation["extra"]
            elif change in ("dtype", "shape"):
                for batch in (before, second.observation):
                    batch["extra"] = (
                        batch["extra"].astype(np.float32)
                        if change == "dtype"
                        else batch["extra"][:, :1]
                    )
            elif change == "reward_dtype":
                second.reward = second.reward.astype(np.float64)
            else:
                second.info["executed_action"] = second.info["executed_action"].astype(
                    np.float64
                )
            retained = writer.buffered_bytes
            with pytest.raises(ValueError, match="layout|schema"):
                writer.append(before, second)
            assert writer.buffered_bytes == retained
            assert all(len(rows) == 1 for rows in writer.pending.values())
            assert writer.failure is None and not writer._writes
            assert not list(writer.root.glob("*.npz"))
            second.observation["extra"] = first.observation["extra"].copy()
            second.reward = second.reward.astype(first.reward.dtype)
            second.info["executed_action"] = second.info["executed_action"].astype(
                first.info["executed_action"].dtype
            )
            writer.append(first.observation, second)
        episodes = list(read_episodes(writer.root))
        assert len(episodes) == 2
        assert all(episode["obs/extra"].shape == (2, 2) for episode in episodes)


@pytest.mark.parametrize("field", ["action", "reward"])
@pytest.mark.parametrize("dtype", [np.complex64, np.bool_])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
def test_nonreal_transition_payload_is_rejected_by_writer_and_reader(
    tmp_path, field, dtype, episode_format
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        with EpisodeRecorder(
            tmp_path / "run", env, episode_format=episode_format
        ) as writer:
            observation = env.observe()
            result = env.step(np.zeros((1, 2)))
            original = (
                result.reward if field == "reward" else result.info["executed_action"]
            )
            if field == "reward":
                result.reward = original.astype(dtype)
            else:
                result.info["executed_action"] = original.astype(dtype)
            with pytest.raises(ValueError, match="Recorded"):
                writer.append(observation, result)
            assert writer.failure is None and not writer.pending
            if field == "reward":
                result.reward = original
            else:
                result.info["executed_action"] = original
            writer.append(observation, result)
    path = next(writer.root.glob("*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays[field] = arrays[field].astype(dtype)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Invalid action/reward"):
        list(read_episodes(writer.root))


def test_proprio_shape_must_match_the_recorded_environment(tmp_path):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        with EpisodeRecorder(tmp_path / "run", env) as writer:
            observation = env.observe()
            result = env.step(np.zeros((1, 2)))
            for batch in (observation, result.observation):
                batch["proprio"] = batch["proprio"][:, :1]
            with pytest.raises(ValueError, match="proprio layout"):
                writer.append(observation, result)
            assert not writer.pending
