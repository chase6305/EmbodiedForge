"""Duplicate identities are rejected before enqueue and before publication."""

import json

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
def test_published_identity_rejected_before_first_row_copy(
    tmp_path, workers, episode_format
):
    with VectorEnv(Config(num_envs=1, max_steps=2)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=workers, episode_format=episode_format
        )
        with writer:
            for _ in range(2):
                writer.append(env.observe(), env.step(np.zeros((1, 2))))
            if workers:
                writer._collect_writes(wait_for=(0, 0))
            path = writer.root / "env00000-episode00000000.npz"
            original = path.read_bytes()
            env.reset(seed=env.seed)
            observation = env.observe()
            result = env.step(np.zeros((1, 2)))
            with pytest.raises(ValueError, match="Duplicate episode ID"):
                writer.append(observation, result)
            assert not writer.pending
            assert writer.buffered_bytes == 0
            assert writer.failure is None
            assert path.read_bytes() == original
        assert len(list(read_episodes(writer.root))) == 1


@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
def test_publication_rechecks_an_archive_created_during_recording(
    tmp_path, workers, episode_format
):
    with VectorEnv(Config(num_envs=1, max_steps=2)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=workers, episode_format=episode_format
        )
        writer.append(env.observe(), env.step(np.zeros((1, 2))))
        path = writer.root / "env00000-episode00000000.npz"
        path.write_bytes(b"existing archive must never be replaced")
        if workers:
            writer.append(env.observe(), env.step(np.zeros((1, 2))))
            with pytest.raises(ValueError, match="Duplicate episode ID"):
                writer.close()
        else:
            with pytest.raises(ValueError, match="Duplicate episode ID"):
                writer.append(env.observe(), env.step(np.zeros((1, 2))))
            writer.close()
        assert path.read_bytes() == b"existing archive must never be replaced"
        assert not path.with_suffix(".json").exists()
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["completed_episodes"] == 0
        assert manifest["incomplete_episodes"] == [
            {"env_id": 0, "episode_id": 0, "steps": 2}
        ]
