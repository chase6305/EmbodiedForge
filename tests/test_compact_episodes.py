"""Compact storage preserves public transitions and bounds unique owned payloads."""

import json

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import (
    EpisodeReadOptions,
    EpisodeRecorder,
    action_windows,
    read_episodes,
)


def record(root, episode_format, *, steps=5):
    with VectorEnv(
        Config(
            num_envs=2,
            max_steps=steps,
            render="raster",
            channels=("rgb", "depth"),
            image_size=16,
        )
    ) as env:
        with EpisodeRecorder(root, env, episode_format=episode_format) as writer:
            for _ in range(2):
                for step in range(steps):
                    obs = env.observe()
                    result = env.step(np.full((2, 2), 0.1 * step))
                    writer.append(obs, result)
                env.reset()


@pytest.mark.parametrize("steps", [1, 5])
def test_compact_matches_legacy_transitions_and_windows(tmp_path, steps):
    legacy, compact = tmp_path / "legacy", tmp_path / "compact"
    record(legacy, "transitions", steps=steps)
    record(compact, "compact", steps=steps)
    for expected, actual in zip(
        read_episodes(legacy), read_episodes(compact), strict=True
    ):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value)
        for name in ("proprio", "rgb", "depth"):
            assert not np.shares_memory(actual[f"obs/{name}"], actual[f"next/{name}"])
    for source, target in zip(
        action_windows(legacy), action_windows(compact), strict=True
    ):
        np.testing.assert_array_equal(source["action"], target["action"])
        assert source["instruction"] == target["instruction"]
        for key, value in source["observation"].items():
            np.testing.assert_array_equal(target["observation"][key], value)
    archive = sorted(compact.glob("*.npz"))[0]
    assert json.loads(archive.with_suffix(".json").read_text())["schema_version"] == 2
    with np.load(archive) as data:
        assert data["state/rgb"].shape[0] == steps + 1
        assert not any(key.startswith(("obs/", "next/")) for key in data.files)


def test_compact_shares_only_owned_continuous_observations_and_releases_budget(
    tmp_path,
):
    with VectorEnv(Config(num_envs=1, max_steps=3)) as env:
        obs = env.observe()
        first = env.step(np.zeros((1, 2)))
        first_bytes = sum(
            v.nbytes
            for v in (
                *obs.values(),
                *first.observation.values(),
                first.info["executed_action"],
                first.reward,
                first.terminated,
                first.truncated,
            )
        )
        continuation_bytes = first_bytes - sum(v.nbytes for v in obs.values())
        with EpisodeRecorder(
            tmp_path / "compact",
            env,
            episode_format="compact",
            max_buffer_bytes=first_bytes + 2 * continuation_bytes,
        ) as writer:
            writer.append(obs, first)
            assert writer.buffered_bytes == first_bytes
            obs = env.observe()
            second = env.step(np.zeros((1, 2)))
            changed = {k: v.copy() for k, v in obs.items()}
            changed["proprio"][:] = 99
            with pytest.raises(ValueError, match="continuity"):
                writer.append(changed, second)
            assert writer.buffered_bytes == first_bytes
            writer.append(obs, second)
            rows = writer.pending[(0, 0)]
            assert rows[1]["obs/proprio"] is rows[0]["next/proprio"]
            assert writer.buffered_bytes == first_bytes + continuation_bytes
            assert not np.shares_memory(rows[1]["obs/proprio"], obs["proprio"])
            obs["proprio"][:] = 99
            writer.append(env.observe(), env.step(np.zeros((1, 2))))
            assert writer.buffered_bytes == 0
            assert not writer.pending
        assert len(list(read_episodes(writer.root))) == 1


def test_compact_projection_does_not_decompress_images(tmp_path, monkeypatch):
    record(tmp_path / "compact", "compact")
    original = np.lib.npyio.NpzFile.__getitem__

    def read(archive, key):
        assert key not in ("state/rgb", "state/depth")
        return original(archive, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read)
    episodes = list(
        read_episodes(
            tmp_path / "compact", options=EpisodeReadOptions(observation_keys=())
        )
    )
    assert len(episodes) == 4
    assert "obs/proprio" in episodes[0]
    assert "obs/rgb" not in episodes[0]


@pytest.mark.parametrize("damage", ["length", "nonfinite", "identity", "field", "done"])
def test_compact_reader_rejects_corrupt_payloads(tmp_path, damage):
    root = tmp_path / "compact"
    record(root, "compact")
    path = sorted(root.glob("*.npz"))[0]
    with np.load(path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if damage == "length":
        arrays["state/proprio"] = arrays["state/proprio"][:-1]
    elif damage == "nonfinite":
        arrays["state/proprio"][0, 0] = np.nan
    elif damage == "identity":
        arrays["state/env_id"][-1] = 100
    elif damage == "field":
        arrays["obs/proprio"] = arrays["state/proprio"][:-1]
    else:
        arrays["truncated"][-1] = False
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        list(read_episodes(root))


@pytest.mark.parametrize("episode_format", [None, "unknown", [], 2])
def test_invalid_format_does_not_create_output(tmp_path, episode_format):
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(ValueError, match="episode_format"):
            EpisodeRecorder(tmp_path / "invalid", env, episode_format=episode_format)
    assert not (tmp_path / "invalid").exists()
