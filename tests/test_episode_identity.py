"""Episode identity stays exact across JSON, byte orders and NumPy versions."""

import json

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


def write_episode(root, *, schema, dtype, identity):
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"schema_version": 1}))
    states = {
        "env_id": np.zeros(4, dtype=dtype),
        "episode_id": np.full(4, identity, dtype=dtype),
        "step_id": np.arange(4, dtype=dtype),
        "time": np.arange(4, dtype=np.float64),
        "proprio": np.zeros((4, 2), dtype=np.float32),
    }
    arrays = {
        "action": np.zeros((3, 2), dtype=np.float32),
        "reward": np.zeros(3, dtype=np.float32),
        "terminated": np.array([False, False, True]),
        "truncated": np.zeros(3, dtype=bool),
    }
    if schema == 2:
        arrays.update({f"state/{key}": value for key, value in states.items()})
    else:
        for key, value in states.items():
            arrays[f"obs/{key}"] = value[:-1]
            arrays[f"next/{key}"] = value[1:]
    np.savez(root / "episode.npz", **arrays)
    (root / "episode.json").write_text(
        json.dumps(
            {
                "schema_version": schema,
                "complete": True,
                "steps": 3,
                "env_id": 0,
                "episode_id": identity,
            }
        )
    )


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("dtype", ["<i8", ">i8", "<u8", ">u8"])
@pytest.mark.parametrize("identity", [0, 1, 2**53 + 1, 2**63 - 1])
def test_exact_identity_roundtrip(tmp_path, schema, dtype, identity):
    root = tmp_path / "episodes"
    write_episode(root, schema=schema, dtype=dtype, identity=identity)
    (episode,) = read_episodes(root)
    assert all(int(value) == identity for value in episode["obs/episode_id"])
    assert episode["obs/episode_id"].dtype == np.dtype(dtype)


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("dtype", ["<u8", ">u8"])
@pytest.mark.parametrize("identity", [2**63, 2**64 - 1])
def test_unsigned_identity_roundtrip(tmp_path, schema, dtype, identity):
    root = tmp_path / "episodes"
    write_episode(root, schema=schema, dtype=dtype, identity=identity)
    (episode,) = read_episodes(root)
    assert all(int(value) == identity for value in episode["next/episode_id"])


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize(
    "identity,marker",
    [
        (0, False),
        (1, True),
        (1, 1.0),
        (2**53 + 1, float(2**53)),
        (2**63 - 1, 2**63),
        (1, -1),
        (1, 2**64),
        (1, "1"),
    ],
)
@pytest.mark.parametrize("field", ["env_id", "episode_id"])
def test_reader_rejects_inexact_or_noninteger_marker(
    tmp_path, schema, identity, marker, field
):
    root = tmp_path / "episodes"
    write_episode(root, schema=schema, dtype="i8", identity=identity)
    path = root / "episode.json"
    metadata = json.loads(path.read_text())
    if field == "env_id":
        # Match a numeric marker for the bool/float cases; these still must fail.
        with np.load(root / "episode.npz") as archive:
            arrays = {key: archive[key] for key in archive.files}
        for key in arrays:
            if key.endswith("/env_id"):
                arrays[key][:] = identity
        np.savez(root / "episode.npz", **arrays)
    metadata[field] = marker
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="identity"):
        list(read_episodes(root))


@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("episode", [2**53, 2**63 - 2, 2**63 - 1])
def test_recorder_rejects_adjacent_unsigned_episode_without_committing(
    tmp_path, episode_format, episode
):
    with VectorEnv(Config(num_envs=1, max_steps=3)) as env:
        with EpisodeRecorder(
            tmp_path / "run", env, episode_format=episode_format
        ) as writer:
            before = env.observe()
            result = env.step(np.zeros((1, 2)))
            before["episode_id"] = np.array([episode], dtype=np.uint64)
            result.observation["episode_id"] = np.array([episode + 1], dtype=np.uint64)
            with pytest.raises(ValueError, match="advance exactly one step"):
                writer.append(before, result)
            assert writer.buffered_bytes == 0
            assert writer.pending == {}
            assert not list(writer.root.glob("*.npz"))
            # Validation failure is retryable with an exact identifier.
            result.observation["episode_id"][0] = episode
            writer.append(before, result)
            assert len(writer.pending[(0, episode)]) == 1
