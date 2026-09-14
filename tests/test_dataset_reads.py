"""Read partitions once and avoid decompressing unrequested camera payloads."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import (
    EpisodeReadOptions,
    EpisodeRecorder,
    action_windows,
    read_episodes,
)
from embodiedforge.rollout import run_rollout


@pytest.fixture
def dataset(tmp_path):
    class ZeroPolicy:
        def act(self, observation):
            return np.zeros((len(observation["env_id"]), 1, 2), dtype=np.float32)

    root = tmp_path / "run"
    with VectorEnv(
        Config(num_envs=4, max_steps=3, render="raster", channels=("rgb", "depth"))
    ) as env:
        with EpisodeRecorder(root, env) as writer:
            run_rollout(env, ZeroPolicy(), steps=6, writer=writer)
    return root


def identity(episode):
    return (int(episode["obs/env_id"][0]), int(episode["obs/episode_id"][0]))


def test_partitions_cover_each_episode_once_without_opening_other_archives(
    dataset, monkeypatch
):
    (dataset / "000-unpublished.npz").write_bytes(b"incomplete archive")
    complete = list(read_episodes(dataset))
    expected = [identity(episode) for episode in complete]
    opened = []
    original = np.load

    def load(path, **kwargs):
        opened.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(np, "load", load)
    partitions = []
    for index in range(3):
        rows = list(
            read_episodes(
                dataset, options=EpisodeReadOptions(shard_index=index, num_shards=3)
            )
        )
        ids = [identity(row) for row in rows]
        assert ids == expected[index::3]
        partitions.extend(ids)
    assert sorted(partitions) == sorted(expected)
    assert len(opened) == len(set(opened)) == len(expected)


def test_projection_skips_camera_decompression_and_preserves_state(
    dataset, monkeypatch
):
    expected = next(read_episodes(dataset))
    reads = []
    cls = np.lib.npyio.NpzFile
    original = cls.__getitem__

    def getitem(self, key):
        reads.append(key)
        assert key not in {"obs/rgb", "next/rgb", "obs/depth", "next/depth"}
        return original(self, key)

    monkeypatch.setattr(cls, "__getitem__", getitem)
    actual = next(
        read_episodes(dataset, options=EpisodeReadOptions(observation_keys=()))
    )
    assert "obs/rgb" not in actual
    assert "next/depth" not in actual
    assert "obs/proprio" in reads
    for key, value in actual.items():
        np.testing.assert_array_equal(value, expected[key])


def test_projected_windows_match_full_windows_and_own_arrays(dataset):
    full = list(action_windows(dataset, history=2, horizon=2))
    projected = list(
        action_windows(
            dataset,
            history=2,
            horizon=2,
            options=EpisodeReadOptions(observation_keys=("rgb",)),
        )
    )
    assert len(full) == len(projected) > 0
    for source, target in zip(full, projected, strict=True):
        np.testing.assert_array_equal(source["action"], target["action"])
        for name, value in target["observation"].items():
            np.testing.assert_array_equal(source["observation"][name], value)
        assert "depth" not in target["observation"]
    projected[0]["observation"]["rgb"][:] = 0
    assert np.any(full[0]["observation"]["rgb"])


def test_unknown_observation_rejected(dataset):
    with pytest.raises(ValueError, match="Missing requested observations"):
        next(
            read_episodes(
                dataset, options=EpisodeReadOptions(observation_keys=("wrist",))
            )
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_shards": 0},
        {"num_shards": True},
        {"shard_index": -1},
        {"shard_index": 1},
        {"shard_index": 0.5},
        {"observation_keys": "rgb"},
        {"observation_keys": ("rgb", "rgb")},
        {"observation_keys": ("obs/rgb",)},
    ],
)
def test_invalid_read_options(kwargs):
    with pytest.raises(ValueError):
        EpisodeReadOptions(**kwargs)


@pytest.mark.parametrize("field", ["proprio", "rgb"])
def test_projection_validation_scope_is_explicit(dataset, field):
    path = sorted(dataset.glob("*.npz"))[0]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[f"next/{field}"][0] = 0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="continuity broken"):
        list(read_episodes(dataset))
    options = EpisodeReadOptions(observation_keys=())
    if field == "proprio":
        with pytest.raises(ValueError, match="continuity broken"):
            list(read_episodes(dataset, options=options))
    else:
        assert list(read_episodes(dataset, options=options))


def test_batches_reproducible_owned_and_epoch_changes_order(dataset):
    from embodiedforge.batching import BatchOptions, window_batches

    def collect(epoch):
        return list(
            window_batches(
                dataset,
                options=EpisodeReadOptions(observation_keys=()),
                batching=BatchOptions(
                    batch_size=5, shuffle_buffer=8, seed=7, epoch=epoch
                ),
            )
        )

    def identities(batches):
        return [
            tuple(row)
            for batch in batches
            for row in np.stack(
                [
                    batch.observation["env_id"][:, -1],
                    batch.observation["episode_id"][:, -1],
                    batch.observation["step_id"][:, -1],
                ],
                axis=1,
            )
        ]

    first, repeat, following = collect(0), collect(0), collect(1)
    assert identities(first) == identities(repeat)
    assert identities(first) != identities(following)
    assert sorted(identities(first)) == sorted(identities(following))
    windows = list(action_windows(dataset))
    assert sum(len(batch.action) for batch in first) == len(windows)
    assert len(set(identities(first))) == len(windows)
    first[0].action[:] = 99
    assert not (repeat[0].action == 99).any()
    assert len(first[0].instruction) == 5


def test_ordered_batches_and_drop_last(dataset):
    from embodiedforge.batching import BatchOptions, window_batches

    windows = list(action_windows(dataset, history=2, horizon=2))
    batches = list(
        window_batches(
            dataset, history=2, horizon=2, batching=BatchOptions(batch_size=3)
        )
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.action for batch in batches]),
        np.stack([sample["action"] for sample in windows]),
    )
    for name in windows[0]["observation"]:
        np.testing.assert_array_equal(
            np.concatenate([batch.observation[name] for batch in batches]),
            np.stack([sample["observation"][name] for sample in windows]),
        )
    dropped = list(
        window_batches(
            dataset,
            history=2,
            horizon=2,
            batching=BatchOptions(batch_size=3, drop_last=True),
        )
    )
    assert all(len(batch.action) == 3 for batch in dropped)
    assert sum(len(batch.action) for batch in dropped) == len(windows) // 3 * 3


def test_batch_shuffle_consumes_bounded_prefix(monkeypatch):
    from embodiedforge import batching as module

    consumed = []

    def windows(*args, **kwargs):
        for index in range(1000):
            consumed.append(index)
            yield {
                "observation": {"proprio": np.zeros((1, 2))},
                "action": np.zeros((1, 2)),
                "instruction": "hold",
            }

    monkeypatch.setattr(module, "action_windows", windows)
    iterator = module.window_batches(
        "unused", batching=module.BatchOptions(batch_size=3, shuffle_buffer=4)
    )
    assert len(next(iterator).action) == 3
    assert len(consumed) == 7
    iterator.close()


@pytest.mark.parametrize("fault", ["shape", "dtype", "schema"])
def test_batch_rejects_incompatible_samples(monkeypatch, fault):
    from embodiedforge import batching as module

    first = {
        "observation": {"proprio": np.zeros((1, 2))},
        "action": np.zeros((1, 2), dtype=np.float32),
        "instruction": "hold",
    }
    second = {**first}
    if fault == "shape":
        second["action"] = np.zeros((2, 2), dtype=np.float32)
    elif fault == "dtype":
        second["action"] = first["action"].astype(np.float64)
    else:
        second["observation"] = {"other": np.zeros((1, 2))}
    monkeypatch.setattr(
        module, "action_windows", lambda *a, **kw: iter([first, second])
    )
    with pytest.raises(ValueError, match="Cannot batch different"):
        next(
            module.window_batches("unused", batching=module.BatchOptions(batch_size=2))
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": True},
        {"shuffle_buffer": -1},
        {"seed": -1},
        {"epoch": 1.5},
        {"drop_last": 1},
    ],
)
def test_invalid_batch_options(kwargs):
    from embodiedforge.batching import BatchOptions

    with pytest.raises(ValueError):
        BatchOptions(**kwargs)
