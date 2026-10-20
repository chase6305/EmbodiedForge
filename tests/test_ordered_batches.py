"""Ordered chunks preserve alignment, ownership, tails and original numeric dtypes."""

import weakref

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.batching import BatchOptions, window_batches
from embodiedforge.data import EpisodeRecorder, read_episodes


@pytest.fixture(params=["transitions", "compact"])
def dataset(tmp_path, request):
    root = tmp_path / "run"
    with VectorEnv(
        Config(
            num_envs=2,
            max_steps=7,
            render="raster",
            channels=("rgb", "depth"),
            image_size=8,
        )
    ) as env:
        with EpisodeRecorder(root, env, episode_format=request.param) as writer:
            for _ in range(3):
                # Keep trajectories away from early success so every length is 7.
                env.task.target[:] = 1.2
                env.sensors.sample(
                    np.arange(2), env.task.scene_update(env.physics.snapshot())
                )
                for step in range(7):
                    writer.append(env.observe(), env.step(np.full((2, 2), step * 0.02)))
                env.reset()
    # Valid nonnative numeric arrays are accepted by the episode schema. Add a
    # legitimate observation named action, which must not collide with actions.
    for path in root.glob("*.npz"):
        with np.load(path) as archive:
            arrays = {key: archive[key] for key in archive.files}
        prefixes = ("state/",) if request.param == "compact" else ("obs/", "next/")
        for prefix in prefixes:
            arrays[prefix + "action"] = arrays[prefix + "proprio"][:, :3].copy()
        arrays = {
            key: value.astype(value.dtype.newbyteorder(">"))
            if value.dtype.kind == "f"
            else value
            for key, value in arrays.items()
        }
        np.savez_compressed(path, **arrays)
    return root


def reference_windows(root, history, horizon):
    result = []
    for episode in read_episodes(root):
        for t in range(history - 1, len(episode["action"]) - horizon + 1):
            result.append(
                (
                    {
                        key[4:]: value[t - history + 1 : t + 1]
                        for key, value in episode.items()
                        if key.startswith("obs/")
                    },
                    episode["action"][t : t + horizon],
                )
            )
    return result


@pytest.mark.parametrize("history,horizon", [(1, 1), (2, 3), (5, 1), (2, 7)])
@pytest.mark.parametrize("size", [1, 3, 8, 64])
@pytest.mark.parametrize("drop_last", [False, True])
def test_ordered_batches_match_independent_transition_slices(
    dataset, history, horizon, size, drop_last
):
    expected = reference_windows(dataset, history, horizon)
    if drop_last:
        expected = expected[: len(expected) // size * size]
    actual = list(
        window_batches(
            dataset,
            history,
            horizon,
            batching=BatchOptions(batch_size=size, drop_last=drop_last),
        )
    )
    assert sum(len(batch.action) for batch in actual) == len(expected)
    offset = 0
    for batch in actual:
        count = len(batch.action)
        samples = expected[offset : offset + count]
        np.testing.assert_array_equal(
            batch.action, np.stack([sample[1] for sample in samples])
        )
        assert batch.action.dtype == samples[0][1].dtype.newbyteorder("=")
        assert batch.action.flags.owndata
        for key, value in batch.observation.items():
            np.testing.assert_array_equal(
                value, np.stack([sample[0][key] for sample in samples])
            )
            assert value.dtype == samples[0][0][key].dtype.newbyteorder("=")
            assert value.flags.owndata
        assert len(batch.instruction) == count
        assert all(isinstance(text, str) for text in batch.instruction)
        offset += count


def test_mutating_an_ordered_batch_cannot_change_next_batch_or_dataset(dataset):
    expected = reference_windows(dataset, 2, 3)
    iterator = window_batches(dataset, 2, 3, batching=BatchOptions(batch_size=2))
    first = next(iterator)
    first.action[:] = 99
    for value in first.observation.values():
        value[:] = 99
    second = next(iterator)
    iterator.close()
    np.testing.assert_array_equal(
        second.action, np.stack([sample[1] for sample in expected[2:4]])
    )
    for key, value in second.observation.items():
        np.testing.assert_array_equal(
            value, np.stack([sample[0][key] for sample in expected[2:4]])
        )
    reread = reference_windows(dataset, 2, 3)
    for source, target in zip(expected, reread, strict=True):
        np.testing.assert_array_equal(source[1], target[1])


@pytest.mark.parametrize("size", [2, 64])
def test_cross_episode_tail_does_not_retain_whole_backing_arrays(
    dataset, monkeypatch, size
):
    loaded = []
    load, getitem = np.load, np.lib.npyio.NpzFile.__getitem__

    def read(archive, key):
        value = getitem(archive, key)
        if key in ("state/rgb", "obs/rgb", "next/rgb"):
            loaded.append(weakref.ref(value))
        return value

    def open_archive(path, **kwargs):
        assert all(array() is None for array in loaded)
        return load(path, **kwargs)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", read)
    monkeypatch.setattr(np, "load", open_archive)
    # One observation and a whole-episode action horizon expose retained views
    # for both completed crossing batches and an oversized final tail.
    batches = list(
        window_batches(
            dataset, history=1, horizon=7, batching=BatchOptions(batch_size=size)
        )
    )
    assert len(batches) == (6 + size - 1) // size
    assert sum(len(batch.action) for batch in batches) == 6
    assert all(array() is None for array in loaded)


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("drop_last", [False, True])
def test_cross_episode_batches_own_strided_readonly_inputs_and_align_instructions(
    monkeypatch, byte_order, drop_last
):
    from embodiedforge import batching as module

    episodes, expected, sources = [], [], []
    for index, count in enumerate([1, 3, 1, 2, 1, 3]):
        # Include overlapping windows, non-contiguous reversed strides and an
        # observation named action. Source arrays must never be writable aliases
        # of the output, even for the episode that completes a batch.
        source = (np.arange((count + 1) * 3) + 100 * index).astype(byte_order + "f4")
        source = source.reshape(-1, 3)[:, ::-1]
        windows = np.moveaxis(
            np.lib.stride_tricks.sliding_window_view(source, 2, axis=0), -1, 1
        )
        observation = {"action": windows, "proprio": windows[:, :, :2]}
        action = windows[:, :1]
        instruction = f"episode {index}"
        episodes.append((observation, action, instruction))
        sources.append(source.copy())
        expected.extend(
            (
                {key: value[row] for key, value in observation.items()},
                action[row],
                instruction,
            )
            for row in range(count)
        )
    monkeypatch.setattr(module, "_episode_windows", lambda *a, **kw: iter(episodes))
    batches = list(
        window_batches(
            "unused", batching=BatchOptions(batch_size=4, drop_last=drop_last)
        )
    )
    if drop_last:
        expected = expected[: len(expected) // 4 * 4]
    assert sum(len(batch.action) for batch in batches) == len(expected)
    offset = 0
    for batch in batches:
        samples = expected[offset : offset + len(batch.action)]
        assert batch.instruction == tuple(sample[2] for sample in samples)
        for key, value in {"control": batch.action, **batch.observation}.items():
            reference = np.stack(
                [
                    sample[1] if key == "control" else sample[0][key]
                    for sample in samples
                ]
            )
            np.testing.assert_array_equal(value, reference)
            assert value.flags.owndata and value.flags.c_contiguous
            assert value.dtype.isnative
            assert not any(np.shares_memory(value, episode[1]) for episode in episodes)
            value[:] = -1
        offset += len(batch.action)
    for episode, source in zip(episodes, sources, strict=True):
        np.testing.assert_array_equal(episode[1][:, 0], source[:-1])


@pytest.mark.parametrize("lengths", [(1, 3), (1, 1, 2)])
def test_cross_episode_batch_rejects_changed_byte_order_before_casting(
    monkeypatch, lengths
):
    from embodiedforge import batching as module

    episodes = [
        ({"proprio": np.zeros((count, 1, 2))}, np.zeros((count, 1, 2), ">f4"), "a")
        for count in lengths
    ]
    observation, action, instruction = episodes[-1]
    episodes[-1] = observation, action.astype("<f4"), instruction
    monkeypatch.setattr(module, "_episode_windows", lambda *a, **kw: iter(episodes))
    with pytest.raises(ValueError, match="different shapes/dtypes for action"):
        next(window_batches("unused", batching=BatchOptions(batch_size=4)))


@pytest.mark.parametrize("lengths", [(1, 3), (1, 1, 2)])
def test_cross_episode_batch_rejects_nonstring_instruction(monkeypatch, lengths):
    from embodiedforge import batching as module

    episodes = [
        ({"proprio": np.zeros((count, 1, 2))}, np.zeros((count, 1, 2)), "a")
        for count in lengths
    ]
    observation, action, _ = episodes[-1]
    episodes[-1] = observation, action, 42
    monkeypatch.setattr(module, "_episode_windows", lambda *a, **kw: iter(episodes))
    with pytest.raises(ValueError, match="instructions must be strings"):
        next(window_batches("unused", batching=BatchOptions(batch_size=4)))


@pytest.mark.parametrize("shuffle_buffer", [0, 5])
def test_batches_keep_native_byte_order_with_shuffling_and_tails(
    dataset, shuffle_buffer
):
    batches = list(
        window_batches(
            dataset, batching=BatchOptions(batch_size=5, shuffle_buffer=shuffle_buffer)
        )
    )
    assert batches
    for batch in batches:
        assert batch.action.dtype == np.dtype("f4")
        assert batch.observation["proprio"].dtype == np.dtype("f4")
        assert batch.observation["time"].dtype == np.dtype("f8")
