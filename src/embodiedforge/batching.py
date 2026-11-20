"""Bounded, reproducible offline batches using NumPy and the episode reader."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import Array, Observation
from .data import EpisodeReadOptions, _episode_windows, _read_options, _window_sample


@dataclass(frozen=True)
class BatchOptions:
    """Batch size and optional bounded shuffle; epoch changes the local RNG stream.

    shuffle_buffer=0 preserves dataset order. A positive buffer gives a streaming
    shuffle, not a uniform global permutation unless it holds the entire input.
    """

    batch_size: int = 32
    shuffle_buffer: int = 0
    seed: int = 42
    epoch: int = 0
    drop_last: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "shuffle_buffer", "seed", "epoch"):
            value = getattr(self, name)
            minimum = 1 if name == "batch_size" else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if type(self.drop_last) is not bool:
            raise ValueError("drop_last must be a boolean")


@dataclass(frozen=True)
class WindowBatch:
    """Owned arrays: observations [B,history,...], action [B,horizon,A].

    Instructions remain one string per sample for the caller's tokenizer.
    A frozen container prevents field rebinding; its arrays remain mutable.
    """

    observation: Observation
    action: Array
    instruction: tuple[str, ...]


def _collate(samples: list[dict]) -> WindowBatch:
    """Reject incompatible layouts before stacking, including silent dtype casts."""
    first = samples[0]
    first_observation, first_action = first["observation"], first["action"]
    for sample in samples:
        observation, action = sample["observation"], sample["action"]
        if observation.keys() != first_observation.keys():
            raise ValueError("Cannot batch different observation schemas")
        if action.shape != first_action.shape or action.dtype != first_action.dtype:
            raise ValueError("Cannot batch different shapes/dtypes for action")
        for key, value in observation.items():
            reference = first_observation[key]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"Cannot batch different shapes/dtypes for obs/{key}")
        if not isinstance(sample["instruction"], str):
            raise ValueError("Batch instructions must be strings")
    # The layouts already match. Fill the leading batch axis directly using
    # the native output byte order that np.stack used for these numeric arrays.
    return WindowBatch(
        observation={
            key: np.array(
                [sample["observation"][key] for sample in samples],
                dtype=value.dtype.newbyteorder("="),
                order="C",
            )
            for key, value in first_observation.items()
        },
        action=np.array(
            [sample["action"] for sample in samples],
            dtype=first_action.dtype.newbyteorder("="),
            order="C",
        ),
        instruction=tuple(sample["instruction"] for sample in samples),
    )


def _collate_episode_rows(
    samples: list[int | dict], observation: Observation, action: Array, instruction: str
) -> WindowBatch:
    if all(type(row) is int for row in samples):

        def gather(value):
            return np.array(
                [value[index] for index in samples],
                dtype=value.dtype.newbyteorder("="),
                order="C",
            )

        return WindowBatch(
            {key: gather(value) for key, value in observation.items()},
            gather(action),
            (instruction,) * len(samples),
        )

    return _collate(
        [
            _window_sample(observation, action, instruction, row, copy=False)
            if type(row) is int
            else row
            for row in samples
        ]
    )


def _shuffled_batches(
    episodes: Iterator[tuple[Observation, Array, str]],
    capacity: int,
    size: int,
    drop_last: bool,
    rng: np.random.Generator,
) -> Iterator[WindowBatch]:
    """Collate current rows directly and copy any survivors before advancing."""
    # Integers refer to rows of the current episode. Samples from earlier
    # episodes already own their arrays and occupy the same buffer slots.
    buffer: list[int | dict] = []
    pending: list[int | dict] = []
    for observation, action, instruction in episodes:
        if not isinstance(instruction, str):
            raise ValueError("Batch instructions must be strings")
        if len(buffer) + len(action) <= capacity:
            buffer.extend(
                _window_sample(observation, action, instruction, row, copy=True)
                for row in range(len(action))
            )
            del observation, action
            continue
        borrowed_slots = set()
        pending_start = len(pending)
        for row in range(len(action)):
            if len(buffer) < capacity:
                borrowed_slots.add(len(buffer))
                buffer.append(row)
                continue
            slot = int(rng.integers(len(buffer)))
            pending.append(buffer[slot])
            buffer[slot] = row
            borrowed_slots.add(slot)
            if len(pending) == size:
                yield _collate_episode_rows(pending, observation, action, instruction)
                pending.clear()
                pending_start = 0

        for slot in borrowed_slots:
            buffer[slot] = _window_sample(
                observation, action, instruction, buffer[slot], copy=True
            )
        # An unfilled batch can also contain rows from the current episode.
        for slot in range(pending_start, len(pending)):
            row = pending[slot]
            if type(row) is int:
                pending[slot] = _window_sample(
                    observation, action, instruction, row, copy=True
                )
        del observation, action

    while buffer:
        slot = int(rng.integers(len(buffer)))
        pending.append(buffer[slot])
        buffer[slot] = buffer[-1]
        buffer.pop()
        if len(pending) == size:
            yield _collate(pending)
            pending.clear()
    if pending and not drop_last:
        yield _collate(pending)


@dataclass(frozen=True)
class _EpisodeChunk:
    observation: Observation
    action: Array
    instruction: str


def _owned_chunk(observation, action, instruction, start, count) -> WindowBatch:
    def owned(value):
        return np.array(
            value[start : start + count],
            dtype=value.dtype.newbyteorder("="),
            copy=True,
            order="C",
        )

    return WindowBatch(
        {key: owned(value) for key, value in observation.items()},
        owned(action),
        (instruction,) * count,
    )


def _slice_chunk(
    observation, action, instruction, start, count, *, copy
) -> _EpisodeChunk:
    selection = slice(start, start + count)

    def select(value):
        part = value[selection]
        return part.copy() if copy else part

    return _EpisodeChunk(
        {key: select(value) for key, value in observation.items()},
        select(action),
        instruction,
    )


def _join_chunks(chunks: list[_EpisodeChunk]) -> WindowBatch:
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk.observation.keys() != first.observation.keys():
            raise ValueError("Cannot batch different observation schemas")
        if (
            chunk.action.shape[1:] != first.action.shape[1:]
            or chunk.action.dtype != first.action.dtype
        ):
            raise ValueError("Cannot batch different shapes/dtypes for action")
        for key, value in chunk.observation.items():
            reference = first.observation[key]
            if value.shape[1:] != reference.shape[1:] or value.dtype != reference.dtype:
                raise ValueError(f"Cannot batch different shapes/dtypes for obs/{key}")
    if len(chunks) == 1:

        def native(value):
            return (
                value
                if value.dtype.isnative
                else value.astype(value.dtype.newbyteorder("="))
            )

        return WindowBatch(
            {key: native(value) for key, value in first.observation.items()},
            native(first.action),
            (first.instruction,) * len(first.action),
        )
    size = sum(len(chunk.action) for chunk in chunks)

    def join(values):
        first_value = values[0]
        # A chunk borrowed from the current episode can have arbitrary strides.
        output = np.empty(
            (size, *first_value.shape[1:]), dtype=first_value.dtype.newbyteorder("=")
        )
        return np.concatenate(values, out=output)

    return WindowBatch(
        {
            key: join([chunk.observation[key] for chunk in chunks])
            for key in first.observation
        },
        join([chunk.action for chunk in chunks]),
        tuple(chunk.instruction for chunk in chunks for _ in range(len(chunk.action))),
    )


def _ordered_batches(episodes, size, drop_last) -> Iterator[WindowBatch]:
    """Copy pending tails before advancing and join directly from the last episode."""
    pending = []
    filled = 0
    for observation, action, instruction in episodes:
        if not isinstance(instruction, str):
            raise ValueError("Batch instructions must be strings")
        start, count = 0, len(action)
        if filled:
            take = min(size - filled, count)
            filled += take
            # Only a tail that outlives this episode needs a separate copy.
            pending.append(
                _slice_chunk(
                    observation, action, instruction, start, take, copy=filled < size
                )
            )
            start += take
            if filled == size:
                yield _join_chunks(pending)
                pending.clear()
                filled = 0
        while count - start >= size:
            yield _owned_chunk(observation, action, instruction, start, size)
            start += size
        if start < count:
            pending.append(
                _slice_chunk(
                    observation,
                    action,
                    instruction,
                    start,
                    count - start,
                    copy=True,
                )
            )
            filled += count - start
        # Pending tails contain copies; never retain old episode backing arrays
        # while fetching the next episode, even with a very long action horizon.
        del observation, action
    if filled and not drop_last:
        yield _join_chunks(pending)


def window_batches(
    directory: str | Path,
    history: int = 1,
    horizon: int = 1,
    *,
    options: EpisodeReadOptions | None = None,
    batching: BatchOptions | None = None,
) -> Iterator[WindowBatch]:
    """Read one partition, optionally shuffle, then yield owned training batches.

    Determinism requires identical files, options, seed and epoch. drop_last is
    applied per partition. Memory includes one decoded episode, the shuffle
    buffer, pending samples and stacked output; this is not a byte/RSS limit.
    No tensors, device transfers, tokenization or padding are performed here.
    """
    options = _read_options(options)
    if batching is None:
        batching = BatchOptions()
    elif not isinstance(batching, BatchOptions):
        raise ValueError("batching must be BatchOptions")
    episodes = _episode_windows(directory, history, horizon, options=options)
    if batching.shuffle_buffer == 0:
        yield from _ordered_batches(episodes, batching.batch_size, batching.drop_last)
        return
    rng = np.random.default_rng(
        np.random.SeedSequence(
            [
                batching.seed,
                batching.epoch,
                options.shard_index,
                options.num_shards,
            ]
        )
    )
    yield from _shuffled_batches(
        episodes,
        batching.shuffle_buffer,
        batching.batch_size,
        batching.drop_last,
        rng,
    )
