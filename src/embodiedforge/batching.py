"""Bounded, reproducible offline batches using NumPy and the episode reader."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import Array, Observation
from .data import EpisodeReadOptions, action_windows


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


def _shuffle(
    samples: Iterator[dict],
    capacity: int,
    rng: np.random.Generator,
) -> Iterator[dict]:
    """Hold at most capacity buffered samples plus the incoming sample."""
    if capacity == 0:
        yield from samples
        return
    buffer = []
    for sample in samples:
        if len(buffer) < capacity:
            buffer.append(sample)
            continue
        index = int(rng.integers(len(buffer)))
        selected = buffer[index]
        buffer[index] = sample
        yield selected
    while buffer:
        index = int(rng.integers(len(buffer)))
        selected = buffer[index]
        buffer[index] = buffer[-1]
        buffer.pop()
        yield selected


def _collate(samples: list[dict]) -> WindowBatch:
    """Reject incompatible layouts before stacking, including silent dtype casts."""
    first = samples[0]
    for sample in samples:
        if sample["observation"].keys() != first["observation"].keys():
            raise ValueError("Cannot batch different observation schemas")
        for name, value in {
            "action": sample["action"],
            **{f"obs/{key}": array for key, array in sample["observation"].items()},
        }.items():
            reference = (
                first["action"] if name == "action" else first["observation"][name[4:]]
            )
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise ValueError(f"Cannot batch different shapes/dtypes for {name}")
        if not isinstance(sample["instruction"], str):
            raise ValueError("Batch instructions must be strings")
    return WindowBatch(
        observation={
            key: np.stack([sample["observation"][key] for sample in samples])
            for key in first["observation"]
        },
        action=np.stack([sample["action"] for sample in samples]),
        instruction=tuple(sample["instruction"] for sample in samples),
    )


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
    options = options or EpisodeReadOptions()
    batching = batching or BatchOptions()
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
    samples = action_windows(directory, history, horizon, options=options)
    pending = []
    for sample in _shuffle(samples, batching.shuffle_buffer, rng):
        pending.append(sample)
        if len(pending) == batching.batch_size:
            yield _collate(pending)
            pending.clear()
    if pending and not batching.drop_last:
        yield _collate(pending)
