"""Measure offline episode/window/batch reads against an existing dataset."""

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge.batching import BatchOptions, window_batches
from embodiedforge.data import EpisodeReadOptions, action_windows, read_episodes


def measure(directory, *, mode, history, horizon, batch_size, shuffle_buffer, keys):
    options = EpisodeReadOptions(observation_keys=keys)

    def samples():
        if mode == "episodes":
            for episode in read_episodes(directory, options=options):
                yield len(episode["action"]), episode, ()
        elif mode == "windows":
            for window in action_windows(directory, history, horizon, options=options):
                yield (
                    1,
                    {
                        "action": window["action"],
                        **{
                            f"obs/{key}": value
                            for key, value in window["observation"].items()
                        },
                    },
                    (window["instruction"],),
                )
        else:
            for batch in window_batches(
                directory,
                history,
                horizon,
                options=options,
                batching=BatchOptions(
                    batch_size=batch_size, shuffle_buffer=shuffle_buffer
                ),
            ):
                yield (
                    len(batch.action),
                    {
                        "action": batch.action,
                        **{
                            f"obs/{key}": value
                            for key, value in batch.observation.items()
                        },
                    },
                    batch.instruction,
                )

    # Time without allocation tracing or checksumming in the hot path.
    started = time.perf_counter()
    count = sum(size for size, _, _ in samples())
    seconds = time.perf_counter() - started
    tracemalloc.start()
    for _ in samples():
        pass
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.sha256()
    for _, arrays, instructions in samples():
        for key in sorted(arrays):
            value = arrays[key]
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(np.ascontiguousarray(value).data)
        digest.update(json.dumps(instructions, ensure_ascii=False).encode())
    return {
        "mode": mode,
        "samples": count,
        "seconds": seconds,
        "peak_traced_bytes": peak,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("episodes", "windows", "batches"), required=True
    )
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--observations", nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.history, args.horizon, args.batch_size) <= 0 or args.shuffle_buffer < 0:
        parser.error(
            "Window/batch sizes must be positive and shuffle-buffer nonnegative"
        )
    report = {
        "scope": "CPU NPZ decode, validation and owned outputs; cached filesystem; timing excludes hash/tracemalloc; traced peak includes NumPy allocations, not total process RSS",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **measure(
            args.dataset,
            mode=args.mode,
            history=args.history,
            horizon=args.horizon,
            batch_size=args.batch_size,
            shuffle_buffer=args.shuffle_buffer,
            keys=None if args.observations is None else tuple(args.observations),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
