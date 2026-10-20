"""Measure owned ordered batches from decoded, overlapping episode windows."""

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge.batching import _ordered_batches


def measure(args, *, trace=False, checksum=False):
    count = max(args.episode_windows)
    states = np.arange(
        (count + args.history - 1) * args.image_size**2 * 3, dtype=np.uint8
    ).reshape(-1, 1, args.image_size, args.image_size, 3)
    controls = np.arange((count + args.horizon - 1) * 2, dtype=np.float32).reshape(
        -1, 2
    )
    observation = {
        "rgb": np.moveaxis(
            np.lib.stride_tricks.sliding_window_view(states, args.history, axis=0),
            -1,
            1,
        ),
    }
    action = np.moveaxis(
        np.lib.stride_tricks.sliding_window_view(controls, args.horizon, axis=0),
        -1,
        1,
    )

    def episodes():
        # Reuse decoded backing arrays to isolate collation from NPZ decoding.
        for index in range(args.episodes):
            length = args.episode_windows[index % len(args.episode_windows)]
            yield (
                {key: value[:length] for key, value in observation.items()},
                action[:length],
                f"episode {index}",
            )

    digest = hashlib.sha256() if checksum else None
    if trace:
        tracemalloc.start()
    try:
        started = time.perf_counter()
        samples = 0
        for batch in _ordered_batches(episodes(), args.batch_size, False):
            samples += len(batch.action)
            if digest is not None:
                for key, value in sorted(
                    {"action": batch.action, **batch.observation}.items()
                ):
                    assert value.flags.owndata and value.flags.c_contiguous
                    digest.update(key.encode())
                    digest.update(str((value.shape, value.dtype.str)).encode())
                    digest.update(value.data)
                digest.update(json.dumps(batch.instruction).encode())
        seconds = time.perf_counter() - started
        peak = tracemalloc.get_traced_memory()[1] if trace else None
    finally:
        if trace:
            tracemalloc.stop()
    return {
        "samples": samples,
        "seconds": seconds,
        "peak_traced_bytes": peak,
        "sha256": digest.hexdigest() if digest is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-windows", type=int, nargs="+", default=[1, 31])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            *args.episode_windows,
            args.episodes,
            args.batch_size,
            args.history,
            args.horizon,
            args.image_size,
            args.repeats,
        )
        <= 0
    ):
        parser.error("All counts must be positive")
    measure(args)  # Warm caches before timing.
    times = [measure(args)["seconds"] for _ in range(args.repeats)]
    report = {
        "scope": (
            "Ordered collation of decoded overlapping uint8 RGB and float32 action "
            "windows, including episode tails, owned outputs and instructions; "
            "excludes NPZ IO/validation, construction and training. Synthetic "
            "episodes reuse backing arrays. Timing, tracing and checksums use "
            "separate passes; traced peak is not process RSS."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "seconds": times,
        "median_seconds": float(np.median(times)),
        "peak_traced_bytes": measure(args, trace=True)["peak_traced_bytes"],
        **{
            key: value
            for key, value in measure(args, checksum=True).items()
            if key in ("samples", "sha256")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
