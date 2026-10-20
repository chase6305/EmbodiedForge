"""Measure motion file loading, decoded validation, or object-conditioned sampling."""

import argparse
import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge.motion import MotionClips


def measure(path, *, scope, calls, repeats, objects, batch_size, seed):
    library = MotionClips.load(path)
    if objects is None:
        objects = list(dict.fromkeys(library.objects))
    requests = [objects[i % len(objects)] for i in range(batch_size)]
    rng = np.random.default_rng(seed)

    def operation():
        if scope == "load":
            return MotionClips.load(path)
        if scope == "validate":
            return MotionClips(library.arrays, library.metadata)
        return library.sample(requests, rng)

    operation()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            operation()
        samples.append((time.perf_counter() - started) / calls)
    tracemalloc.start()
    try:
        operation()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # Checksum a separate, reproducible result, outside timing and tracing.
    rng = np.random.default_rng(seed)
    result = operation()
    digest = hashlib.sha256()
    if scope == "sample":
        for value in result:
            digest.update(value.tobytes())
        digest.update(json.dumps(rng.bit_generator.state, sort_keys=True).encode())
    else:
        digest.update(json.dumps(result.metadata, sort_keys=True).encode())
        for key, value in sorted(result.arrays.items()):
            digest.update(key.encode())
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
    return {
        "seconds_per_call": samples,
        "median_seconds_per_call": statistics.median(samples),
        "peak_traced_bytes": peak,
        "result_sha256": digest.hexdigest(),
        "input_sha256": library.sha256,
        "frames": int(library.starts[-1]),
        "clips": len(library.names),
        "requested_objects": objects if scope == "sample" else None,
        "batch_size": batch_size if scope == "sample" else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--scope", choices=("load", "validate", "sample"), required=True
    )
    parser.add_argument("--calls", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--objects", nargs="+")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.calls, args.repeats, args.batch_size) <= 0 or args.seed < 0:
        parser.error("Counts must be positive and seed nonnegative")
    if args.objects is not None and args.scope != "sample":
        parser.error("--objects is only used with --scope sample")
    report = {
        "scope": {
            "load": "CPU open, cached NPZ decode, SHA256 and MotionClips validation",
            "validate": "CPU MotionClips construction over already decoded input arrays",
            "sample": "CPU object-conditioned clip/frame sampling with an existing NumPy generator and decoded library",
        }[args.scope]
        + "; timing excludes setup, output hashing and tracing; traced peak excludes preloaded inputs and is not RSS; no simulator or training throughput claim",
        "numpy_version": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **measure(
            args.input,
            scope=args.scope,
            calls=args.calls,
            repeats=args.repeats,
            objects=args.objects,
            batch_size=args.batch_size,
            seed=args.seed,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "median_seconds_per_call",
                    "peak_traced_bytes",
                    "result_sha256",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
