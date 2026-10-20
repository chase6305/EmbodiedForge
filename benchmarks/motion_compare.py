"""Measure decoded, validated motion scoring without simulator or file loading."""

import argparse
import hashlib
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge.motion import MotionClips, compare_clips


def measure(reference, rollout, *, calls, repeats):
    expected = compare_clips(reference, rollout)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(calls):
            compare_clips(reference, rollout)
        samples.append((time.perf_counter() - started) / calls)
    tracemalloc.start()
    try:
        compare_clips(reference, rollout)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    result = compare_clips(reference, rollout)
    if result != expected:
        raise AssertionError("Motion scoring changed between identical calls")
    encoded = json.dumps(result, sort_keys=True, allow_nan=False).encode()
    return {
        "seconds_per_call": samples,
        "median_seconds_per_call": statistics.median(samples),
        "peak_traced_bytes": peak,
        "result_sha256": hashlib.sha256(encoded).hexdigest(),
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.calls, args.repeats) <= 0:
        parser.error("calls and repeats must be positive")
    reference, rollout = (
        MotionClips.load(path) for path in (args.reference, args.rollout)
    )
    report = {
        "scope": "CPU compare_clips over already decoded and validated inputs; excludes loading, hashing, simulation and construction. Timing and allocation tracing are separate. Traced peak is temporary Python/NumPy allocation, not input storage or process RSS. No training throughput claim.",
        "numpy_version": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "reference_frames": int(reference.starts[-1]),
        "rollout_frames": int(rollout.starts[-1]),
        "bodies": len(reference.metadata["body_names"]),
        "joints": len(reference.metadata["joint_names"]),
        **measure(reference, rollout, calls=args.calls, repeats=args.repeats),
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
