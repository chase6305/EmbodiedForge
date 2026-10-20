"""Measure depth validation separately from rendering, allocation and hashing."""

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge.core import Config, RenderBatch, SceneUpdate, StateSnapshot
from embodiedforge.sensors import SensorPipeline


def measure(num_envs, image_size, calls, layout):
    config = Config(num_envs=num_envs, image_size=image_size, channels=("depth",))
    rng = np.random.default_rng(42)
    depth = rng.random((num_envs, 1, image_size, image_size), dtype=np.float32)
    depth[depth < 0.2] = 0
    if layout == "fortran":
        depth = np.asfortranarray(depth)
    elif layout == "reversed":
        depth = depth[..., ::-1, ::-1]
    elif layout == "strided":
        depth = np.repeat(depth, 2, axis=-1)[..., ::2]
    valid = depth > 0
    if layout == "mixed":
        valid = np.asfortranarray(valid)
    state = StateSnapshot(
        np.zeros((num_envs, 2)),
        np.zeros((num_envs, 2)),
        np.zeros(num_envs),
        np.zeros(num_envs, dtype=np.int64),
    )
    batch = RenderBatch(
        {"depth": depth, "depth_valid": valid}, state.time, state.version
    )
    update = SceneUpdate(state, {})
    pipeline = SensorPipeline(config, None)
    ids = np.arange(num_envs)
    pipeline._validate(batch, ids, update)
    started = time.perf_counter()
    for _ in range(calls):
        pipeline._validate(batch, ids, update)
    seconds = time.perf_counter() - started
    tracemalloc.start()
    pipeline._validate(batch, ids, update)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    digest = hashlib.sha256()
    for values in (depth, valid):
        digest.update(str((values.shape, values.dtype.str)).encode())
        digest.update(np.ascontiguousarray(values).data)
    return {"seconds": seconds, "peak_traced_bytes": peak, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=540)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument(
        "--layout",
        choices=("contiguous", "fortran", "reversed", "mixed", "strided"),
        default="contiguous",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.image_size, args.calls) <= 0:
        parser.error("Environment count, image size and calls must be positive")
    report = {
        "scope": "SensorPipeline depth batch validation only; excludes rendering, input generation and hashing; timing separate from traced allocation; inputs excluded from traced peak; not process RSS",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **measure(args.num_envs, args.image_size, args.calls, args.layout),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
