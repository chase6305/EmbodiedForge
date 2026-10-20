"""Measure owned visual observations returned by full and partial CPU resets."""

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv


def measure(num_envs, selected, image_size, resets):
    config = Config(
        num_envs=num_envs,
        render="raster",
        image_size=image_size,
        channels=("rgb", "depth", "instance_id", "semantic_id"),
    )
    # Noncanonical order exercises the selection contract too.
    ids = np.arange(selected - 1, -1, -1)
    with VectorEnv(config) as env:
        start = time.perf_counter()
        for _ in range(resets):
            env.reset(ids)
        seconds = time.perf_counter() - start
        tracemalloc.start()
        env.reset(ids)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    digest = hashlib.sha256()
    with VectorEnv(config) as env:
        for _ in range(resets):
            observation = env.reset(ids)
            for name in sorted(observation):
                digest.update(name.encode())
                digest.update(observation[name].data)
        # Include unselected rows in the final correctness comparison.
        for name, value in sorted(env.observe().items()):
            digest.update(name.encode())
            digest.update(value.data)
    return {"seconds": seconds, "peak_traced_bytes": peak, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--selected", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--resets", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.selected <= args.num_envs or min(args.image_size, args.resets) <= 0:
        parser.error("Require 0 < selected <= num-envs and positive sizes/counts")
    report = {
        "scope": "NumPy physics/task reset, reference raster, sensor cache and owned observations; four channels; excludes initial construction; timing excludes tracing/checksumming; traced peak is not process RSS",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **measure(args.num_envs, args.selected, args.image_size, args.resets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
