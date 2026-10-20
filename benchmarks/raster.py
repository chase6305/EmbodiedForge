"""Isolate CPU raster rendering time, traced allocation peak and pixel identity."""

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.backends.render import RasterRenderer


def measure(
    num_envs,
    image_size,
    frames,
    channels=("rgb", "depth", "instance_id", "semantic_id"),
):
    config = Config(
        num_envs=num_envs,
        image_size=image_size,
        channels=channels,
    )
    rng = np.random.default_rng(19)
    positions = rng.uniform(-1.6, 1.6, (frames, 2, num_envs, 2))
    ids = np.arange(num_envs)
    renderer = RasterRenderer()
    with VectorEnv(Config(num_envs=num_envs)) as env:
        update = env.task.scene_update(env.physics.snapshot())
        renderer.build(env.task.scene, config)
        renderer.sync(update)

        def draw(index):
            update.entities["agent"][:] = positions[index, 0]
            update.entities["target"][:] = positions[index, 1]
            return renderer.render(ids)

        draw(0)
        started = time.perf_counter()
        for index in range(frames):
            draw(index)
        seconds = time.perf_counter() - started
        tracemalloc.start()
        draw(0)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        digest = hashlib.sha256()
        for index in range(frames):
            rendered = draw(index)
            for name in sorted(rendered.images):
                digest.update(name.encode())
                digest.update(rendered.images[name].data)
        renderer.close()
    return {"seconds": seconds, "peak_traced_bytes": peak, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument(
        "--channels",
        nargs="+",
        choices=("rgb", "depth", "instance_id", "semantic_id"),
        default=["rgb", "depth", "instance_id", "semantic_id"],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.image_size, args.frames) <= 0:
        parser.error("Counts must be positive")
    report = {
        "scope": "CPU reference raster only; requested channels; excludes physics, sensor copies, JPEG and web delivery; timing excludes tracemalloc/checksumming; traced peak is not RSS",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **measure(args.num_envs, args.image_size, args.frames, args.channels),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
