"""Measure portable NumPy dynamics or the full environment step/reset path."""

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.backends.physics import NumpyPhysics
from embodiedforge.core import SceneSpec


def measure(args, *, trace=False, checksum=False):
    rng = np.random.default_rng(67)
    actions = rng.uniform(-1, 1, (args.steps, args.num_envs, 2))
    config = Config(
        num_envs=args.num_envs,
        render=args.render,
        image_size=args.image_size,
        channels=("rgb", "depth", "instance_id", "semantic_id")
        if args.render == "raster"
        else (),
    )
    digest = hashlib.sha256() if checksum else None

    def capture(arrays):
        if digest is not None:
            for name, value in sorted(arrays.items()):
                digest.update(name.encode())
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(np.ascontiguousarray(value).data)

    if args.scope == "physics":
        backend = NumpyPhysics()
        backend.build(SceneSpec(), config)
        active = rng.random(args.num_envs) < args.active_fraction

        def step(action):
            backend.apply_control(action)
            backend.step(0.02, 4, active)
            state = backend.snapshot()
            capture(vars(state))

    else:
        backend = VectorEnv(config)

        def step(action):
            result = backend.step(action)
            capture(result.observation)
            capture(
                dict(
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            )
            ids = np.flatnonzero(result.terminated | result.truncated)
            if ids.size:
                capture(backend.reset(ids))

    try:
        for action in actions[: min(8, len(actions))]:
            step(action)
        if trace:
            tracemalloc.start()
        start = time.perf_counter()
        for action in actions:
            step(action)
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if trace else None
        if digest is not None and args.scope == "environment":
            capture(backend.observe())
    finally:
        if trace:
            tracemalloc.stop()
        backend.close()
    return seconds, peak, digest.hexdigest() if digest is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("physics", "environment"), required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--active-fraction", type=float, default=1.0)
    parser.add_argument("--render", choices=("null", "raster"), default="null")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.image_size) <= 0:
        parser.error("Counts must be positive")
    if not 0 <= args.active_fraction <= 1:
        parser.error("active-fraction must be in [0, 1]")
    if args.scope == "physics" and args.render != "null":
        parser.error("physics scope does not render")
    if args.scope == "environment" and args.active_fraction != 1:
        parser.error("environment scope resets completed rows after each step")
    seconds, _, _ = measure(args)
    _, peak, _ = measure(args, trace=True)
    _, _, checksum = measure(args, checksum=True)
    report = {
        "scope": (
            "NumPy planar control + four Euler substeps + owned snapshot; fixed active mask"
            if args.scope == "physics"
            else "NumPy planar environment step, task, optional four-channel raster, sensor copies and immediate completed-row reset"
        )
        + "; excludes construction; timing excludes tracing/checksums; traced peak is not RSS",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "seconds": seconds,
        "peak_traced_bytes": peak,
        "sha256": checksum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
