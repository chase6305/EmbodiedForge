"""Compare identical planar force dynamics, including control and snapshot copies."""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from embodiedforge.backends.mjbatch import MjbatchPhysics
from embodiedforge.backends.physics import MujocoPhysics
from embodiedforge.core import Config, SceneSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.threads, args.repeats) <= 0:
        parser.error("All counts must be positive")
    times = {"mujoco_sequential": [], "mjbatch": []}
    factories = {
        "mujoco_sequential": MujocoPhysics,
        "mjbatch": lambda: MjbatchPhysics(args.threads),
    }
    action = np.random.default_rng(0).uniform(-1, 1, (args.num_envs, 2))
    active = np.ones(args.num_envs, bool)
    final = {}
    for repeat in range(args.repeats):
        for name in list(factories)[:: 1 if repeat % 2 == 0 else -1]:
            backend = factories[name]()
            try:
                backend.build(SceneSpec(), Config(num_envs=args.num_envs))
                backend.reset(np.arange(args.num_envs), np.zeros((args.num_envs, 2)))
                for _ in range(10):
                    backend.apply_control(action)
                    backend.step(0.02, 4, active)
                    backend.snapshot()
                start = time.perf_counter()
                for _ in range(args.steps):
                    backend.apply_control(action)
                    backend.step(0.02, 4, active)
                    state = backend.snapshot()
                times[name].append(time.perf_counter() - start)
                final[name] = state
            finally:
                backend.close()
    for field in ("position", "velocity", "time", "version"):
        np.testing.assert_allclose(
            getattr(final["mjbatch"], field),
            getattr(final["mujoco_sequential"], field),
            atol=1e-12,
        )
    medians = {name: statistics.median(values) for name, values in times.items()}
    result = {
        "num_envs": args.num_envs,
        "steps": args.steps,
        "threads": args.threads,
        "repeats": args.repeats,
        "seconds": times,
        "median_seconds": medians,
        "transitions_per_second": {
            name: args.num_envs * args.steps / elapsed
            for name, elapsed in medians.items()
        },
        "speedup": medians["mujoco_sequential"] / medians["mjbatch"],
        "final_states_match": True,
        "logical_cpus": os.cpu_count(),
        "platform": platform.platform(),
        "scope": "planar 2-DOF, control copy + 4 physics substeps + owned snapshot; excludes task/render",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
