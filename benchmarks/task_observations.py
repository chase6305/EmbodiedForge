"""Measure task feature assembly or complete camera-free CPU environment steps."""

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.core import StateSnapshot
from embodiedforge.tasks import make_task


def measure(args, *, trace=False, checksum=False):
    task = make_task(args.task)
    digest = hashlib.sha256() if checksum else None
    if args.scope == "task":
        task.build(args.num_envs)
        rng = np.random.default_rng(81)
        fields = []
        for _ in range(3):
            value = rng.normal(size=(args.num_envs, 2)).astype(args.dtype)
            if args.layout == "F":
                value = np.asfortranarray(value)
            elif args.layout == "strided":
                value = np.repeat(value, 2, axis=0)[::2]
            value.flags.writeable = False
            fields.append(value)
        position, velocity, task.target = fields
        state = StateSnapshot(
            position,
            velocity,
            np.zeros(args.num_envs),
            np.zeros(args.num_envs, np.int64),
        )

        def step():
            return task.observe(state)

        def close():
            pass
    else:
        env = VectorEnv(
            Config(task=args.task, num_envs=args.num_envs, max_steps=args.steps + 32),
            task,
        )
        env.physics.position[:] = 0.5
        env.physics.velocity[:] = 0
        if args.task == "reach":
            task.target[:] = 1
        action = np.zeros((args.num_envs, 2))

        def step():
            result = env.step(action)
            return {
                **result.observation,
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
            }

        close = env.close
    try:
        for _ in range(8):
            step()
        if trace:
            tracemalloc.start()
        start = time.perf_counter()
        for _ in range(args.steps):
            values = step()
            if digest is not None:
                for name, value in sorted(values.items()):
                    digest.update(name.encode())
                    digest.update(str((value.shape, value.dtype.str)).encode())
                    digest.update(value.tobytes())
            del values
        seconds = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if trace else None
        return seconds, peak, digest.hexdigest() if digest is not None else None
    finally:
        if trace:
            tracemalloc.stop()
        close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("reach", "hold"), default="reach")
    parser.add_argument("--scope", choices=("task", "environment"), default="task")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--layout", choices=("C", "F", "strided"), default="C")
    parser.add_argument(
        "--dtype", choices=("float32", "float64", ">f8"), default="float64"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.steps) <= 0:
        parser.error("Counts must be positive")
    if args.scope == "environment" and (args.dtype != "float64" or args.layout != "C"):
        parser.error("Environment state uses native float64 and C layout")
    seconds, _, _ = measure(args)
    _, peak, _ = measure(args, trace=True)
    _, _, checksum = measure(args, checksum=True)
    report = dict(
        scope="Synthetic CPU task.observe or complete no-camera VectorEnv.step; "
        "excludes construction/warmup; timing excludes tracing/checksums; "
        "transient traced peak is not RSS or training throughput",
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        seconds=seconds,
        peak_traced_bytes=peak,
        sha256=checksum,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
