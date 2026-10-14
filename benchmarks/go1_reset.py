"""Measure selected-row Go1 resets in the isolated mjbatch/MuJoCo SDK.

Timing excludes construction, warmup, allocation tracing and trajectory hashing.
The independent 40-step digest includes state, rewards and terminal flags.
"""

import argparse
import gc
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np


def measure(num_envs, selected, resets):
    from embodiedforge.locomotion.go1 import Go1

    ids = np.arange(selected - 1, -1, -1, dtype=np.int64)

    def make():
        return Go1(num_envs, seed=61042, num_threads=1)

    env = make()
    try:
        for _ in range(3):
            env.reset(ids)
        started = time.perf_counter()
        for _ in range(resets):
            env.reset(ids)
        seconds = time.perf_counter() - started
        tracemalloc.start()
        env.reset(ids)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        del env
        gc.collect()
    digest = hashlib.sha256()
    env = make()
    try:
        for step in range(40):
            if step % 3 == 0:
                env.reset(ids)
            result = env.step(np.full((num_envs, 12), 0.025 * np.sin(step), np.float32))
            for value in [
                env.qpos,
                env.qvel,
                env.command,
                env.until,
                env.clock,
                env.steps,
                env.action,
                env.down,
                env.friction,
                *env.foot_pos,
                *result[:3],
            ]:
                digest.update(str((value.shape, value.dtype.str)).encode())
                digest.update(np.ascontiguousarray(value).data)
    finally:
        del env
        gc.collect()
    return dict(
        num_envs=num_envs,
        selected=selected,
        resets=resets,
        seconds=seconds,
        peak_traced_bytes=peak,
        sha256=digest.hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--selected", type=int, default=1)
    parser.add_argument("--resets", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.selected <= args.num_envs or args.resets <= 0:
        parser.error("Require 0 < selected <= num-envs and positive reset count")
    report = {
        "scope": "Complete native Go1.reset, CPU threads=1; construction/warmup, trace and independent 40-step trajectory hashing excluded from timing; traced NumPy/Python allocation is not native model memory or RSS",
        **measure(args.num_envs, args.selected, args.resets),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
