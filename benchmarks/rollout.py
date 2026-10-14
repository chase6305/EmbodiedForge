"""Measure full rollouts with synchronized timeouts and owned visual resets."""

import argparse
import hashlib
import json
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.rollout import run_rollout
from embodiedforge.tasks import ReachTask


class TimeoutTask(ReachTask):
    def evaluate(self, state, actions, previous):
        n = len(state.time)
        return (
            np.ones(n, dtype=np.float32),
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=bool),
        )


class ZeroPolicy:
    def act(self, observation):
        return np.zeros((len(observation["env_id"]), 1, 2), dtype=np.float32)


class DigestWriter:
    def __init__(self):
        self.digest = hashlib.sha256()

    def append(self, observation, result):
        for arrays in (observation, result.observation, result.info):
            for key, value in sorted(arrays.items()):
                self.digest.update(key.encode())
                self.digest.update(str((value.shape, value.dtype.str)).encode())
                self.digest.update(np.ascontiguousarray(value).data)
        for value in (result.reward, result.terminated, result.truncated):
            self.digest.update(value.data)


def measure(config, steps):
    with VectorEnv(config, task=TimeoutTask()) as env:
        run_rollout(env, ZeroPolicy(), steps=2)
        start = time.perf_counter()
        summary = run_rollout(env, ZeroPolicy(), steps=steps)
        seconds = time.perf_counter() - start
    with VectorEnv(config, task=TimeoutTask()) as env:
        tracemalloc.start()
        run_rollout(env, ZeroPolicy(), steps=steps)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    writer = DigestWriter()
    with VectorEnv(config, task=TimeoutTask()) as env:
        run_rollout(env, ZeroPolicy(), steps=steps, writer=writer)
    return {
        "seconds": seconds,
        "peak_traced_bytes": peak,
        "sha256": writer.digest.hexdigest(),
        "summary": asdict(summary),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--channels", nargs="*", default=["rgb", "depth"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    config = Config(
        num_envs=args.num_envs,
        image_size=args.image_size,
        max_steps=args.max_steps,
        render="raster" if args.channels else "null",
        channels=tuple(args.channels),
    )
    report = {
        "scope": "NumPy point task, zero policy, complete step/reset loop; synchronized timeouts; excludes construction, writer and hashing from timing; traced peak is not RSS",
        "config": config.to_dict(),
        "steps": args.steps,
        **measure(config, args.steps),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
