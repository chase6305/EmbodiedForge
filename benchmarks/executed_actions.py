"""Measure executed-action metadata or complete camera-free reference steps."""

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.env import _executed_action


def measure(args, *, trace=False, checksum=False):
    rng = np.random.default_rng(91)
    action = rng.uniform(-0.5, 0.5, (args.num_envs, args.action_dim))
    active = np.zeros(args.num_envs, bool)
    active[: int(args.num_envs * args.active_fraction)] = True
    if args.layout == "F":
        action = np.asfortranarray(action)
    action.flags.writeable = active.flags.writeable = False
    env = None
    if args.scope == "environment":
        env = VectorEnv(
            Config(task=args.task, num_envs=args.num_envs, max_steps=args.steps + 32)
        )
        env.physics.position[:] = 0.5
        env.physics.velocity[:] = 0
        if args.task == "reach":
            env.task.target[:] = 1
        env.terminated[:] = ~active
        action = np.zeros_like(action)
        action.flags.writeable = False

        def call():
            result = env.step(action)
            return {
                **result.observation,
                **result.info,
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
            }
    else:

        def call():
            return {"executed_action": _executed_action(action, active)}

    digest = hashlib.sha256() if checksum else None
    try:
        for _ in range(8):
            call()
        if trace:
            tracemalloc.start()
        tick = time.perf_counter()
        for _ in range(args.steps):
            output = call()
            if digest is not None:
                for key, value in sorted(output.items()):
                    digest.update(key.encode())
                    digest.update(str((value.shape, value.dtype.str)).encode())
                    digest.update(value.tobytes())
            del output
        elapsed = time.perf_counter() - tick
        peak = tracemalloc.get_traced_memory()[1] if trace else None
        return elapsed, peak, digest.hexdigest() if digest is not None else None
    finally:
        if trace:
            tracemalloc.stop()
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("action", "environment"), default="action")
    parser.add_argument("--task", choices=("reach", "hold"), default="reach")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--layout", choices=("C", "F"), default="C")
    parser.add_argument("--active-fraction", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.action_dim, args.steps) <= 0:
        parser.error("Counts must be positive")
    if not 0 <= args.active_fraction <= 1:
        parser.error("active-fraction must lie in [0, 1]")
    if args.scope == "environment" and (args.layout != "C" or args.action_dim != 2):
        parser.error("Reference environment uses two-dimensional actions and C layout")
    seconds, _, _ = measure(args)
    _, peak, _ = measure(args, trace=True)
    _, _, digest = measure(args, checksum=True)
    report = dict(
        scope="Synthetic CPU metadata or complete no-camera reference step. "
        "Construction/warmup excluded; timing excludes tracing and checksums; "
        "separate traced transient peaks are not process RSS or training throughput.",
        python=platform.python_version(),
        numpy=np.__version__,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        seconds=seconds,
        peak_traced_bytes=peak,
        sha256=digest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
