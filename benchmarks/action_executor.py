"""Measure CPU action-chunk execution with independent allocation tracing."""

import argparse
import hashlib
import json
import platform
import time
import tracemalloc
from pathlib import Path

import numpy as np

from embodiedforge.policies import ActionExecutor


def measure(*, num_envs, action_dim, horizon, steps):
    observation = {"env_id": np.arange(num_envs)}
    # Precomputed policy output keeps neural inference out of this measurement.
    chunks = (
        (np.arange(num_envs * horizon * action_dim, dtype=np.float32) % 31) / 31 - 0.5
    ).reshape(num_envs, horizon, action_dim)

    class Policy:
        def act(self, batch):
            return chunks[batch["env_id"]]

    policy = Policy()
    executor = ActionExecutor(num_envs, action_dim, horizon)
    for _ in range(2 * horizon):
        executor.act(observation, policy)
    started = time.perf_counter()
    for _ in range(steps):
        action = executor.act(observation, policy)
    seconds = time.perf_counter() - started
    # Construction and initial policy inference happen outside allocation tracing.
    executor = ActionExecutor(num_envs, action_dim, horizon)
    executor.act(observation, policy)
    tracemalloc.start()
    try:
        action = executor.act(observation, policy)
        live, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert action.flags.owndata and action.flags.c_contiguous
    assert not np.shares_memory(action, executor.chunks)
    np.testing.assert_array_equal(action, chunks[:, 1 % horizon])
    return {
        "seconds": seconds,
        "actions_per_second": num_envs * steps / seconds,
        "single_act_live_traced_bytes": live,
        "single_act_peak_traced_bytes": peak,
        "action_sha256": hashlib.sha256(action.tobytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.action_dim, args.horizon, args.steps) <= 0:
        parser.error("All counts must be positive")
    result = measure(
        num_envs=args.num_envs,
        action_dim=args.action_dim,
        horizon=args.horizon,
        steps=args.steps,
    )
    report = {
        "scope": (
            "NumPy CPU ActionExecutor with precomputed policy chunks, including "
            "periodic replanning validation/copies; excludes construction, warmup, "
            "physics and neural inference. Allocation tracing uses a separate "
            "single call after one warmup call; horizon=1 includes replanning. "
            "Traced bytes are not process RSS or native/CUDA memory."
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
