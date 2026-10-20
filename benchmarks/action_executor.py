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


def measure(*, num_envs, action_dim, horizon, steps, metadata=False, repeat_every=0):
    observation = {"env_id": np.arange(num_envs)}
    if metadata:
        observation.update(
            episode_id=np.zeros(num_envs, dtype=np.int64),
            step_id=np.zeros(num_envs, dtype=np.int64),
        )
    # Precomputed policy output keeps neural inference out of this measurement.
    chunks = (
        (np.arange(num_envs * horizon * action_dim, dtype=np.float32) % 31) / 31 - 0.5
    ).reshape(num_envs, horizon, action_dim)

    class Policy:
        def __init__(self):
            self.calls = 0

        def act(self, batch):
            self.calls += 1
            return chunks[batch["env_id"]]

    policy = Policy()
    executor = ActionExecutor(num_envs, action_dim, horizon)
    for control in range(2 * horizon):
        executor.act(observation, policy)
        if metadata and (not repeat_every or (control + 1) % repeat_every):
            observation["step_id"] += 1
    policy.calls = 0
    started = time.perf_counter()
    for control in range(2 * horizon, 2 * horizon + steps):
        action = executor.act(observation, policy)
        if metadata and (not repeat_every or (control + 1) % repeat_every):
            observation["step_id"] += 1
    seconds = time.perf_counter() - started
    timed_policy_calls = policy.calls
    # Construction and initial policy inference happen outside allocation tracing.
    executor = ActionExecutor(num_envs, action_dim, horizon)
    executor.act(observation, policy)
    if metadata:
        observation["step_id"] += 1
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
        "timed_policy_calls": timed_policy_calls,
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
    parser.add_argument(
        "--metadata", action="store_true", help="Track episode/step IDs"
    )
    parser.add_argument(
        "--repeat-every",
        type=int,
        default=0,
        help="With --metadata, repeat every Kth step identity (0 disables)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.action_dim, args.horizon, args.steps) <= 0:
        parser.error("All counts must be positive")
    if args.repeat_every < 0 or args.repeat_every and not args.metadata:
        parser.error("--repeat-every requires --metadata and a nonnegative value")
    result = measure(
        num_envs=args.num_envs,
        action_dim=args.action_dim,
        horizon=args.horizon,
        steps=args.steps,
        metadata=args.metadata,
        repeat_every=args.repeat_every,
    )
    report = {
        "scope": (
            "NumPy CPU ActionExecutor with precomputed policy chunks, including "
            "periodic replanning validation/copies; excludes construction, warmup, "
            "physics and neural inference. Allocation tracing uses a separate "
            "single call after one warmup call; horizon=1 includes replanning. "
            "With --metadata, timing also includes per-step ID increments and "
            "automatic chunk tracking. --repeat-every repeats every Kth step "
            "identity, forcing replanning on its next consumption. Allocation "
            "tracing covers one consecutive act alone, without identity repeat. "
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
