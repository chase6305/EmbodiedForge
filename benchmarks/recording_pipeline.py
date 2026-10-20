"""End-to-end CPU acquisition with optional bounded background episode writes."""

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


def measure(
    root,
    *,
    workers,
    episode_format,
    num_envs,
    steps,
    cycles,
    image_size,
    budget_mib,
    noise,
):
    config = Config(
        num_envs=num_envs,
        max_steps=steps,
        image_size=image_size,
        render="raster",
        channels=("rgb", "depth"),
    )
    rng = np.random.default_rng(16)
    append_times = []
    peak = 0
    with VectorEnv(config) as env:
        writer = EpisodeRecorder(
            root,
            env,
            episode_format=episode_format,
            write_workers=workers,
            max_buffer_bytes=budget_mib * 1024**2,
        )
        start = time.perf_counter()
        with writer:
            for _ in range(cycles):
                observation = env.observe()
                for _ in range(steps):
                    result = env.step(np.zeros((num_envs, 2)))
                    if noise:
                        result.observation["rgb"][:] = rng.integers(
                            0, 256, result.observation["rgb"].shape, dtype=np.uint8
                        )
                        result.observation["depth"][:] = rng.random(
                            result.observation["depth"].shape, dtype=np.float32
                        )
                        result.observation["depth_valid"][:] = (
                            result.observation["depth"] > 0
                        )
                    before = time.perf_counter()
                    writer.append(observation, result)
                    append_times.append(time.perf_counter() - before)
                    peak = max(peak, writer.buffered_bytes)
                    observation = result.observation
                env.reset()
            close_start = time.perf_counter()
        close_seconds = time.perf_counter() - close_start
        seconds = time.perf_counter() - start
    digest = hashlib.sha256()
    episodes = transitions = 0
    for episode in read_episodes(root):
        episodes += 1
        transitions += len(episode["action"])
        for name in sorted(episode):
            digest.update(name.encode())
            digest.update(episode[name].data)
    if episodes != num_envs * cycles or transitions != num_envs * steps * cycles:
        raise AssertionError("Acquisition must reach every configured time limit")
    return {
        "workers": workers,
        "format": episode_format,
        "seconds_including_close": seconds,
        "append_seconds": sum(append_times),
        "max_append_seconds": max(append_times),
        "p99_append_seconds": float(np.percentile(append_times, 99)),
        "close_seconds": close_seconds,
        "max_observed_buffered_bytes": peak,
        "episodes": episodes,
        "transitions": transitions,
        "sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--format", choices=("transitions", "compact"), default="compact"
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--budget-mib", type=int, default=256)
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.workers < 0
        or min(args.num_envs, args.steps, args.cycles, args.image_size, args.budget_mib)
        <= 0
    ):
        parser.error("Counts/budget must be positive and workers nonnegative")
    with tempfile.TemporaryDirectory(prefix="ef-acquisition-") as temporary:
        result = measure(
            Path(temporary) / "run",
            workers=args.workers,
            episode_format=args.format,
            num_envs=args.num_envs,
            steps=args.steps,
            cycles=args.cycles,
            image_size=args.image_size,
            budget_mib=args.budget_mib,
            noise=args.noise,
        )
    report = {
        "scope": "NumPy physics+raster+recording+final writer drain; excludes environment/writer construction and subsequent full-data audit; buffer count excludes stack/compression scratch; noise is synthetic, not a physical camera",
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
