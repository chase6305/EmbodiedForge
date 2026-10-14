"""Paired episode recording measurements, including owned copies and compression."""

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


def measure(root, *, episode_format, num_envs, steps, image_size, images, noise):
    config = Config(
        num_envs=num_envs,
        max_steps=steps,
        image_size=image_size,
        render="raster" if images else "null",
        channels=("rgb", "depth") if images else (),
    )
    rng = np.random.default_rng(1234)
    timings, peak = [], 0
    # Enough for the legacy layout; comparisons must not fail at different budgets.
    budget = max(256 * 1024**2, num_envs * steps * image_size**2 * 16)
    with VectorEnv(config) as env:
        with EpisodeRecorder(
            root, env, episode_format=episode_format, max_buffer_bytes=budget
        ) as writer:
            observation = env.observe()
            for _ in range(steps):
                result = env.step(rng.uniform(-0.2, 0.2, (num_envs, 2)))
                if noise and images:
                    # Synthetic high-entropy payloads isolate compression cost;
                    # they do not represent physical camera observations.
                    result.observation["rgb"][:] = rng.integers(
                        0, 256, result.observation["rgb"].shape, dtype=np.uint8
                    )
                    result.observation["depth"][:] = rng.random(
                        result.observation["depth"].shape, dtype=np.float32
                    )
                    result.observation["depth_valid"][:] = (
                        result.observation["depth"] > 0
                    )
                start = time.perf_counter()
                writer.append(observation, result)
                timings.append(time.perf_counter() - start)
                peak = max(peak, writer.buffered_bytes)
                observation = result.observation
    digest = hashlib.sha256()
    started = time.perf_counter()
    episodes = transitions = 0
    for episode in read_episodes(root):
        episodes += 1
        transitions += len(episode["action"])
        for name in sorted(episode):
            digest.update(name.encode())
            digest.update(episode[name].tobytes())
    if episodes != num_envs or transitions != steps * num_envs:
        raise AssertionError(
            "Benchmark trajectories must reach the configured time limit"
        )
    return {
        "format": episode_format,
        "append_seconds": sum(timings),
        "max_append_seconds": max(timings),
        "p99_append_seconds": float(np.percentile(timings, 99)),
        "peak_retained_payload_bytes": peak,
        "archive_bytes": sum(path.stat().st_size for path in root.glob("*.npz")),
        "read_seconds": time.perf_counter() - started,
        "episodes": episodes,
        "transitions": transitions,
        "transition_sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument(
        "--noise", action="store_true", help="Synthetic high-entropy image payloads"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.image_size, args.pairs) <= 0:
        parser.error("All counts must be positive")
    rows = []
    for pair in range(args.pairs):
        for images in (False, True):
            order = (
                ("transitions", "compact")
                if pair % 2 == 0
                else ("compact", "transitions")
            )
            for episode_format in order:
                with tempfile.TemporaryDirectory(
                    prefix="ef-recorder-benchmark-"
                ) as temporary:
                    row = measure(
                        Path(temporary) / "run",
                        episode_format=episode_format,
                        num_envs=args.num_envs,
                        steps=args.steps,
                        image_size=args.image_size,
                        images=images,
                        noise=args.noise,
                    )
                rows.append({"pair": pair, "images": images, **row})
    summary = []
    for images in (False, True):
        group = [row for row in rows if row["images"] == images]
        if len({row["transition_sha256"] for row in group}) != 1:
            raise AssertionError("Recording formats changed public transitions")
        keys = (
            "append_seconds",
            "max_append_seconds",
            "peak_retained_payload_bytes",
            "archive_bytes",
            "read_seconds",
        )
        summary.append(
            {
                "images": images,
                "transitions_identical": True,
                "medians": {
                    name: {
                        key: statistics.median(
                            row[key] for row in group if row["format"] == name
                        )
                        for key in keys
                    }
                    for name in ("transitions", "compact")
                },
            }
        )
    report = {
        "scope": "NumPy CPU, synchronous recorder validation/copies/NPZ compression; excludes physics/render time; retained bytes exclude Python objects and stacking/compression scratch",
        "image_payload": "synthetic noise" if args.noise else "raster",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "samples": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
