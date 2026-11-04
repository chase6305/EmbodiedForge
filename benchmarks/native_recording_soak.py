"""Sustained CPU recording parity with bounded temporary disk usage.

Paired native environments receive identical seeds/actions. Every segment compares
synchronous transition storage against asynchronous compact storage through the
public reader, then removes only its own verified temporary data. Linux resource
samples expose drift without treating allocator RSS as a strict memory contract.
"""

import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
import tracemalloc
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np

from embodiedforge.data import (
    EpisodeReadOptions,
    EpisodeRecorder,
    RecordingMetadata,
    read_episodes,
)
from embodiedforge.envs.evaluation import evaluate_policy
from embodiedforge.envs.locomotion import Go1Environment, H1Environment
from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import file_sha256
from embodiedforge.runtime.implementation import component
from embodiedforge.runtime.process import termination_signals
from embodiedforge.runtime.records import managed_run, run_identity


def resources():
    root = Path("/proc/self")
    if not root.is_dir():
        return {}
    status = dict(
        line.split(":", 1) for line in (root / "status").read_text().splitlines()
    )
    result = {
        "rss_bytes": int(status["VmRSS"].split()[0]) * 1024,
        "threads": int(status["Threads"]),
        "file_descriptors": len(list((root / "fd").iterdir())),
    }
    if tracemalloc.is_tracing():
        result["python_traced_bytes"], result["python_traced_peak_bytes"] = (
            tracemalloc.get_traced_memory()
        )
    return result


def environment(args):
    if args.task == "core":
        from embodiedforge import Config, VectorEnv

        return VectorEnv(
            Config(
                num_envs=args.num_envs,
                seed=args.seed,
                max_steps=args.episode_steps,
                render="raster" if args.image_size else "null",
                channels=("rgb", "depth") if args.image_size else (),
                image_size=args.image_size or 64,
            )
        )
    if args.task == "go1":
        from embodiedforge.locomotion.go1 import Go1

        native = Go1(
            args.num_envs,
            seed=args.seed,
            num_threads=1,
            episode_steps=args.episode_steps,
        )
        return Go1Environment(native)
    from embodiedforge.locomotion.h1_native import H1, build_model

    native = H1(
        build_model(args.h1_model),
        args.num_envs,
        seed=args.seed,
        threads=1,
        episode_steps=args.episode_steps,
    )
    return H1Environment(native)


def digest_episodes(root, *, verify_archive=False):
    digest = hashlib.sha256()
    episodes = transitions = 0
    options = EpisodeReadOptions(verify_archive=True) if verify_archive else None
    for episode in read_episodes(root, options=options):
        episodes += 1
        transitions += len(episode["action"])
        for name, value in sorted(episode.items()):
            digest.update(f"{name}:{value.dtype}:{value.shape}:".encode())
            digest.update(value.tobytes(order="C"))
    return digest.hexdigest(), episodes, transitions


def segment(env, root, metadata, args, *, asynchronous):
    peak = 0
    with EpisodeRecorder(
        root,
        env,
        metadata=metadata,
        episode_format="compact" if asynchronous else "transitions",
        write_workers=2 if asynchronous else 0,
        max_buffer_bytes=args.budget_mib * 1024**2,
        **({"verify_archive": True} if args.verify_archive else {}),
    ) as recorder:

        class Writer:
            def append(self, observation, result):
                nonlocal peak
                recorder.append(observation, result)
                peak = max(peak, recorder.buffered_bytes)

        summary = evaluate_policy(
            env,
            lambda observation: np.zeros(
                (args.num_envs, env.spec["action_shape"][0]), dtype=np.float32
            ),
            steps=args.steps,
            writer=Writer(),
        )
    if (
        recorder.buffered_bytes
        or recorder.pending
        or recorder._writes
        or recorder._executor is not None
    ):
        raise AssertionError("Recorder retained payloads or worker handles after close")
    digest, episodes, transitions = digest_episodes(
        root, verify_archive=args.verify_archive
    )
    if episodes != summary.completed_episodes or episodes == 0:
        raise AssertionError("Published episode count differs from evaluation")
    return {
        "sha256": digest,
        "episodes": episodes,
        "transitions": transitions,
        "evaluation": asdict(summary),
        "max_observed_buffered_bytes": peak,
    }


@termination_signals()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["core", "go1", "h1"], required=True)
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument(
        "--segments", type=int, help="Optional segment cap, useful for quick checks"
    )
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episode-steps", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--budget-mib", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Core raster RGB/depth size; zero disables images",
    )
    parser.add_argument("--trace-allocations", action="store_true")
    parser.add_argument(
        "--verify-archive",
        action="store_true",
        help="Check ZIP CRCs before publication and require SHA256 on read-back",
    )
    parser.add_argument(
        "--h1-model",
        type=Path,
        default=Path("/home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not np.isfinite(args.duration)
        or args.duration <= 0
        or args.seed < 0
        or args.image_size < 0
        or min(args.num_envs, args.episode_steps, args.steps, args.budget_mib) <= 0
        or args.segments is not None
        and args.segments <= 0
    ):
        parser.error("Duration/counts must be positive and seed nonnegative")
    if args.steps < args.episode_steps:
        parser.error("steps must cover at least one full episode")
    if args.image_size and args.task != "core":
        parser.error("image-size is supported only by the core raster control")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / "benchmark.py")
    with (
        managed_run(
            args.output,
            run_identity("benchmark", "validate", args.task),
            metadata={
                "arguments": {
                    name: str(value) if isinstance(value, Path) else value
                    for name, value in vars(args).items()
                },
                "python": sys.executable,
                "benchmark_sha256": file_sha256(Path(__file__)),
            },
        ) as run,
        ExitStack() as owned,
    ):
        environments = [
            owned.enter_context(environment(args))
            if args.task == "core"
            else environment(args)
            for _ in range(2)
        ]
        run["components"] = {
            "environment": component(
                getattr(environments[0], "native", environments[0])
            ),
            "adapter": component(environments[0]),
            "evaluator": component(evaluate_policy),
            "recorder": component(EpisodeRecorder),
            "reader": component(read_episodes),
        }
        write_json(args.output / "run.json", run)
        if args.trace_allocations:
            tracemalloc.start()
        initial_allocations = None
        metadata = (
            None
            if args.task == "core"
            else RecordingMetadata(
                family=args.task,
                max_steps=args.episode_steps,
                seed=args.seed,
                instruction="Follow the velocity command",
                scene=f"native-{args.task}:v1",
                scene_spec={"kind": args.task},
                config={
                    "num_envs": args.num_envs,
                    "episode_steps": args.episode_steps,
                    "seed": args.seed,
                    "threads": 1,
                },
                source={"purpose": "resource and storage parity probe; zero policy"},
            )
        )
        started = time.monotonic()
        samples = []
        episodes = transitions = segments = 0
        trace = hashlib.sha256()
        with (args.output / "samples.jsonl").open("x") as log:
            while time.monotonic() - started < args.duration and (
                args.segments is None or segments < args.segments
            ):
                root = args.output / "segment"
                root.mkdir()
                pair = [
                    segment(env, root / label, metadata, args, asynchronous=bool(index))
                    for index, (label, env) in enumerate(
                        zip(("sync", "async"), environments, strict=True)
                    )
                ]
                if any(
                    pair[0][key] != pair[1][key]
                    for key in ("sha256", "episodes", "transitions", "evaluation")
                ):
                    write_json(args.output / "mismatch.json", pair)
                    raise AssertionError(
                        "Synchronous and asynchronous episode data diverged"
                    )
                shutil.rmtree(root)
                trace.update(pair[0]["sha256"].encode())
                segments += 1
                episodes += pair[0]["episodes"]
                transitions += pair[0]["transitions"]
                gc.collect()
                if args.trace_allocations and initial_allocations is None:
                    initial_allocations = tracemalloc.take_snapshot()
                sample = {
                    "segment": segments,
                    "seconds": time.monotonic() - started,
                    "episodes": pair[0]["episodes"],
                    "transitions": pair[0]["transitions"],
                    "sha256": pair[0]["sha256"],
                    "resources": resources(),
                    "peak_payload_bytes": [
                        value["max_observed_buffered_bytes"] for value in pair
                    ],
                }
                log.write(json.dumps(sample, allow_nan=False) + "\n")
                log.flush()
                if not samples or sample["seconds"] - samples[-1]["seconds"] >= 30:
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
        summary = {
            "scope": "CPU physics, zero policy, paired sync transitions/async compact recording; optional core raster images; timings include read-back and cleanup; resource samples include native allocators",
            "task": args.task,
            "segments": segments,
            "episodes_per_mode": episodes,
            "transitions_per_mode": transitions,
            "trace_sha256": trace.hexdigest(),
            "seconds": time.monotonic() - started,
            "samples": samples,
            "final_resources": resources(),
            "all_segments_equal": True,
        }
        write_json(args.output / "summary.json", summary)
        if initial_allocations is not None:
            final_allocations = tracemalloc.take_snapshot()
            write_json(
                args.output / "allocations.json",
                {
                    "scope": "Python-traced live allocation difference after first verified segment; excludes native allocations",
                    "differences": [
                        {
                            "location": str(item.traceback),
                            "bytes": item.size_diff,
                            "count": item.count_diff,
                        }
                        for item in final_allocations.compare_to(
                            initial_allocations, "lineno"
                        )[:25]
                    ],
                },
            )
            tracemalloc.stop()
        run.update(
            segments=segments,
            episodes_per_mode=episodes,
            transitions_per_mode=transitions,
        )
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
