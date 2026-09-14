"""Counterbalanced CPU PPO and inference microbenchmark in the Go1 SDK.

Use PYTHONPATH=src and the isolated mjbatch Python. A reference PPO source file
may be supplied from an older run's implementation snapshot. This measures
learner/inference time; it does not estimate end-to-end simulation throughput.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-ppo", type=Path)
    parser.add_argument("--num-envs", type=positive, default=512)
    parser.add_argument("--horizon", type=positive, default=24)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--pairs", type=positive, default=8)
    parser.add_argument("--updates", type=positive, default=3)
    parser.add_argument("--inference-repeats", type=positive, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.num_envs * args.horizon < 8 or args.num_envs * args.horizon % 4:
        parser.error("Rollout must contain at least 8 samples, divisible by 4")

    import numpy as np
    import torch

    from embodiedforge.locomotion import go1_ppo as current
    from embodiedforge.locomotion.go1 import Go1

    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = current.ActorCritic()
    model.load_state_dict(checkpoint["model_state_dict"])
    env = Go1(
        args.num_envs,
        seed=42,
        num_threads=args.threads,
        semantics=checkpoint.get("task_semantics", "upstream-v1"),
        reward_profile=checkpoint.get("reward_profile", "original"),
        command_profile=checkpoint.get("command_profile", "original"),
    )
    batch, _ = current.rollout(model, env, horizon=args.horizon, record_policy=True)
    plain_batch = {
        key: value
        for key, value in batch.items()
        if key not in ("policy_mean", "policy_log_std")
    }
    adv, ret = current.gae(batch)
    modes = [
        ("current_plain", current, False, plain_batch),
        ("current_recomputed_diagnostics", current, True, plain_batch),
        ("current_cached_diagnostics", current, True, batch),
    ]
    reference = None
    if args.reference_ppo:
        spec = importlib.util.spec_from_file_location(
            "embodiedforge.locomotion._benchmark_reference_ppo", args.reference_ppo
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        modes = [
            ("reference_plain", reference, False, plain_batch),
            ("reference_diagnostics", reference, True, plain_batch),
            *modes,
        ]
    rows = []
    for pair in range(args.pairs):
        offset = pair % len(modes)
        order = modes[offset:] + modes[:offset]
        if (pair // len(modes)) % 2:
            order = list(reversed(order))
        for name, module, diagnostics, data in order:
            net = module.ActorCritic()
            net.load_state_dict(checkpoint["model_state_dict"])
            optimizer = torch.optim.Adam(net.parameters(), lr=0.00005)
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(
                    copy.deepcopy(checkpoint["optimizer_state_dict"])
                )
            torch.manual_seed(20)
            module.update(net, optimizer, data, adv, ret, diagnostics=diagnostics)
            start = time.perf_counter()
            for _ in range(args.updates):
                module.update(net, optimizer, data, adv, ret, diagnostics=diagnostics)
            rows.append(
                {
                    "pair": pair,
                    "mode": name,
                    "seconds_per_update": (time.perf_counter() - start) / args.updates,
                }
            )
    medians = {
        name: float(
            np.median([r["seconds_per_update"] for r in rows if r["mode"] == name])
        )
        for name, *_ in modes
    }
    obs = batch["obs"][0, :32]
    methods = {
        "current_with_critic": lambda: model(obs)[0],
        "current_actor_only": lambda: model.action_mean(obs),
    }
    if reference is not None:
        old = reference.ActorCritic()
        old.load_state_dict(checkpoint["model_state_dict"])
        methods["reference_with_critic"] = lambda: old(obs)[0]
    inference = []
    with torch.inference_mode():
        expected = model.action_mean(obs)
        for fn in methods.values():
            torch.testing.assert_close(fn(), expected, rtol=0, atol=0)
        for pair in range(args.pairs):
            order = list(methods.items())
            offset = pair % len(order)
            order = order[offset:] + order[:offset]
            for name, fn in order:
                fn()
                start = time.perf_counter()
                for _ in range(args.inference_repeats):
                    fn()
                inference.append(
                    {
                        "pair": pair,
                        "mode": name,
                        "seconds_per_call": (time.perf_counter() - start)
                        / args.inference_repeats,
                    }
                )
    report = {
        "scope": "CPU learner-only, reused real rollout; excludes physics and I/O. Machine load affects timings.",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": torch.__version__,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "current_source_sha256": hashlib.sha256(
            Path(current.__file__).read_bytes()
        ).hexdigest(),
        "reference_source_sha256": hashlib.sha256(
            args.reference_ppo.read_bytes()
        ).hexdigest()
        if args.reference_ppo
        else None,
        "cached_distribution_bytes": sum(
            batch[key].numel() * batch[key].element_size()
            for key in ("policy_mean", "policy_log_std")
        ),
        "samples": rows,
        "median_seconds_per_update": medians,
        "inference_num_envs": len(obs),
        "inference_samples": inference,
        "inference_median_seconds": {
            name: float(
                np.median(
                    [r["seconds_per_call"] for r in inference if r["mode"] == name]
                )
            )
            for name in methods
        },
        "inference_actions_identical": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), "median_seconds_per_update": medians}, indent=2
        )
    )


if __name__ == "__main__":
    main()
