"""Compare initial and trained policies on the same fixed one-episode batches.

PYTHONPATH=src python benchmarks/check_learning.py runs/ppo/checkpoint.pt
"""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from embodiedforge import VectorEnv
from embodiedforge.training import ActorCritic, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    args = parser.parse_args()
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be nonnegative and unique")
    torch.set_num_threads(1)
    checkpoint = load_checkpoint(args.checkpoint)
    torch.manual_seed(0)
    policies = {
        "untrained_seed_0": ActorCritic(**checkpoint.policy.model_spec).eval(),
        "trained": checkpoint.policy,
    }
    report = {
        "metadata": {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": hashlib.sha256(
                args.checkpoint.read_bytes()
            ).hexdigest(),
            "training_environment": asdict(checkpoint.env_config),
            "num_envs": args.num_envs,
            "seeds": args.seeds,
            "steps": checkpoint.env_config.max_steps,
        }
    }
    for name, policy in policies.items():
        results = []
        for seed in args.seeds:
            config = replace(checkpoint.env_config, num_envs=args.num_envs, seed=seed)
            with VectorEnv(config) as env:
                checkpoint.validate_environment(env.spec)
                reward = np.zeros(config.num_envs)
                success = np.zeros(config.num_envs, dtype=bool)
                for _ in range(config.max_steps):
                    result = env.step(policy.act(env.observe())[:, 0])
                    reward += result.reward
                    success |= result.info["success"]
                results.append(
                    {
                        "seed": seed,
                        "mean_episode_return": float(reward.mean()),
                        "success_rate": float(success.mean()),
                    }
                )
        report[name] = results
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
