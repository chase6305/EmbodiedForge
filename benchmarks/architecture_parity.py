"""Compare deterministic CPU training across two source trees in fresh processes.

This checks numerical compatibility of a refactor, not learning quality. Each
tree gets new run directories; source inputs and existing runs are read-only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def _digest(value):
    import numpy as np
    import torch

    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().numpy()
        if isinstance(item, np.ndarray):
            digest.update(f"{item.dtype}:{item.shape}:".encode())
            digest.update(item.tobytes(order="C"))
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())

    visit(value)
    return digest.hexdigest()


def _probe(args):
    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    if args.component == "core":
        from embodiedforge import Config
        from embodiedforge.training import PPOConfig, train_ppo

        model = train_ppo(
            Config(num_envs=4, seed=args.seed, max_steps=5),
            PPOConfig(
                updates=8, rollout_steps=8, epochs=2, minibatch_size=16, seed=args.seed
            ),
            args.output / "training",
        )
        checkpoint = torch.load(
            args.output / "training/checkpoint.pt",
            map_location="cpu",
            weights_only=True,
        )
        result = {
            "policy": _digest(model.state_dict()),
            "optimizer": _digest(checkpoint["optimizer"]),
        }
    elif args.component == "go1":
        from embodiedforge.locomotion import go1, go1_ppo

        env = go1.Go1(4, seed=args.seed, num_threads=1, episode_steps=5)
        model = go1_ppo.ActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=go1_ppo.LR)
        traces = []
        for _ in range(3):
            batch, _ = go1_ppo.rollout(model, env, horizon=8)
            traces.append(_digest(batch))
            go1_ppo.update(model, optimizer, batch, *go1_ppo.gae(batch))
        result = {
            "policy": _digest(model.state_dict()),
            "optimizer": _digest(optimizer.state_dict()),
            "rollouts": traces,
        }
    else:
        from embodiedforge.locomotion import h1_native, h1_ppo

        env = h1_native.H1(
            h1_native.build_model(args.h1_model),
            4,
            seed=args.seed,
            threads=1,
            episode_steps=5,
        )
        model = h1_ppo.Policy()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        observation = env.obs()
        rng = np.random.default_rng(args.seed)
        traces = []
        for _ in range(3):
            batch, observation = h1_ppo.collect(model, env, observation, horizon=8)
            traces.append(_digest(batch))
            h1_ppo.optimize(model, optimizer, batch, rng)
        result = {
            "policy": _digest(model.state_dict()),
            "optimizer": _digest(optimizer.state_dict()),
            "rollouts": traces,
        }
    (args.output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument(
        "--candidate-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--native", action="store_true")
    parser.add_argument(
        "--h1-model",
        type=Path,
        default=Path("/home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml"),
    )
    parser.add_argument(
        "--component", choices=["core", "go1", "h1"], help=argparse.SUPPRESS
    )
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.component:
        _probe(args)
        return
    if args.reference_root is None:
        parser.error("--reference-root is required")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct nonnegative integers")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "scope": "deterministic CPU policy, optimizer and rollout equality",
        "comparisons": [],
    }
    for component in ["core", "go1", "h1"] if args.native else ["core"]:
        for seed in args.seeds:
            results = {}
            for name, root in (
                ("reference", args.reference_root),
                ("candidate", args.candidate_root),
            ):
                output = args.output / f"{component}-{seed}-{name}"
                output.mkdir()
                environment = {
                    **os.environ,
                    "PYTHONPATH": str(root.resolve() / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                command = [
                    str(args.python.absolute()),
                    str(Path(__file__).resolve()),
                    "--component",
                    component,
                    "--seed",
                    str(seed),
                    "--output",
                    str(output),
                    "--h1-model",
                    str(args.h1_model.resolve()),
                ]
                with (output / "console.log").open("wb") as log:
                    subprocess.run(
                        command,
                        cwd=root.resolve(),
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=180,
                    )
                results[name] = json.loads((output / "result.json").read_text())
            comparison = {
                "component": component,
                "seed": seed,
                "match": results["reference"] == results["candidate"],
                **results,
            }
            summary["comparisons"].append(comparison)
            print(
                json.dumps(
                    {key: comparison[key] for key in ("component", "seed", "match")}
                ),
                flush=True,
            )
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
    if not all(case["match"] for case in summary["comparisons"]):
        raise SystemExit("Numerical parity failed; inspect summary.json")


if __name__ == "__main__":
    main()
