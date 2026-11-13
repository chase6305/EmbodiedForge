"""CPU artifact checks executed only inside the separate IsaacLab environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import math
import sys
from pathlib import Path

if not __package__:
    # -P omits the script directory; expose the package without shadowing logging.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embodiedforge.learning.checkpoints import finite_tensors
from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import stream_sha256


def check_checkpoint(path: Path, *, include_hash=False) -> dict:
    import torch

    with path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
        for key in (
            "actor_state_dict",
            "critic_state_dict",
            "optimizer_state_dict",
            "iter",
        ):
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing {key}")
        iteration = checkpoint["iter"]
        if type(iteration) is not int or iteration < 0:
            raise ValueError("Invalid checkpoint iteration")
        actor, critic = checkpoint["actor_state_dict"], checkpoint["critic_state_dict"]
        if (
            tuple(actor["mlp.0.weight"].shape) != (128, 69)
            or tuple(actor["mlp.6.weight"].shape) != (19, 128)
            or tuple(critic["mlp.0.weight"].shape) != (128, 69)
            or tuple(critic["mlp.6.weight"].shape) != (1, 128)
        ):
            raise ValueError("Checkpoint does not match the H1 flat actor/critic")
        count = finite_tensors(checkpoint, check_scalars=True)
        result = {"checkpoint_iteration": iteration, "finite_checkpoint_tensors": count}
        if include_hash:
            stream.seek(0)
            result["checkpoint_sha256"] = stream_sha256(stream)
        return result


def preflight(args) -> dict:
    if sys.version_info[:2] != (3, 12):
        raise ValueError("The verified IsaacLab environment requires Python 3.12")
    packages = {}
    for name in ("isaaclab", "isaaclab_tasks", "isaaclab_rl", "isaaclab_newton"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Package not installed: {name}")
        path = Path(spec.origin).resolve()
        if not path.is_relative_to(args.repo.resolve()):
            raise ValueError(
                f"{name} resolves outside the selected IsaacLab checkout: {path}"
            )
        packages[name] = str(path)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "warp-lang", "newton", "rsl-rl-lib", "tensorboard")
    }
    if versions["rsl-rl-lib"] != "5.0.1":
        raise ValueError("This launcher validates the RSL-RL 5.0.1 checkpoint contract")
    result = {"python": sys.version, "packages": packages, "versions": versions}
    if args.checkpoint:
        result.update(check_checkpoint(args.checkpoint))
    return result


def verify(args) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    expected = args.start + args.updates - 1
    checkpoints = list(args.run.glob(f"logs/rsl_rl/h1_flat/*/model_{expected}.pt"))
    if len(checkpoints) != 1:
        raise ValueError(
            f"Expected one final model_{expected}.pt, found {len(checkpoints)}"
        )
    checkpoint = checkpoints[0]
    result = check_checkpoint(checkpoint, include_hash=True)
    if result["checkpoint_iteration"] != expected:
        raise ValueError("Checkpoint iteration does not match requested updates")
    events = EventAccumulator(
        str(checkpoint.parent), size_guidance={"scalars": 0}
    ).Reload()
    scalars = {tag: events.Scalars(tag) for tag in events.Tags()["scalars"]}
    required_steps = set(range(args.start, args.start + args.updates))
    for tag in ("Loss/value", "Loss/surrogate", "Policy/mean_std"):
        entries = scalars.get(tag, [])
        if (
            len(entries) != args.updates
            or {entry.step for entry in entries} != required_steps
        ):
            raise ValueError(f"Incomplete PPO metric history: {tag}")
    if not all(
        math.isfinite(entry.value) for entries in scalars.values() for entry in entries
    ):
        raise ValueError("Non-finite training metric")
    for name in ("agent.yaml", "env.yaml"):
        if not (checkpoint.parent / "params" / name).is_file():
            raise ValueError(f"Missing resolved configuration: {name}")
    result.update(
        checkpoint=str(checkpoint.relative_to(args.run)),
        completed_updates=args.updates,
        scalar_count=sum(len(entries) for entries in scalars.values()),
        all_scalars_finite=True,
        last_scalars={
            tag: entries[-1].value for tag, entries in scalars.items() if entries
        },
        scope="PPO updates and numeric checks only; walking quality is not assessed.",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("preflight")
    probe.add_argument("--repo", type=Path, required=True)
    probe.add_argument("--checkpoint", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("--run", type=Path, required=True)
    check.add_argument("--start", type=int, required=True)
    check.add_argument("--updates", type=int, required=True)
    for child in (probe, check):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args) if args.command == "preflight" else verify(args)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
