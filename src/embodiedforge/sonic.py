"""Evaluate the pinned SONIC X2 policies in headless MuJoCo on CPU."""

import argparse
import subprocess
from pathlib import Path

from ._motion_sdk import launch

REVISION = "c9959443f80276533083a8b31ecd3760a41eaa65"
MODELS = {
    "transfer-v2": "x2_sonic_frozen_g1core_lora_v2.onnx",
    "incumbent-14000": "x2_sonic_14000_g1.onnx",
}
MOTIONS = {
    "walk": "x2_relaxed_walk.pkl",
    "idle": "x2_idle_stand.pkl",
    "dance": "x2_gangam_dance.pkl",
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--sonic-root", dest="root", type=Path, required=True)
    evaluate.add_argument("--python", type=Path, required=True, help="SONIC SDK Python")
    evaluate.add_argument("--model", choices=MODELS, default="transfer-v2")
    evaluate.add_argument("--motion", choices=MOTIONS, required=True)
    evaluate.add_argument(
        "--clip", help="Exact bundled clip key; default evaluates all"
    )
    evaluate.add_argument("--init-frame", type=int, default=0)
    evaluate.add_argument("--threads", type=int, default=2, help="CPU ORT threads")
    evaluate.add_argument("--headless", action="store_true", help="Always headless")
    evaluate.add_argument(
        "--output", type=Path, required=True, help="New run directory"
    )
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.init_frame < 0 or args.threads < 1:
        cli.error("--init-frame must be nonnegative and --threads positive")
    repo = args.root.expanduser().resolve()
    sources = {
        "policy.onnx": repo / "models" / MODELS[args.model],
        "motions.pkl": repo / "motions" / MOTIONS[args.motion],
    }
    if args.model == "incumbent-14000":
        sources["tuning.yaml"] = repo / "configs/real_deploy_tuning/bigrun.yaml"
    try:
        print(
            launch(
                "sonic",
                args,
                REVISION,
                {
                    "model": args.model,
                    "motion": args.motion,
                    "clip": args.clip,
                    "init_frame": args.init_frame,
                    "threads": args.threads,
                },
                sources,
            )
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"SONIC evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
