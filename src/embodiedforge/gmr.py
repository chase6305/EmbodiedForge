"""Headless BVH retargeting through an explicitly selected GMR checkout."""

import argparse
import math
import subprocess
from pathlib import Path

from ._motion_sdk import launch

REVISION = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
# Only advertise pairs present in the pinned IK_CONFIG_DICT.
ROBOTS = {
    "lafan1": (
        "unitree_g1",
        "unitree_g1_with_hands",
        "booster_t1_29dof",
        "fourier_n1",
        "stanford_toddy",
        "engineai_pm01",
        "pal_talos",
    ),
    "nokov": ("unitree_g1",),
    "xsens": ("unitree_g1", "unitree_h1_2"),
}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    retarget = commands.add_parser("retarget")
    retarget.add_argument("--gmr-root", dest="root", type=Path, required=True)
    retarget.add_argument("--python", type=Path, required=True, help="GMR SDK Python")
    retarget.add_argument("--input", type=Path, required=True, help="BVH source")
    retarget.add_argument("--format", choices=ROBOTS, required=True)
    retarget.add_argument("--robot", default="unitree_g1")
    retarget.add_argument("--human-height", type=float, default=1.75, help="Metres")
    retarget.add_argument(
        "--output", type=Path, required=True, help="New run directory"
    )
    retarget.add_argument("--headless", action="store_true", help="Always headless")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.robot not in ROBOTS[args.format]:
        cli.error(f"{args.format} supports: {', '.join(ROBOTS[args.format])}")
    if not math.isfinite(args.human_height) or args.human_height <= 0:
        cli.error("--human-height must be finite and positive")
    try:
        print(
            launch(
                "gmr",
                args,
                REVISION,
                {
                    "format": args.format,
                    "robot": args.robot,
                    "human_height": args.human_height,
                },
                {"source.bvh": args.input},
            )
        )
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        cli.exit(1, f"GMR retargeting failed: {exc}\n")


if __name__ == "__main__":
    main()
