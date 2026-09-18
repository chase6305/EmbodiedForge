"""Command-line composition root; core runtime never imports this module."""

import argparse
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

from .backends import PHYSICS, RENDER, execution_plan
from .core import Config
from .data import EpisodeRecorder, read_episodes
from .env import VectorEnv
from .logging import auto_configure_debug_logging, get_logger, setup_logging
from .rollout import DemonstrationPolicy, run_rollout


def main() -> None:
    """Parse configuration, select a workflow and own its process-level resources."""
    if sys.argv[1:2] == ["act"]:
        from .act import main as act_main

        act_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["h1-native"]:
        from .native_h1 import main as native_h1_main

        native_h1_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["live"]:
        from .go1_live import main as live_main

        live_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["replay"]:
        from .robot_replay import main as replay_main

        replay_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["recipes"]:
        from .recipes import main as recipes_main

        recipes_main(sys.argv[2:])
        return
    if sys.argv[1:2] == ["h1"]:
        from .h1 import main as h1_main

        h1_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        description="EmbodiedForge CPU reference training platform"
    )
    parser.add_argument(
        "command",
        choices=[
            "plan",
            "doctor",
            "rollout",
            "train",
            "evaluate",
            "inspect",
            "h1",
            "h1-native",
            "recipes",
            "replay",
            "live",
            "act",
        ],
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--task", help="Registered task name, e.g. reach or hold")
    parser.add_argument("--physics", choices=sorted(PHYSICS))
    parser.add_argument("--render", choices=sorted(RENDER))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--updates", type=int, default=30)
    parser.add_argument(
        "--chunk-horizon",
        type=int,
        default=1,
        help="Maximum policy chunk length for rollout/evaluate",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--record-buffer-mib",
        type=int,
        default=256,
        help="Retained recording array budget in MiB; excludes compression scratch",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--graphics",
        choices=("egl",),
        help="doctor only: render a test frame and report the actual EGL device",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument(
        "--log-json", action="store_true", help="Structured logs on stderr"
    )
    args = parser.parse_args()
    if args.graphics and args.command != "doctor":
        parser.error("--graphics is only supported by doctor")
    auto_configure_debug_logging()
    if args.log_level is not None or args.log_json:
        setup_logging(args.log_level or "INFO", with_json_format=args.log_json)
    logger = get_logger(__name__)
    if args.command in ("rollout", "evaluate"):
        if args.steps <= 0 or args.chunk_horizon <= 0:
            parser.error("--steps and --chunk-horizon must be positive")
        if args.record_buffer_mib <= 0:
            parser.error("--record-buffer-mib must be positive")
    checkpoint = None
    if args.command == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        from .training import load_checkpoint

        checkpoint = load_checkpoint(args.checkpoint)
    config = (
        Config.load(args.config)
        if args.config
        else (checkpoint.env_config if checkpoint else Config())
    )
    overrides = {
        key: getattr(args, key)
        for key in ("task", "physics", "render", "num_envs")
        if getattr(args, key) is not None
    }
    if args.render == "null":
        overrides["channels"] = ()
    elif args.render == "raster" and not config.channels:
        overrides["channels"] = ("rgb", "depth", "instance_id")
    config = replace(config, **overrides)
    logger.info(
        "Starting %s: task=%s physics=%s render=%s",
        args.command,
        config.task,
        config.physics,
        config.render,
    )
    if args.command == "doctor":
        from .diagnostics import installation_report

        report = installation_report(config)
        if args.graphics:
            from .graphics import egl_render_report

            report["graphics"] = egl_render_report()
            report["check_scope"] = "installation_metadata_and_egl_framebuffer"
        print(json.dumps(report, indent=2))
        valid = report["metadata_ok"] and report.get("graphics", {}).get("ok", True)
        raise SystemExit(0 if valid else 1)
    if args.command == "plan":
        print(json.dumps(execution_plan(config), indent=2))
        return
    if args.command == "inspect":
        if args.dataset is None:
            parser.error("inspect requires --dataset")
        lengths = [len(e["action"]) for e in read_episodes(args.dataset)]
        print(json.dumps({"episodes": len(lengths), "transitions": sum(lengths)}))
        return
    if args.command == "train":
        if args.output is None:
            parser.error("train requires a new --output directory")
        import torch

        from .training import PPOConfig, train_ppo

        torch.set_num_threads(1)
        train_ppo(config, PPOConfig(updates=args.updates), args.output)
        return
    policy = checkpoint.policy if checkpoint else None
    with VectorEnv(config) as env:
        if checkpoint:
            checkpoint.validate_environment(env.spec)
        if policy is None and not callable(getattr(env.task, "expert_action", None)):
            raise ValueError(
                "rollout requires a task-provided expert_action demonstrator"
            )
        if policy is None:
            policy = DemonstrationPolicy(env.task.expert_action)
        recorder = (
            EpisodeRecorder(
                args.output,
                env,
                max_buffer_bytes=args.record_buffer_mib * 1024 * 1024,
            )
            if args.output
            else nullcontext()
        )
        with recorder as writer:
            summary = run_rollout(
                env,
                policy,
                steps=args.steps,
                chunk_horizon=args.chunk_horizon,
                writer=writer,
            )
    print(json.dumps(asdict(summary)))
