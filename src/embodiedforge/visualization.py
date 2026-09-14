"""Debug one selected environment through a unified Web UI or native viewer.

Run with python -m embodiedforge.visualization --render-backend mujoco.
The simulation stays in the main thread; GUI callbacks only signal requests.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from .backends import PHYSICS
from .core import Config
from .env import VectorEnv
from .logging import auto_configure_debug_logging, get_logger, setup_logging
from .tasks import TASKS
from .viewers import create_viewer, require_viewer_dependencies, viewer_dependencies
from .viewers.timing import RateClock
from .viewers.viser import DebugViewer as DebugViewer  # Backward-compatible import.


def main() -> None:
    """Run a real-time demonstrator; Ctrl+C or --duration closes all resources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", choices=sorted(PHYSICS), default="numpy")
    parser.add_argument("--task", choices=sorted(TASKS), default="reach")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument(
        "--viewer", choices=("web", "viser", "gl", "rtx"), default="web"
    )
    parser.add_argument(
        "--render-backend",
        choices=("raster", "mujoco", "gl", "rtx"),
        default=None,
        help="Web viewer frame renderer (default: raster)",
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check viewer package metadata without opening a window",
    )
    parser.add_argument(
        "--env-id", type=int, default=0, help="Initial selected environment"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Native viewers: render without a window",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Run seconds after the first frame; 0 runs until Ctrl+C",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument(
        "--log-json", action="store_true", help="Structured logs on stderr"
    )
    args = parser.parse_args()
    debug_configured = auto_configure_debug_logging()
    if args.log_level is not None or args.log_json:
        setup_logging(args.log_level or "INFO", with_json_format=args.log_json)
    elif not debug_configured and not args.check:
        setup_logging("INFO")
    logger = get_logger(__name__)
    if args.check:
        if args.viewer == "web":
            from .viewers.frames import frame_dependencies

            report = frame_dependencies(args.render_backend or "raster")
        else:
            report = viewer_dependencies(args.viewer)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["metadata_ok"] else 1)
    if not 1 <= args.fps <= 60 or not 1 <= args.port <= 65535:
        parser.error("--fps must be 1..60 and --port must be 1..65535")
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("--duration must be finite and nonnegative")
    if not 0 <= args.env_id < args.num_envs:
        parser.error("--env-id must be in range for --num-envs")
    if any(not 64 <= value <= 1920 for value in (args.width, args.height)):
        parser.error("--width and --height must be 64..1920")
    if args.render_backend is not None and args.viewer != "web":
        parser.error("--render-backend is only used with --viewer web")
    if args.headless and args.viewer in ("viser", "web"):
        parser.error(
            "--headless is for gl/rtx; browser viewers already run without a window"
        )
    try:
        require_viewer_dependencies(args.viewer)
    except ImportError as error:
        parser.error(str(error))
    if args.viewer == "web":
        from .viewers.frames import frame_dependencies

        report = frame_dependencies(args.render_backend or "raster")
        if not report["metadata_ok"]:
            parser.error(
                f"Web renderer dependencies unavailable; install embodiedforge[{report['install_extra']}]"
            )
        if sys.platform.startswith("linux"):
            os.environ.setdefault("MUJOCO_GL", "egl")
    config = Config(
        task=args.task,
        physics=args.physics,
        render="raster" if args.viewer == "viser" else "null",
        num_envs=args.num_envs,
        channels=("rgb",) if args.viewer == "viser" else (),
        image_size=128,
        camera_hz=10,
    )
    logger = get_logger(
        __name__, physics=config.physics, viewer=args.viewer, task=config.task
    )
    logger.info(
        "Preparing %s viewer (physics=%s, task=%s)",
        args.viewer,
        config.physics,
        config.task,
    )
    preparing = time.monotonic()
    with VectorEnv(config) as env:
        viewer = create_viewer(
            args.viewer,
            config,
            env.task.scene,
            host=args.host,
            port=args.port,
            env_id=args.env_id,
            headless=args.headless,
            **(
                {
                    "render_backend": args.render_backend or "raster",
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                }
                if args.viewer == "web"
                else {}
            ),
        )
        try:
            logger.info(
                "Rendering initial frame; first-time shader compilation may take longer"
            )
            reward = np.zeros(config.num_envs)
            viewer.update(
                env.task.scene_update(env.physics.snapshot()), env.observe(), reward
            )
            started = time.monotonic()
            logger.info("Viewer ready in %.2f s", started - preparing)
            next_frame = started + 1 / args.fps
            physics_clock = RateClock()
            while viewer.is_running() and (
                args.duration == 0 or time.monotonic() - started < args.duration
            ):
                tick = time.monotonic()
                while (index := viewer.consume_reset()) is not None:
                    env.reset([index])
                    reward[index] = 0
                step_due = args.viewer != "web" or physics_clock.due(
                    tick, config.control_hz * viewer.playback_speed
                )
                if step_due and viewer.should_step():
                    done = env.terminated | env.truncated
                    if done.any():
                        env.reset(np.flatnonzero(done))
                    result = env.step(env.task.expert_action(env.observe()))
                    reward = result.reward
                if (
                    viewer.update_due(tick)
                    if args.viewer == "web"
                    else tick >= next_frame
                ):
                    viewer.update(
                        env.task.scene_update(env.physics.snapshot()),
                        env.observe(),
                        reward,
                    )
                    next_frame = tick + 1 / args.fps
                interval = 1 / (
                    config.control_hz * getattr(viewer, "playback_speed", 1.0)
                )
                if args.viewer == "web":
                    time.sleep(
                        max(
                            0,
                            min(
                                0.02,
                                min(physics_clock.next, viewer.next_frame_time)
                                - time.monotonic(),
                            ),
                        )
                    )
                else:
                    time.sleep(max(0, interval - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            pass
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                viewer.close()
            except Exception:
                if not active_error:
                    raise
                logger.exception(
                    "Viewer cleanup also failed; preserving the original error"
                )
            else:
                logger.info("Viewer stopped")


if __name__ == "__main__":
    main()
