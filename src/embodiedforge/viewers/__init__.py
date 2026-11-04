"""Interactive viewers are independent of physics and training camera backends."""

from typing import Protocol

from embodiedforge.backends import NEWTON_VERSION as NEWTON_VERSION
from embodiedforge.core import Array, Config, Observation, SceneSpec, SceneUpdate

from .dependencies import require_viewer_dependencies as _require_viewer_dependencies
from .dependencies import viewer_dependencies as viewer_dependencies


class ViewerBackend(Protocol):
    """Main-thread viewer lifecycle; update consumes borrowed public snapshots."""

    def is_running(self) -> bool: ...
    def should_step(self) -> bool: ...
    def consume_reset(self) -> int | None: ...
    def update(
        self, scene: SceneUpdate, observation: Observation, reward: Array
    ) -> None: ...
    def close(self) -> None: ...


def require_viewer_dependencies(name: str) -> None:
    """Check dependencies through the public report hook before creating physics."""
    _require_viewer_dependencies(name, report=viewer_dependencies(name))


def create_viewer(
    name: str,
    config: Config,
    scene: SceneSpec,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    env_id: int = 0,
    headless: bool = False,
    render_backend: str = "raster",
    width: int = 960,
    height: int = 540,
    fps: int = 30,
    replay_frames: tuple[int, ...] | None = None,
    camera=None,
) -> ViewerBackend:
    """Select a lazy adapter; reject unsupported scenes before SDK allocation."""
    if name not in ("web", "viser", "gl", "rtx"):
        raise ValueError(f"Unknown viewer: {name}")
    if scene.kind == "robot_motion" and name != "web":
        raise ValueError("Robot mesh scenes require the Web viewer")
    if scene.kind != "robot_motion" and (
        scene.kind != "point_reach" or not {"agent", "target"} <= set(scene.entity_ids)
    ):
        raise ValueError("Preview adapters require the point_reach agent/target scene")
    if type(env_id) is not int or not 0 <= env_id < config.num_envs:
        raise ValueError("Viewer env_id must be an integer in range")
    if name == "web":
        from .web import WebViewer

        return WebViewer(
            config,
            scene,
            backend=render_backend,
            host=host,
            port=port,
            env_id=env_id,
            width=width,
            height=height,
            fps=fps,
            replay_frames=replay_frames,
            camera=camera,
        )
    if name == "viser":
        if headless or "rgb" not in config.channels:
            raise ValueError("Viser requires RGB observations and no headless flag")
        from .viser import DebugViewer

        viewer = DebugViewer(config, host, port, radius=scene.radius)
        viewer.selected.value = str(env_id)
        return viewer
    from .newton import NewtonViewer

    return NewtonViewer(name, scene, env_id=env_id, headless=headless)
