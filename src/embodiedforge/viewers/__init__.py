"""Interactive viewers are independent of physics and training camera backends."""

from importlib import metadata
from platform import python_version
from sys import version_info
from typing import Protocol

from embodiedforge.backends import NEWTON_VERSION
from embodiedforge.core import Array, Config, Observation, SceneSpec, SceneUpdate


def viewer_dependencies(name: str) -> dict:
    """Read package metadata only; does not certify GPU/display/driver availability."""
    requirements = {
        "viser": ("viser", "pillow"),
        "gl": ("newton", "warp-lang", "pyglet"),
        "rtx": ("newton", "warp-lang", "pyglet", "usd-core", "ovrtx"),
        "web": ("pillow",),
    }
    if name not in requirements:
        raise ValueError(f"Unknown viewer: {name}")
    packages = {}
    optional = {}
    for package in requirements[name]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    if name in ("gl", "rtx"):
        try:
            optional["imgui-bundle"] = metadata.version("imgui-bundle")
        except metadata.PackageNotFoundError:
            optional["imgui-bundle"] = None
    notes = []
    if name == "rtx" and not (3, 11) <= version_info[:2] <= (3, 13):
        notes.append(
            "OVRTX upstream documents Python 3.11-3.13 support. "
            "An installed package on this Python version is not a supported-environment check."
        )
    if name in ("gl", "rtx") and optional.get("imgui-bundle") is None:
        notes.append(
            "Native UI panels are unavailable: install embodiedforge[viz-ui]. "
            "Python 3.10 has no imgui-bundle >=1.92 wheel; use a separate Python 3.11 "
            "viewer environment for binary installation."
        )
    return {
        "viewer": name,
        "python": python_version(),
        "packages": packages,
        "optional_packages": optional,
        "environment_notes": notes,
        "metadata_ok": all(packages.values())
        and ("newton" not in packages or packages["newton"] == NEWTON_VERSION),
        "check_scope": "installation_metadata_only",
        "install_extra": "viz" if name == "viser" else f"viz-{name}",
    }


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
    """Fail before physics initialization with an actionable install command."""
    report = viewer_dependencies(name)
    if report["metadata_ok"]:
        return
    missing = [
        package for package, version in report["packages"].items() if version is None
    ]
    problems = [f"missing packages: {', '.join(missing)}"] if missing else []
    installed = report["packages"].get("newton")
    if installed is not None and installed != NEWTON_VERSION:
        problems.append(f"Newton {installed} installed; requires {NEWTON_VERSION}")
    raise ImportError(
        f"Viewer {name}: {'; '.join(problems)}. "
        f"In the active environment, run: python -m pip install -e '.[{report['install_extra']}]'"
    )


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
