"""Window-independent frame renderers for the shared browser viewer.

Adapters consume public point snapshots or recorded MJCF articulation states.
Graphics SDKs run in isolated processes; no private physics state is shared.
"""

import multiprocessing
import os
import sys
from dataclasses import dataclass, replace
from importlib import metadata
from typing import Protocol

import numpy as np

from embodiedforge.core import Config, Observation, SceneSpec, SceneUpdate
from embodiedforge.logging import get_logger

FRAME_BACKENDS = ("raster", "mujoco", "gl", "rtx")
LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class OrbitCamera:
    azimuth: float = -90.0
    elevation: float = 45.0
    distance: float = 3.5
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def position(self):
        azimuth, elevation = np.deg2rad([self.azimuth, self.elevation])
        return np.asarray(self.target) + self.distance * np.array(
            [
                np.cos(elevation) * np.cos(azimuth),
                np.cos(elevation) * np.sin(azimuth),
                np.sin(elevation),
            ]
        )


class FrameRenderer(Protocol):
    """Main-thread render lifecycle; RGB output is owned, uint8 and top-left."""

    camera_control: bool

    def render(
        self,
        scene: SceneUpdate,
        observation: Observation,
        env_id: int,
        camera: OrbitCamera,
    ) -> np.ndarray: ...
    def close(self) -> None: ...


def frame_dependencies(name, scene=None):
    from . import viewer_dependencies

    if name not in FRAME_BACKENDS:
        raise ValueError(f"Unknown frame renderer: {name}")
    if name in ("gl", "rtx"):
        report = viewer_dependencies(name)
        requirements = dict(report["packages"])
        compatible = report["metadata_ok"]
    else:
        requirements = {"mujoco": None} if name == "mujoco" else {}
        compatible = True
    robot = scene is not None and scene.kind == "robot_motion"
    if robot:
        requirements["mujoco"] = None
    for package in (*requirements, "pillow"):
        try:
            requirements[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            requirements[package] = None
    if robot and requirements["mujoco"] is not None:
        compatible = compatible and tuple(
            int(v) for v in requirements["mujoco"].split(".")[:2]
        ) >= (3, 11)
    return {
        "renderer": name,
        "packages": requirements,
        "metadata_ok": compatible and all(requirements.values()),
        "supported": not (robot and name == "raster"),
        "check_scope": "installation_metadata_only",
        "install_extra": {
            "raster": "viz-web",
            "mujoco": "viz-web,mujoco",
            "gl": "viz-web,viz-gl",
            "rtx": "viz-web,viz-rtx",
        }[name]
        if not robot
        else "viz-robot" + (f",viz-{name}" if name in ("gl", "rtx") else ""),
    }


class RasterFrames:
    camera_control = False

    def __init__(self, scene, config, width, height):
        from embodiedforge.backends.render import RasterRenderer

        self.renderer = RasterRenderer()
        self.renderer.build(
            scene, replace(config, channels=("rgb",), image_size=min(width, height))
        )
        size = min(width, height)
        y, x = np.indices((size, size))
        tile = (x * 12 // size + y * 12 // size) % 2
        self.background = np.asarray([(28, 35, 46), (32, 40, 52)], dtype=np.uint8)[tile]

    def render(self, scene, observation, env_id, camera):
        self.renderer.sync(scene)
        image = self.renderer.render(np.array([env_id])).images["rgb"][0, 0].copy()
        background = np.all(image == (20, 24, 32), axis=-1)
        image[background] = self.background[background]
        return image

    def close(self):
        self.renderer.close()


class MujocoFrames:
    camera_control = True

    def __init__(self, scene, config, width, height):
        # Respect an explicit backend choice. EGL permits browser rendering on
        # Linux without an X display; this must precede the first MuJoCo import.
        if sys.platform.startswith("linux"):
            os.environ.setdefault("MUJOCO_GL", "egl")
        import mujoco

        self.mujoco = mujoco
        self.renderer = None
        from .style import FLOOR_A, FLOOR_B, SKY_HORIZON, SKY_TOP, style_mujoco_lighting

        def rgb(values):
            return " ".join(str(value) for value in values)

        self.model = mujoco.MjModel.from_xml_string(f'''<mujoco>
          <visual><global offwidth="{width}" offheight="{height}"/></visual>
          <asset><texture type="skybox" builtin="gradient" width="64" height="64"
            rgb1="{rgb(SKY_TOP)}" rgb2="{rgb(SKY_HORIZON)}"/>
            <texture name="grid" type="2d" builtin="checker" width="128" height="128"
            rgb1="{rgb(FLOOR_A)}" rgb2="{rgb(FLOOR_B)}"/>
            <material name="floor" texture="grid" texrepeat="1 1" texuniform="true"
            specular=".1" shininess=".1" reflectance="0"/></asset>
          <worldbody><light pos="0 -2 4" dir="0 .4 -1"/>
            <geom type="plane" size="0 0 .1" material="floor"/>
            <body name="agent" mocap="true"><geom type="sphere" size="{scene.radius}"
              rgba=".27 .61 1 1"/></body>
            <body name="target" mocap="true"><geom type="sphere" size=".1"
              rgba=".25 .82 .47 1"/></body>
          </worldbody></mujoco>''')
        style_mujoco_lighting(self.model)
        self.data = mujoco.MjData(self.model)
        self.radius = scene.radius
        self.camera = mujoco.MjvCamera()
        try:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        except BaseException:
            self.close()
            raise

    def render(self, scene, observation, env_id, camera):
        for i, (name, radius) in enumerate((("agent", self.radius), ("target", 0.1))):
            self.data.mocap_pos[i] = (*scene.entities[name][env_id], radius)
        self.mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = camera.target
        self.camera.distance = camera.distance
        # MuJoCo azimuth describes the viewing direction; our orbit azimuth
        # describes the camera's position around the target.
        self.camera.azimuth = camera.azimuth + 180
        self.camera.elevation = -camera.elevation
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class NativeFrames:
    camera_control = True

    def __init__(self, backend, scene, config, width, height):
        from .newton import NewtonViewer

        self.bridge = NewtonViewer(
            backend,
            scene,
            env_id=0,
            headless=True,
            width=width,
            height=height,
            synchronous=True,
        )
        self.backend = backend
        self.camera = None
        self.reward = np.zeros(config.num_envs)

    def render(self, scene, observation, env_id, camera):
        if camera != self.camera:
            import warp as wp

            self.bridge.viewer.set_camera(
                wp.vec3(*camera.position()),
                pitch=-camera.elevation,
                yaw=camera.azimuth + 180,
            )
            self.camera = camera
        self.bridge.selected_env = env_id
        self.bridge.update(scene, observation, self.reward)
        if self.backend == "rtx":
            return self.bridge.viewer._capture_screenshot_pixels()[..., :3].copy()
        return self.bridge.viewer.get_frame().numpy().copy()

    def close(self):
        self.bridge.close()


def _frame_worker(connection, name, scene, config, width, height):
    """Own all SDK calls on this process's main thread, including destruction."""
    renderer = None
    try:
        if sys.platform.startswith("linux"):
            os.environ.setdefault("MUJOCO_GL", "egl")
        renderer = create_frame_renderer(
            name, scene, config, width, height, isolated=False
        )
        connection.send((True, renderer.camera_control))
        while True:
            command, arguments = connection.recv()
            if command == "close":
                break
            connection.send((True, renderer.render(*arguments)))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        LOGGER.exception("Frame renderer worker failed (%s)", name)
        try:
            connection.send((False, f"{type(error).__name__}: {error}"))
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        try:
            if renderer is not None:
                renderer.close()
        finally:
            connection.close()


class IsolatedFrames:
    """Isolate GL/EGL/Vulkan globals and native crashes from physics and HTTP.

    Only public snapshots and owned RGB arrays cross the local pipe. Spawn is
    deliberate: forking a process with an initialized GPU context is unsafe.
    """

    def __init__(self, name, scene, config, width, height):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(
            target=_frame_worker,
            args=(child, name, scene, config, width, height),
            name=f"embodiedforge-render-{name}",
            daemon=True,
        )
        self.closed = False
        try:
            self.process.start()
            child.close()
            self.camera_control = self._receive(180)
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self, timeout):
        if not self.connection.poll(timeout):
            raise TimeoutError(f"Renderer did not respond within {timeout} seconds")
        try:
            success, payload = self.connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError("Renderer process exited unexpectedly") from error
        if not success:
            raise RuntimeError(payload)
        return payload

    def render(self, scene, observation, env_id, camera):
        if self.closed:
            raise RuntimeError("Renderer is closed")
        try:
            self.connection.send(("render", (scene, observation, env_id, camera)))
            return self._receive(180)  # Includes first-frame shader compilation.
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.pid is not None:
                if self.process.is_alive():
                    try:
                        self.connection.send(("close", ()))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
                    self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(timeout=2)
        finally:
            self.connection.close()
            if self.process.pid is not None and not self.process.is_alive():
                self.process.close()


def create_frame_renderer(
    name, scene: SceneSpec, config: Config, width=960, height=540, *, isolated=True
):
    if scene.kind != "robot_motion" and (
        scene.kind != "point_reach" or not {"agent", "target"} <= set(scene.entity_ids)
    ):
        raise ValueError("Frame adapters require the point_reach agent/target scene")
    if any(
        type(value) is not int or not 64 <= value <= 1920 for value in (width, height)
    ):
        raise ValueError("Frame dimensions must be integers in 64..1920")
    report = frame_dependencies(name, scene)
    if not report["supported"]:
        raise ValueError("Raster does not support robot meshes; select mujoco/gl/rtx")
    if not report["metadata_ok"]:
        raise ImportError(
            f"Renderer {name} dependencies unavailable; install embodiedforge[{report['install_extra']}]"
        )
    if name == "raster":
        return RasterFrames(scene, config, width, height)
    if isolated:
        return IsolatedFrames(name, scene, config, width, height)
    if scene.kind == "robot_motion":
        from .robot import RobotFrames

        return RobotFrames(name, scene, config, width, height)
    if name == "mujoco":
        return MujocoFrames(scene, config, width, height)
    return NativeFrames(name, scene, config, width, height)
