"""Newton GL / OVRTX previews driven by portable scene snapshots.

Only two point batches are submitted; this adapter never reads a physics SDK's
private model/state. Native model inspection, picking forces and material asset
translation require a later scene bridge and are not provided by this adapter.
"""

import threading

import numpy as np

from embodiedforge.backends import NEWTON_VERSION
from embodiedforge.core import Array, Observation, SceneSpec, SceneUpdate


class NewtonViewer:
    """Own one native viewer; render a fixed selected environment as spheres."""

    def __init__(
        self,
        backend: str,
        scene: SceneSpec,
        *,
        env_id: int,
        headless: bool,
        width: int = 1280,
        height: int = 720,
        synchronous: bool = False,
    ) -> None:
        if backend not in ("gl", "rtx"):
            raise ValueError(f"Unsupported native viewer: {backend}")
        try:
            import newton
            import newton.viewer
            import warp as wp
        except ImportError as error:
            raise ImportError(
                "Native previews require embodiedforge[newton]"
            ) from error
        if newton.__version__ != NEWTON_VERSION:
            raise RuntimeError(f"Native viewers require Newton {NEWTON_VERSION}")
        self.closed = False
        self.backend = backend
        self.headless = headless
        self.selected_env = env_id
        self.reset_requested = threading.Event()
        self.radius = scene.radius
        self.viewer = None
        self.cleanup_error: Exception | None = None
        try:
            # Scene buffers live on CPU; RTX presentation still uses CUDA/GL interop.
            with wp.ScopedDevice("cpu"):
                if backend == "gl":
                    self.viewer = newton.viewer.ViewerGL(
                        width=width,
                        height=height,
                        headless=headless,
                        enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE,
                    )
                    from .style import SKY_HORIZON, SKY_TOP

                    if hasattr(self.viewer, "renderer"):
                        self.viewer.renderer.sky_upper = SKY_TOP
                        self.viewer.renderer.sky_lower = SKY_HORIZON
                        self.viewer.renderer._light_color = (1.0, 0.96, 0.90)
                        self.viewer.renderer._sun_direction = np.array(
                            (-0.4, -0.6, 0.7), dtype=np.float32
                        )
                else:
                    from .rtx import RtxColorOutputMixin

                    class CompatibleViewerRTX(
                        RtxColorOutputMixin, newton.viewer.ViewerRTX
                    ):
                        pass

                    self.viewer = CompatibleViewerRTX(
                        width=width,
                        height=height,
                        headless=headless,
                        up_axis="Z",
                        environment="studio",
                        **({"async_rendering": False} if synchronous else {}),
                    )
                self.agent = wp.zeros(1, dtype=wp.vec3, device="cpu")
                self.target = wp.zeros(1, dtype=wp.vec3, device="cpu")
                self.agent_color = wp.array(
                    [(0.27, 0.61, 1.0)], dtype=wp.vec3, device="cpu"
                )
                self.target_color = wp.array(
                    [(0.25, 0.82, 0.47)], dtype=wp.vec3, device="cpu"
                )
            self.viewer.set_camera(wp.vec3(0.0, -3.0, 3.0), pitch=-45.0, yaw=90.0)
            self.viewer.set_reset_callback(self.reset_requested.set)
            self._ground_added = False
        except BaseException:
            try:
                self.close()
            except Exception as error:
                self.cleanup_error = error
            raise

    def is_running(self) -> bool:
        """Pump native input each control tick, independently of render FPS."""
        if self.closed or not self.viewer.is_running():
            return False
        if not self.headless:
            # Window access is tied to the checked Newton version. RTX creates
            # its window lazily during the initial frame.
            window = (
                self.viewer.renderer.window
                if self.backend == "gl"
                else self.viewer._window
            )
            if window is not None:
                window.switch_to()
                window.dispatch_events()
        return self.viewer.is_running()

    def should_step(self) -> bool:
        """Use Newton's native pause/single-step controls without nested loops."""
        return self.viewer.should_step()

    def consume_reset(self) -> int | None:
        if self.reset_requested.is_set():
            self.reset_requested.clear()
            return self.selected_env
        return None

    def update(
        self, scene: SceneUpdate, observation: Observation, reward: Array
    ) -> None:
        """Upload only selected point positions; preserve the viewer's frame lifecycle."""
        if self.closed:
            raise RuntimeError("Viewer is closed")
        index = self.selected_env
        for name, buffer, radius in (
            ("agent", self.agent, self.radius),
            ("target", self.target, 0.1),
        ):
            xy = scene.entities[name][index]
            buffer.assign(np.array([[xy[0], xy[1], radius]], dtype=np.float32))
        self.viewer.begin_frame(float(observation["time"][index]))
        if not self._ground_added:
            import newton
            import warp as wp

            from .style import floor_texture

            ground = newton.Mesh.create_plane(200, 200, compute_inertia=False)
            vertices = np.asarray(ground.vertices)
            self.viewer.log_mesh(
                "/stage/floor",
                wp.array(vertices, dtype=wp.vec3, device="cpu"),
                wp.array(ground.indices, dtype=wp.int32, device="cpu"),
                uvs=wp.array(vertices[:, :2], dtype=wp.vec2, device="cpu"),
                texture=floor_texture(),
                roughness=0.85,
                metallic=0.0,
            )
            self._ground_added = True
        self.viewer.log_points(
            "/agent", self.agent, radii=self.radius, colors=self.agent_color
        )
        self.viewer.log_points(
            "/target", self.target, radii=0.1, colors=self.target_color
        )
        self.viewer.end_frame()

    def close(self) -> None:
        """Idempotent cleanup, including failures after native construction."""
        if not self.closed:
            self.closed = True
            if self.viewer is not None:
                self.viewer.close()
