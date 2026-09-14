"""Viser browser adapter; GUI callbacks request main-thread simulation changes."""

import threading

import numpy as np

from embodiedforge.core import Config, Observation, SceneUpdate


class DebugViewer:
    """Own Viser resources; consume scene/observation snapshots, never step physics.

    Only the selected row is sent to the browser. This reference viewer supports
    the planar agent/target scene and one RGB camera, not general robot assets.
    """

    def __init__(
        self, config: Config, host: str, port: int, *, radius: float = 0.06
    ) -> None:
        try:
            import viser
        except ImportError as error:
            raise ImportError(
                "Install the viewer with pip install -e '.[viz]'"
            ) from error

        self.closed = False
        self.radius = radius
        self.reset_requested = threading.Event()
        self.step_requested = threading.Event()
        self.server = viser.ViserServer(host=host, port=port, label="EmbodiedForge")
        try:
            self.server.scene.set_up_direction("+z")
            self.server.scene.add_grid("/ground", width=3, height=3, cell_size=0.25)
            self.agent = self.server.scene.add_icosphere(
                "/agent",
                radius=radius,
                color=(70, 155, 255),
            )
            self.target = self.server.scene.add_icosphere(
                "/target",
                radius=0.1,
                color=(65, 210, 120),
                wireframe=True,
            )
            self.server.gui.add_markdown(
                "## EmbodiedForge 调试\n蓝色：受控物体 · 绿色：目标\n\n"
                f"物理：**{config.physics}** · 任务：**{config.task}**"
            )
            self.selected = self.server.gui.add_dropdown(
                "环境",
                options=[str(i) for i in range(config.num_envs)],
                initial_value="0",
            )
            self.paused = self.server.gui.add_checkbox("暂停", initial_value=False)
            step = self.server.gui.add_button("单步（暂停时）")
            reset = self.server.gui.add_button("重置选中环境")
            step.on_click(lambda _: self.step_requested.set())
            reset.on_click(lambda _: self.reset_requested.set())
            self.stats = self.server.gui.add_markdown("等待仿真状态…")
            self.rgb = self.server.gui.add_image(
                np.zeros((config.image_size, config.image_size, 3), dtype=np.uint8),
                label="RGB · 正交调试相机",
            )

            @self.server.on_client_connect
            def set_camera(client):
                client.camera.position = (0.0, -2.8, 3.2)
                client.camera.look_at = (0.0, 0.0, 0.0)
                client.camera.up_direction = (0.0, 0.0, 1.0)
        except BaseException:
            self.server.stop()
            raise

    def update(
        self, scene: SceneUpdate, observation: Observation, reward: np.ndarray
    ) -> None:
        """Publish one row atomically; image age distinguishes sensor and state time."""
        index = int(self.selected.value)
        position = scene.entities["agent"][index]
        target = scene.entities["target"][index]
        with self.server.atomic():
            self.agent.position = (float(position[0]), float(position[1]), self.radius)
            self.target.position = (float(target[0]), float(target[1]), 0.06)
            self.rgb.image = observation["rgb"][index, 0]
            self.stats.content = (
                f"Episode **{observation['episode_id'][index]}** · "
                f"Step **{observation['step_id'][index]}**\n\n"
                f"时间：{observation['time'][index]:.2f} s · "
                f"图像帧龄：{observation['frame_age'][index]:.2f} s\n\n"
                f"位置：({position[0]:.3f}, {position[1]:.3f})\n\n"
                f"速度：{np.linalg.norm(scene.state.velocity[index]):.3f} m/s · "
                f"Reward：{reward[index]:.4f}"
            )

    @property
    def selected_env(self) -> int:
        """Global environment ID selected in the browser."""
        return int(self.selected.value)

    def consume_reset(self) -> int | None:
        """Consume a reset request without modifying the environment."""
        if self.reset_requested.is_set():
            self.reset_requested.clear()
            return self.selected_env
        return None

    def should_step(self) -> bool:
        """Consume at most one single-step request."""
        single_step = self.step_requested.is_set()
        self.step_requested.clear()
        return not self.paused.value or single_step

    def is_running(self) -> bool:
        """The browser server runs until its owner closes it."""
        return not self.closed

    def close(self) -> None:
        """Stop the HTTP/WebSocket server and release its background threads."""
        if not self.closed:
            self.closed = True
            self.server.stop()
