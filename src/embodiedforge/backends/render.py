"""Independent CPU orthographic renderer for the portable point-reach scene."""

import numpy as np

from embodiedforge.core import (
    Array,
    Capabilities,
    Config,
    RenderBatch,
    SceneSpec,
    SceneUpdate,
)


class NullRenderer:
    """No image output; useful when only state observations are requested."""

    capabilities = Capabilities()

    def build(self, scene: SceneSpec, config: Config) -> None:
        self.config, self.scene = config, scene

    def sync(self, update: SceneUpdate) -> None:
        self.state = update.state
        self.entities = update.entities

    def render(self, ids: Array) -> RenderBatch:
        return RenderBatch(
            {}, self.state.time[ids].copy(), self.state.version[ids].copy()
        )

    def close(self) -> None:
        pass


class RasterRenderer(NullRenderer):
    """Debug planar disks, not photorealistic robot training imagery.

    Orthographic top-down camera at z=2. Objects are represented as flat disks
    on z=0; depth is 2m on foreground, 0 on background. IDs: agent=1, target=2.
    """

    capabilities = Capabilities(
        channels=frozenset({"rgb", "depth", "instance_id", "semantic_id"}),
        required_entities=frozenset({"agent", "target"}),
    )

    def build(self, scene: SceneSpec, config: Config) -> None:
        super().build(scene, config)
        s = config.image_size
        coord = (np.arange(s) + 0.5) * 3 / s - 1.5
        self.x, self.y = np.meshgrid(coord, -coord)

    def render(self, ids: Array) -> RenderBatch:
        s = self.config.image_size
        labels = np.zeros((len(ids), 1, s, s), dtype=np.int32)
        for entity, positions, radius in (
            (2, self.entities["target"], 0.10),
            (1, self.entities["agent"], self.scene.radius),
        ):
            p = positions[ids]
            mask = (
                (self.x[None] - p[:, 0, None, None]) ** 2
                + (self.y[None] - p[:, 1, None, None]) ** 2
            ) <= radius**2
            labels[:, 0][mask] = entity
        palette = np.array(
            [[20, 24, 32], [70, 155, 255], [65, 210, 120]], dtype=np.uint8
        )
        values = {
            "rgb": lambda: palette[labels],
            "depth": lambda: np.where(labels != 0, 2.0, 0.0).astype(np.float32),
            "instance_id": lambda: labels.copy(),
            "semantic_id": lambda: labels.copy(),
        }
        images = {channel: values[channel]() for channel in self.config.channels}
        if "depth" in images:
            images["depth_valid"] = labels != 0
        return RenderBatch(
            images,
            self.state.time[ids].copy(),
            self.state.version[ids].copy(),
            {
                "model": "orthographic",
                "bounds_xy": [-1.5, 1.5, -1.5, 1.5],
                "meters_per_pixel": 3 / s,
                "depth": "optical_z_meters",
                "T_world_camera": [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 2],
                    [0, 0, 0, 1],
                ],
                "ids": {"background": 0, "agent": 1, "target": 2},
            },
        )
