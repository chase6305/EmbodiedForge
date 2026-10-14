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
        # Squared distances are separable: broadcast two axes only when adding.
        # This avoids computing each axis's difference/square S times per frame.
        self.x, self.y = coord[None, :], -coord[:, None]
        self.palette = np.array(
            [[20, 24, 32], [70, 155, 255], [65, 210, 120]], dtype=np.uint8
        )

    def _draw_disk(self, labels: Array, positions: Array, radius: float, entity: int):
        size = self.config.image_size
        width = min(size, int(np.ceil(2 * min(radius, 3) * size / 3)) + 3)
        if len(positions) * size**2 < 65536 or width * 2 >= size:
            mask = (
                (self.x[None] - positions[:, 0, None, None]) ** 2
                + (self.y[None] - positions[:, 1, None, None]) ** 2
            ) <= radius**2
            labels[mask] = entity
            return
        # Bound the work to each disk's pixel rectangle. Expand by a pixel on
        # either side before the original distance test, including tangent
        # pixels despite rounding in the rectangle bounds. Large disks and
        # small images use the cheaper full-frame broadcast above.
        coord = self.x[0]
        bounds = np.asarray(
            positions, dtype=np.result_type(positions.dtype, np.float64)
        )
        offsets = np.arange(width)
        x0 = np.clip(np.searchsorted(coord, bounds[:, 0] - radius) - 1, 0, size - width)
        y0 = np.clip(
            np.searchsorted(coord, -bounds[:, 1] - radius) - 1, 0, size - width
        )
        xs, ys = x0[:, None] + offsets, y0[:, None] + offsets
        mask = (
            (coord[xs][:, None, :] - positions[:, 0, None, None]) ** 2
            + (-coord[ys][:, :, None] - positions[:, 1, None, None]) ** 2
        ) <= radius**2
        row, y, x = np.nonzero(mask)
        labels[row, ys[row, y], xs[row, x]] = entity

    def render(self, ids: Array) -> RenderBatch:
        s = self.config.image_size
        labels = np.zeros((len(ids), 1, s, s), dtype=np.int32)
        for entity, positions, radius in (
            (2, self.entities["target"], 0.10),
            (1, self.entities["agent"], self.scene.radius),
        ):
            self._draw_disk(labels[:, 0], positions[ids], radius, entity)
        if self.config.channels == ("rgb",):
            # Release the int32 labels before take allocates RGB. Otherwise
            # its intp index conversion stays live alongside both arrays.
            labels = labels.astype(np.intp, copy=False)
        # Labels are constructed here and always in [0, 2]. take copies the
        # three-byte palette entries faster than general advanced indexing.
        values = {
            "rgb": lambda: np.take(self.palette, labels, axis=0, mode="clip"),
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
