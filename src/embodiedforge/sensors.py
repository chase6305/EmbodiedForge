"""Synchronous sensor scheduling, output validation and frame-buffer ownership."""

from copy import deepcopy

import numpy as np

from .core import Array, Config, Observation, RenderBackend, RenderBatch, SceneUpdate

# The CPU reference pipeline supports one camera. New modalities must define
# layout/dtype semantics here before adapters advertise support for them.
CHANNEL_DTYPES = {
    "rgb": "uint8",
    "depth": "float32",
    "depth_valid": "bool",
    "instance_id": "int32",
    "semantic_id": "int32",
}


class SensorPipeline:
    """Own camera caches, independent of task state and physics stepping.

    sample borrows a SceneUpdate until it returns; cached outputs are copied.
    observe returns owned arrays. No channels means no sync/render submission.
    """

    def __init__(self, config: Config, renderer: RenderBackend) -> None:
        self.config, self.renderer = config, renderer
        self.images: Observation = {}
        self.frame_time = np.zeros(config.num_envs)
        self.frame_version = np.zeros(config.num_envs, dtype=np.int64)
        self.calibration: dict = {}

    def due(self, steps: Array, active: Array, done: Array) -> Array:
        """Return due environment IDs, always including newly finished episodes."""
        interval = self.config.control_hz // self.config.camera_hz
        return np.flatnonzero(active & ((steps % interval == 0) | done))

    def _validate(self, batch: RenderBatch, ids: Array, update: SceneUpdate) -> None:
        state = update.state
        if (
            batch.time.shape != (len(ids),)
            or batch.version.shape != (len(ids),)
            or not np.array_equal(batch.time, state.time[ids])
            or not np.array_equal(batch.version, state.version[ids])
        ):
            raise ValueError(
                "Renderer returned a stale or mismatched state version/time"
            )
        required = set(self.config.channels)
        if "depth" in required:
            required.add("depth_valid")
        if batch.images.keys() != required:
            raise ValueError("Renderer output channels do not match the request")
        for name, values in batch.images.items():
            shape = (len(ids), 1, self.config.image_size, self.config.image_size)
            if name == "rgb":
                shape += (3,)
            dtype = CHANNEL_DTYPES.get(name)
            if (
                dtype is None
                or values.shape != shape
                or values.dtype != np.dtype(dtype)
            ):
                raise ValueError(f"Invalid renderer layout/dtype for {name}")
            if name == "depth" and (
                not np.isfinite(values).all() or (values < 0).any()
            ):
                raise ValueError("Depth must be finite and nonnegative")
        if "depth" in batch.images:
            if not np.array_equal(
                batch.images["depth"] > 0, batch.images["depth_valid"]
            ):
                raise ValueError("Depth validity must match positive depth pixels")

    def sample(self, ids: Array, update: SceneUpdate) -> None:
        """Render selected rows; validate the whole batch before updating caches."""
        if len(ids) == 0 or not self.config.channels:
            return
        self.renderer.sync(update)
        batch = self.renderer.render(ids)
        self._validate(batch, ids, update)
        for name, values in batch.images.items():
            if name not in self.images:
                self.images[name] = np.zeros(
                    (self.config.num_envs, *values.shape[1:]), dtype=values.dtype
                )
            self.images[name][ids] = values
        self.frame_time[ids], self.frame_version[ids] = batch.time, batch.version
        self.calibration = deepcopy(batch.calibration)

    def observe(self, time: Array) -> Observation:
        """Return owned images and timestamps; low-rate frames report their age."""
        if not self.images:
            return {}
        return {
            **{name: value.copy() for name, value in self.images.items()},
            "frame_time": self.frame_time.copy(),
            "frame_version": self.frame_version.copy(),
            "frame_age": time - self.frame_time,
        }
