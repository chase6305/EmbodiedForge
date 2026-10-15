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


def _valid_depth(depth: Array) -> bool:
    """Keep validation scratch bounded for large, plain numeric image batches."""
    if type(depth) is not np.ndarray or depth.size <= 4 * 1024 * 1024:
        return bool(np.isfinite(depth).all() and not (depth < 0).any())
    with np.nditer(
        depth,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=256 * 1024,
    ) as chunks:
        return all(
            np.isfinite(chunk).all() and not (chunk < 0).any() for chunk in chunks
        )


def _matching_depth_validity(depth: Array, valid: Array) -> bool:
    if (
        type(depth) is not np.ndarray
        or type(valid) is not np.ndarray
        or depth.size <= 4 * 1024 * 1024
    ):
        return np.array_equal(depth > 0, valid)
    with np.nditer(
        [depth, valid],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        buffersize=256 * 1024,
    ) as chunks:
        return all(np.array_equal(values > 0, mask) for values, mask in chunks)


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
            if name == "depth" and not _valid_depth(values):
                raise ValueError("Depth must be finite and nonnegative")
        if "depth" in batch.images:
            if not _matching_depth_validity(
                batch.images["depth"], batch.images["depth_valid"]
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

    def observe(self, time: Array, *, env_ids: Array | None = None) -> Observation:
        """Return owned selected images/timestamps; IDs are validated by VectorEnv."""
        if not self.images:
            return {}
        if env_ids is not None:
            # Integer row indexing already owns its output. Select before
            # copying instead of copying every cached image for a partial reset.
            return {
                **{name: value[env_ids] for name, value in self.images.items()},
                "frame_time": self.frame_time[env_ids],
                "frame_version": self.frame_version[env_ids],
                "frame_age": time[env_ids] - self.frame_time[env_ids],
            }
        return {
            **{name: value.copy() for name, value in self.images.items()},
            "frame_time": self.frame_time.copy(),
            "frame_version": self.frame_version.copy(),
            "frame_age": time - self.frame_time,
        }
