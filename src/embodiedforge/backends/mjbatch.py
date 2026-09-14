"""CPU thread-pool dynamics for the portable planar scene."""

import numpy as np

from embodiedforge.core import Array, Config, SceneSpec

from .physics import NumpyPhysics, planar_model


class MjbatchPhysics(NumpyPhysics):
    """Own a native batch; bound arrays never escape through snapshots."""

    def __init__(self, num_threads: int = 0):
        self.num_threads = num_threads

    def build(self, scene: SceneSpec, config: Config) -> None:
        try:
            import mujoco
            from mjbatch import Batch
        except ImportError as exc:
            raise ImportError(
                "Install embodiedforge[mjbatch] to use physics=mjbatch"
            ) from exc
        super().build(scene, config)
        self.batch = Batch(
            planar_model(mujoco, scene, config),
            config.num_envs,
            num_threads=self.num_threads,
        )
        self.position = self.batch.bind("qpos")
        self.velocity = self.batch.bind("qvel")
        self.control = self.batch.bind("ctrl")
        self.time = self.batch.bind("time").reshape(-1)
        self.timestep = self.batch.expand("timestep")

    def reset(self, ids: Array, position: Array) -> None:
        if not len(ids):
            return
        # mjbatch requires sorted IDs. Preserve the caller's row-to-position map.
        order = np.argsort(ids)
        selected = np.ascontiguousarray(ids[order], dtype=np.int64)
        self.batch.reset(selected)
        self.position[selected] = position[order]
        self.velocity[selected] = 0
        self.control[selected] = 0
        self.time[selected] = 0
        self.batch.forward(selected)
        self.version[selected] += 1

    def step(self, dt: float, substeps: int, active: Array) -> None:
        if not active.any():
            return
        self.timestep[:] = dt / substeps
        self.batch.step(np.ascontiguousarray(active, dtype=bool), nstep=substeps)
        self.version[active] += 1

    def close(self) -> None:
        # Bound NumPy views keep their owner alive, so release them as well.
        self.position = self.velocity = self.control = self.time = None
        self.timestep = self.batch = None
