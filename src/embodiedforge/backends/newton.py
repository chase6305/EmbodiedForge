"""Newton v1.6.0rc1 adapter for the portable planar point-mass scene.

The tagged source archive is the compatibility target. Physics is executed by
Newton's SolverSemiImplicit on Warp CPU arrays, not by the NumPy reference
integrator. This adapter deliberately advertises CPU host snapshots only.
"""

import numpy as np

from embodiedforge.core import Array, Config, SceneSpec

from . import NEWTON_SOURCE, NEWTON_VERSION
from .physics import NumpyPhysics


class NewtonPhysics(NumpyPhysics):
    """One isolated Newton particle-world per environment; XY force control.

    Reuses the CPU snapshot/clock buffers from NumpyPhysics but overrides every
    integration step. Contacts are disabled because the reference scene has no
    interacting entities. This is not yet an articulated-robot adapter.
    """

    def build(self, scene: SceneSpec, config: Config) -> None:
        """Import the exact supported SDK lazily and allocate two Newton states."""
        try:
            import newton
            import warp as wp
        except ImportError as error:
            raise ImportError(
                f"Install embodiedforge[newton] for the tagged Newton adapter: {NEWTON_SOURCE}"
            ) from error
        if newton.__version__ != NEWTON_VERSION:
            raise RuntimeError(
                f"Newton adapter requires {NEWTON_VERSION}; found {newton.__version__}"
            )
        self.wp = wp
        self.active_flag = int(newton.ParticleFlags.ACTIVE)
        super().build(scene, config)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for index in range(config.num_envs):
            builder.begin_world(label=f"env_{index}")
            builder.add_particle(
                pos=(0.0, 0.0, 0.0),
                vel=(0.0, 0.0, 0.0),
                mass=scene.mass,
                radius=scene.radius,
            )
            builder.end_world()
        self.model = builder.finalize(device="cpu")
        # No contacts exist in this task; disabling the grid also prevents
        # accidental interactions between overlapping environment instances.
        self.model.particle_grid = None
        self.solver = newton.solvers.SolverSemiImplicit(self.model)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.newton_control = self.model.control()

    def reset(self, ids: Array, position: Array) -> None:
        """Reset selected rows, synchronizing both integration buffers exactly."""
        super().reset(ids, position)
        positions = np.zeros((len(self.position), 3), dtype=np.float32)
        velocities = np.zeros_like(positions)
        positions[:, :2], velocities[:, :2] = self.position, self.velocity
        for state in (self.state_in, self.state_out):
            state.particle_q.assign(positions)
            state.particle_qd.assign(velocities)
            state.clear_forces()
        # Snapshots must reflect the float32 values actually stored by Newton.
        self.position[:] = positions[:, :2]
        self.velocity[:] = velocities[:, :2]

    def step(self, dt: float, substeps: int, active: Array) -> None:
        """Use Newton integration and freeze both position and velocity of done rows."""
        if not active.any():
            return
        flags = np.where(active, self.active_flag, 0).astype(np.int32)
        self.model.particle_flags.assign(flags)
        forces = np.zeros((len(self.position), 3), dtype=np.float32)
        forces[:, :2] = np.where(active[:, None], self.control, 0)
        for _ in range(substeps):
            self.state_in.clear_forces()
            self.state_in.particle_f.assign(forces)
            # In v1.6.0rc1 inactive-particle integration copies position only.
            # Prefill output velocity so the frozen row cannot pick up stale data.
            self.wp.copy(self.state_out.particle_qd, self.state_in.particle_qd)
            self.solver.step(
                self.state_in, self.state_out, self.newton_control, None, dt / substeps
            )
            self.state_in, self.state_out = self.state_out, self.state_in
        self.position[:] = self.state_in.particle_q.numpy()[:, :2]
        self.velocity[:] = self.state_in.particle_qd.numpy()[:, :2]
        self.time[active] += dt
        self.version[active] += 1

    def close(self) -> None:
        """Drop owned solver/state/model resources; safe after a failed build."""
        self.state_in = self.state_out = self.newton_control = None
        self.solver = self.model = None
