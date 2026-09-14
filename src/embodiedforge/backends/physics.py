"""Two implementations of the same unit-mass, planar force-controlled scene."""

import numpy as np

from embodiedforge.core import (
    Array,
    Capabilities,
    Config,
    ControlSpec,
    SceneSpec,
    StateSnapshot,
)


class NumpyPhysics:
    """Analytic reference dynamics; not a general robot simulator."""

    capabilities = Capabilities(control=ControlSpec(2, "newtons_xy"))

    def build(self, scene: SceneSpec, config: Config) -> None:
        """Allocate CPU arrays for the portable planar scene."""
        self.scene = scene
        self.position = np.zeros((config.num_envs, 2), dtype=np.float64)
        self.velocity = np.zeros_like(self.position)
        self.control = np.zeros_like(self.position)
        self.time = np.zeros(config.num_envs)
        self.version = np.zeros(config.num_envs, dtype=np.int64)

    def reset(self, ids: Array, position: Array) -> None:
        """Set selected positions; clear velocity/control/time only for those rows."""
        self.position[ids] = position
        self.velocity[ids] = 0
        self.control[ids] = 0
        self.time[ids] = 0
        self.version[ids] += 1

    def apply_control(self, actions: Array) -> None:
        """Copy XY forces [N,2] in newtons into the control buffer."""
        self.control[:] = actions

    def step(self, dt: float, substeps: int, active: Array) -> None:
        """Semi-implicit Euler; active rows advance by a total of dt seconds."""
        h = dt / substeps
        for _ in range(substeps):
            self.velocity[active] += self.control[active] * h / self.scene.mass
            self.position[active] += self.velocity[active] * h
        self.time[active] += dt
        self.version[active] += 1

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            self.position.copy(),
            self.velocity.copy(),
            self.time.copy(),
            self.version.copy(),
        )

    def close(self) -> None:
        pass


class MujocoPhysics(NumpyPhysics):
    """Real MuJoCo dynamics, one MjData per environment, CPU sequential batch."""

    def build(self, scene: SceneSpec, config: Config) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "Install embodiedforge[mujoco] to use physics=mujoco"
            ) from exc
        self.mj = mujoco
        super().build(scene, config)
        self.model = planar_model(mujoco, scene, config)
        self.data = [mujoco.MjData(self.model) for _ in range(config.num_envs)]

    def reset(self, ids: Array, position: Array) -> None:
        super().reset(ids, position)
        for i, p in zip(ids, position, strict=True):
            d = self.data[i]
            self.mj.mj_resetData(self.model, d)
            d.qpos[:] = p
            self.mj.mj_forward(self.model, d)

    def step(self, dt: float, substeps: int, active: Array) -> None:
        self.model.opt.timestep = dt / substeps
        for i in np.flatnonzero(active):
            d = self.data[i]
            d.ctrl[:] = self.control[i]
            self.mj.mj_step(self.model, d, nstep=substeps)
            self.position[i] = d.qpos
            self.velocity[i] = d.qvel
            self.time[i] = d.time
            self.version[i] += 1

    def close(self) -> None:
        self.data = []
        self.model = None


def planar_model(mujoco, scene: SceneSpec, config: Config):
    """Shared model keeps sequential and threaded MuJoCo dynamics identical."""
    return mujoco.MjModel.from_xml_string(f"""
        <mujoco>
          <option gravity="0 0 0" integrator="Euler" timestep="{1 / config.physics_hz}"/>
          <worldbody>
            <body name="agent">
              <joint name="x" type="slide" axis="1 0 0"/>
              <joint name="y" type="slide" axis="0 1 0"/>
              <geom type="sphere" size="{scene.radius}" mass="{scene.mass}"/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="x" gear="1"/>
            <motor joint="y" gear="1"/>
          </actuator>
        </mujoco>""")
