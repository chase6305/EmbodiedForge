"""Read-only browser preview for native and Light Loco Go1 training."""

from contextlib import contextmanager

import numpy as np

from ._wuji_training_viewer import TrainingPreview


class Go1Preview(TrainingPreview):
    title = "Go1 training preview"

    def _initialize_renderer(self, env):
        import mujoco

        self.model = env.batch.model
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 2.0
        self.camera.azimuth = 135
        self.camera.elevation = -25
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def snapshot(self, env):
        return np.r_[env.clock[0], env.qpos[0], env.qvel[0]]

    def aim_camera(self):
        self.camera.lookat[:] = self.data.qpos[:3]


@contextmanager
def training_preview(env, request):
    preview = Go1Preview(request["viewer_port"], request["viewer_fps"])
    original = env.step

    def step(actions):
        result = original(actions)
        preview.update(env)
        return result

    env.step = step
    try:
        yield preview
    finally:
        del env.step
        preview.close()
