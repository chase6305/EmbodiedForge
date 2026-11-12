"""Opt-in, read-only preview of Wuji's actual training rollout, environment 0.

Rendering stays on the training thread; HTTP threads only read encoded images.
No renderer, HTTP service, or optional imports are created in headless training.
"""

import logging
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .viewers.training import MujocoTrainingPreview

LOGGER = logging.getLogger(__name__)


class TrainingPreview(MujocoTrainingPreview):
    """Adapt UniSim's model and snapshot layout to the shared frame preview."""

    title = "Wuji training preview"

    def _initialize_renderer(self, env):
        import mujoco
        from unisim.backend.mujoco.playback import resolve_render_play_model_files

        with TemporaryDirectory(prefix="wuji-training-preview-") as directory:
            files = resolve_render_play_model_files(env, num_envs=1, tmp_dir=directory)
            path = files if isinstance(files, str) else files[0]
            self.model = (
                mujoco.MjModel.from_binary_path(str(path))
                if Path(path).suffix == ".mjb"
                else mujoco.MjModel.from_xml_path(str(path))
            )
        self.data = mujoco.MjData(self.model)
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 0.65
        self.camera.azimuth = 135
        self.camera.elevation = -30
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def snapshot(self, env):
        import numpy as np

        # UniSim's public snapshot contract includes mocap poses when present.
        return np.asarray(env.get_physics_state_snapshot())[0]

    def aim_camera(self):
        import mujoco
        import numpy as np

        palm = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "robot/right_palm_link"
        )
        if palm >= 0:
            self.camera.lookat[:] = self.data.xpos[palm] + np.array([0.0, 0.0, 0.06])


@contextmanager
def training_preview(request):
    from wuji_unilab.rl.runtime import WujiWrapper

    # The upstream resolver returns this class directly. Restore even on errors.
    inherited = "step" not in WujiWrapper.__dict__
    original = WujiWrapper.step
    preview = TrainingPreview(request["viewer_port"], request["viewer_fps"])

    def step(wrapper, actions):
        result = original(wrapper, actions)
        preview.update(wrapper.env)
        return result

    WujiWrapper.step = step
    try:
        try:
            yield preview
        finally:
            if inherited:
                del WujiWrapper.step
            else:
                WujiWrapper.step = original
    except BaseException:
        try:
            preview.close()
        except BaseException:
            LOGGER.exception("Training preview cleanup after training failure")
        raise
    else:
        preview.close()
