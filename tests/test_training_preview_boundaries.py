"""Robot previews depend on a shared MuJoCo frame owner, not another robot."""

import os
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from embodiedforge.viewers import training


def test_preview_adapters_import_without_robot_or_render_sdks():
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        forbidden = {
            'mujoco', 'PIL', 'torch', 'unisim', 'wuji_unilab',
            'embodiedforge._wuji_training_viewer',
        }
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.')
                       for name in forbidden):
                    raise RuntimeError('Forbidden preview dependency: ' + fullname)
        sys.meta_path.insert(0, Block())
        from embodiedforge._go1_training_viewer import Go1Preview
        from embodiedforge.viewers.training import MujocoTrainingPreview
        assert issubclass(Go1Preview, MujocoTrainingPreview)
        assert 'embodiedforge._wuji_training_viewer' not in sys.modules
        forbidden.remove('embodiedforge._wuji_training_viewer')
        from embodiedforge._wuji_training_viewer import TrainingPreview
        assert issubclass(TrainingPreview, MujocoTrainingPreview)
        assert TrainingPreview.__module__ == 'embodiedforge._wuji_training_viewer'
        assert not any(name == prefix or name.startswith(prefix + '.')
                       for name in sys.modules for prefix in forbidden)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("invalid", [None, "shape", "nonfinite"])
def test_shared_preview_restores_mocap_snapshot_before_rendering(monkeypatch, invalid):
    row = np.arange(12, dtype=np.float64)
    if invalid == "shape":
        row = row[:-1]
    elif invalid == "nonfinite":
        row[-1] = np.nan
    before = row.copy()
    events = []
    mujoco = ModuleType("mujoco")
    mujoco.mj_forward = lambda model, data: events.append("forward")
    pillow = ModuleType("PIL")
    pillow.Image = SimpleNamespace(
        fromarray=lambda pixels: SimpleNamespace(
            save=lambda output, **kwargs: output.write(b"encoded-jpeg")
        )
    )
    monkeypatch.setitem(sys.modules, "mujoco", mujoco)
    monkeypatch.setitem(sys.modules, "PIL", pillow)
    monkeypatch.setattr(training, "monotonic", lambda: 5.0)
    preview = training.MujocoTrainingPreview.__new__(training.MujocoTrainingPreview)
    preview.steps, preview.next_frame, preview.period = 0, 0.0, 0.1
    preview.frame = (b"previous-jpeg", 0)
    preview.model = SimpleNamespace(nq=2, nv=2, nmocap=1)
    preview.data = SimpleNamespace(
        time=-1,
        qpos=np.full(2, -1.0),
        qvel=np.full(2, -1.0),
        mocap_pos=np.full((1, 3), -1.0),
        mocap_quat=np.full((1, 4), -1.0),
    )
    preview.camera = object()
    preview.snapshot = lambda env: row
    preview.aim_camera = lambda: events.append("aim")
    preview.renderer = SimpleNamespace(
        update_scene=lambda data, **kwargs: events.append("scene"),
        render=lambda: events.append("render"),
    )
    if invalid:
        message = "snapshot shape" if invalid == "shape" else "Non-finite"
        with pytest.raises(ValueError, match=message):
            preview.update(object())
        assert not events
        assert preview.data.time == -1
        assert (preview.data.qpos == -1).all()
        assert preview.frame == (b"previous-jpeg", 0)
    else:
        preview.update(object())
        np.testing.assert_array_equal(preview.data.qpos, row[1:3])
        np.testing.assert_array_equal(preview.data.qvel, row[3:5])
        np.testing.assert_array_equal(preview.data.mocap_pos, row[5:8].reshape(1, 3))
        np.testing.assert_array_equal(preview.data.mocap_quat, row[8:].reshape(1, 4))
        assert preview.data.time == row[0]
        assert events == ["forward", "aim", "scene", "render"]
        assert preview.frame == (b"encoded-jpeg", 1)
        assert preview.next_frame == 5.1
    np.testing.assert_array_equal(row, before)
