"""Validate the portable bridge separately from optional GL/RTX drivers."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.core import SceneSpec
from embodiedforge.viewers import create_viewer


@pytest.mark.parametrize(
    "backend,sdk_class", [("gl", "ViewerGL"), ("rtx", "ViewerRTX")]
)
def test_native_bridge_uses_selected_snapshot_and_lifecycle(
    monkeypatch, backend, sdk_class
):
    pytest.importorskip("newton")
    import newton.viewer

    captured = []

    class Native:
        class CudaInterop:
            NONE = 0

        def __init__(self, **kwargs):
            self.frames = []
            self.points = {}
            self.closed = 0
            self.meshes = {}
            if backend == "rtx":
                assert kwargs["environment"] == "studio"
            captured.append(self)

        def set_camera(self, *args, **kwargs):
            pass

        def set_reset_callback(self, callback):
            self.reset = callback

        def is_running(self):
            return True

        def should_step(self):
            return False

        def begin_frame(self, time):
            self.frames.append(time)

        def log_points(self, name, points, **kwargs):
            self.points[name] = points.numpy().copy()
            # CPU/GL presentation requires per-point Warp colors, not tuples.
            assert kwargs["colors"].numpy().shape == (len(points), 3)

        def log_mesh(self, name, points, indices, **kwargs):
            self.meshes[name] = kwargs
            assert kwargs["uvs"].numpy().shape == (len(points), 2)
            assert kwargs["texture"].shape == (128, 128, 3)

        def end_frame(self):
            self.frames.append("end")

        def close(self):
            self.closed += 1

    monkeypatch.setattr(newton.viewer, sdk_class, Native)
    with VectorEnv(Config(num_envs=2)) as env:
        viewer = create_viewer(
            backend, env.config, env.task.scene, env_id=1, headless=True
        )
        native = captured[0]
        assert viewer.is_running()
        assert not viewer.should_step()
        native.reset()
        assert viewer.consume_reset() == 1
        assert viewer.consume_reset() is None
        observation = env.observe()
        state = env.physics.snapshot()
        viewer.update(env.task.scene_update(state), observation, np.zeros(2))
        np.testing.assert_allclose(
            native.points["/agent"][0, :2], state.position[1], atol=1e-7
        )
        assert native.frames == [observation["time"][1], "end"]
        assert native.meshes["/stage/floor"]["roughness"] == 0.85
        viewer.close()
        viewer.close()
        assert native.closed == 1
        assert not viewer.is_running()


def test_native_version_guard_precedes_construction(monkeypatch):
    newton = pytest.importorskip("newton")
    monkeypatch.setattr(newton, "__version__", "1.6.0")
    with pytest.raises(RuntimeError, match="require Newton 1.6.0rc1"):
        create_viewer("gl", Config(), SceneSpec())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "unknown"},
        {"name": "gl", "env_id": -1},
        {"name": "viser", "headless": True},
    ],
)
def test_viewer_preflight(kwargs):
    with pytest.raises(ValueError):
        create_viewer(config=Config(), scene=SceneSpec(), **kwargs)


def test_viewer_dependency_report_checks_exact_newton_without_sdk(monkeypatch):
    from importlib import metadata

    from embodiedforge.viewers import viewer_dependencies

    def version(name):
        if name == "ovrtx":
            raise metadata.PackageNotFoundError(name)
        return "1.6.0" if name == "newton" else "1.0"

    monkeypatch.setattr(metadata, "version", version)
    report = viewer_dependencies("rtx")
    assert not report["metadata_ok"]
    assert report["packages"]["ovrtx"] is None
    assert report["install_extra"] == "viz-rtx"


def test_postconstruction_failure_preserves_original_and_closes(monkeypatch):
    pytest.importorskip("newton")
    import newton.viewer

    closed = []

    class Broken:
        def __init__(self, **kwargs):
            pass

        def set_camera(self, *args, **kwargs):
            raise RuntimeError("camera failure")

        def close(self):
            closed.append(True)
            raise RuntimeError("cleanup failure")

    monkeypatch.setattr(newton.viewer, "ViewerRTX", Broken)
    with pytest.raises(RuntimeError, match="camera failure"):
        create_viewer("rtx", Config(), SceneSpec())
    assert closed == [True]


def test_missing_rtx_dependency_fails_before_physics(monkeypatch, capsys):
    from embodiedforge import viewers, visualization

    monkeypatch.setattr(
        "sys.argv", ["visualize", "--viewer", "rtx", "--physics", "newton"]
    )
    monkeypatch.setattr(
        viewers,
        "viewer_dependencies",
        lambda name: {
            "metadata_ok": False,
            "packages": {"newton": "1.6.0rc1", "ovrtx": None},
            "install_extra": "viz-rtx",
        },
    )
    monkeypatch.setattr(
        visualization,
        "VectorEnv",
        lambda config: pytest.fail("must not initialize physics"),
    )
    with pytest.raises(SystemExit) as error:
        visualization.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "missing packages: ovrtx" in stderr
    assert "python -m pip install -e '.[viz-rtx]'" in stderr
    assert "Traceback" not in stderr


def test_optional_imgui_does_not_block_native_rendering(monkeypatch):
    from importlib import metadata

    from embodiedforge.viewers import require_viewer_dependencies, viewer_dependencies

    def version(name):
        if name == "imgui-bundle":
            raise metadata.PackageNotFoundError(name)
        return "1.6.0rc1" if name == "newton" else "1.0"

    monkeypatch.setattr(metadata, "version", version)
    report = viewer_dependencies("rtx")
    assert report["metadata_ok"]
    assert report["optional_packages"]["imgui-bundle"] is None
    require_viewer_dependencies("rtx")


@pytest.mark.parametrize(
    "backend,sdk_class", [("gl", "ViewerGL"), ("rtx", "ViewerRTX")]
)
def test_input_and_close_are_consumed_without_rendering(
    monkeypatch, backend, sdk_class
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    pytest.importorskip("newton")
    import newton.viewer

    class Native:
        CudaInterop = SimpleNamespace(NONE=0)

        def __init__(self, **kwargs):
            self.running = True
            self.paused = False
            self.event = None
            self._window = SimpleNamespace(
                switch_to=MagicMock(), dispatch_events=self.dispatch
            )
            self.renderer = SimpleNamespace(window=self._window)

        def dispatch(self):
            if self.event == "pause":
                self.paused = True
            elif self.event == "close":
                self.running = False
            self.event = None

        def set_camera(self, *a, **kw):
            pass

        def set_reset_callback(self, callback):
            pass

        def is_running(self):
            return self.running

        def should_step(self):
            return not self.paused

        def close(self):
            self.running = False

    monkeypatch.setattr(newton.viewer, sdk_class, Native)
    viewer = create_viewer(backend, Config(), SceneSpec(), headless=False)
    try:
        viewer.viewer.event = "pause"
        assert viewer.is_running()
        assert not viewer.should_step()
        viewer.viewer.event = "close"
        assert not viewer.is_running()
    finally:
        viewer.close()
    assert not viewer.is_running()
