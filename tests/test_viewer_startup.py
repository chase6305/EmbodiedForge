"""Startup rendering must finish before the requested run duration begins."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embodiedforge import visualization


@pytest.mark.parametrize("first_frame_fails", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_first_frame_readiness_duration_and_cleanup(
    monkeypatch, first_frame_fails, cleanup_fails
):
    clock = [0.0]
    observed_times = []
    logger = MagicMock()

    def update(scene, observation, reward):
        observed_times.append(float(observation["time"][0]))
        if len(observed_times) == 1:
            clock[0] += 60.0  # Simulate slow shader compilation.
            if first_frame_fails:
                raise RuntimeError("first frame failed")

    viewer = SimpleNamespace(
        update=update,
        close=MagicMock(),
        is_running=lambda: True,
        should_step=lambda: True,
        consume_reset=lambda: None,
    )
    if cleanup_fails:
        viewer.close.side_effect = RuntimeError("cleanup failed")
    monkeypatch.setattr(visualization, "create_viewer", lambda *a, **kw: viewer)
    monkeypatch.setattr(visualization, "require_viewer_dependencies", lambda name: None)
    monkeypatch.setattr(visualization, "setup_logging", lambda *a, **kw: None)
    monkeypatch.setattr(visualization, "auto_configure_debug_logging", lambda: False)
    monkeypatch.setattr(visualization, "get_logger", lambda *a, **kw: logger)
    monkeypatch.setattr(
        visualization,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
        ),
    )
    monkeypatch.setattr("sys.argv", ["visualize", "--viewer", "rtx", "--duration", "1"])
    if first_frame_fails:
        with pytest.raises(RuntimeError, match="first frame failed"):
            visualization.main()
        assert not any(
            c.args[0].startswith("Viewer ready") for c in logger.info.call_args_list
        )
        if cleanup_fails:
            logger.exception.assert_called_once()
    else:
        if cleanup_fails:
            with pytest.raises(RuntimeError, match="cleanup failed"):
                visualization.main()
        else:
            visualization.main()
        assert clock[0] >= 61.0
        assert len(observed_times) > 1
        assert any(
            c.args[0].startswith("Viewer ready") for c in logger.info.call_args_list
        )
    assert observed_times[0] == 0.0
    viewer.close.assert_called_once()
