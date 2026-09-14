"""Optional browser viewer consumes public snapshots and releases its server."""

import socket
import urllib.request

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.visualization import DebugViewer


def test_viewer_serves_page_and_updates_selected_environment():
    pytest.importorskip("viser")
    pytest.importorskip("PIL")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = Config(num_envs=2, render="raster", channels=("rgb",))
    with VectorEnv(config) as env:
        viewer = DebugViewer(config, "127.0.0.1", port)
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}", timeout=5
            ) as response:
                assert response.status == 200
                assert b"<html" in response.read()
            viewer.selected.value = "1"
            viewer.update(
                env.task.scene_update(env.physics.snapshot()),
                env.observe(),
                np.zeros(2),
            )
            np.testing.assert_allclose(
                viewer.agent.position[:2], env.physics.snapshot().position[1]
            )
            np.testing.assert_array_equal(viewer.rgb.image, env.observe()["rgb"][1, 0])
            assert "Episode" in viewer.stats.content
        finally:
            viewer.close()


def test_partial_viewer_initialization_stops_server(monkeypatch):
    viser = pytest.importorskip("viser")
    stopped = []

    class BrokenScene:
        def set_up_direction(self, direction):
            raise RuntimeError("scene creation failed")

    class Server:
        scene = BrokenScene()

        def stop(self):
            stopped.append(True)

    monkeypatch.setattr(viser, "ViserServer", lambda **kwargs: Server())
    with pytest.raises(RuntimeError, match="scene creation failed"):
        DebugViewer(Config(), "127.0.0.1", 8080)
    assert stopped == [True]
