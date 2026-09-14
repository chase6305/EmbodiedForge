"""Video resource ownership and interruption behavior, without a GL context."""

import io
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge import _microduck_worker as worker


@pytest.fixture
def recorder(tmp_path, monkeypatch):
    events = []
    path = tmp_path / "policy.mp4"
    process = SimpleNamespace(stdin=io.BytesIO(), returncode=None)

    def wait(timeout=None):
        events.append("encoder_wait")
        process.returncode = 0
        path.write_bytes(b"encoded-video")
        return 0

    process.wait = wait
    process.poll = lambda: process.returncode

    def kill():
        events.append("encoder_kill")
        process.returncode = -9

    process.kill = kill

    class Renderer:
        def __init__(self, model, cfg, scene):
            assert cfg.env_idx == 0 and cfg.max_extra_envs == 0

        def initialize(self):
            events.append("renderer_init")

        def update(self, data):
            events.append("capture")

        def render(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def close(self):
            events.append("renderer_close")

    class Config(SimpleNamespace):
        OriginType = SimpleNamespace(ASSET_ROOT="root")
        env_idx = 0

    monkeypatch.setitem(
        sys.modules, "imageio_ffmpeg", SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg")
    )
    monkeypatch.setitem(
        sys.modules,
        "mjlab.viewer.offscreen_renderer",
        SimpleNamespace(OffscreenRenderer=Renderer),
    )
    monkeypatch.setitem(
        sys.modules, "mjlab.viewer.viewer_config", SimpleNamespace(ViewerConfig=Config)
    )
    monkeypatch.setitem(
        sys.modules,
        "OpenGL",
        SimpleNamespace(
            GL=SimpleNamespace(
                GL_VENDOR=0,
                GL_RENDERER=1,
                GL_VERSION=2,
                glGetString=lambda token: (
                    b"test-vendor",
                    b"test-renderer",
                    b"test-version",
                )[token],
            )
        ),
    )

    def popen(command, **kwargs):
        assert "-n" in command and command[-1] == str(path)
        events.append("encoder_start")
        return process

    monkeypatch.setattr(worker.subprocess, "Popen", popen)
    env = SimpleNamespace(
        step_dt=0.02, sim=SimpleNamespace(mj_model=None, data=None), scene=None
    )
    return env, path, process, Renderer, events


def test_disabled_video_needs_no_environment_or_renderer():
    with worker.evaluation_video(None, None) as (capture, info):
        capture()
        assert info is None


def test_video_streams_frames_and_finalizes_before_renderer(recorder):
    env, path, process, _, events = recorder
    with worker.evaluation_video(env, path) as (capture, info):
        capture()
        capture()
        assert process.stdin.tell() == 2 * 480 * 640 * 3
    assert info["frames"] == 2 and info["fps"] == 50
    assert len(info["sha256"]) == 64
    assert process.stdin.closed
    assert events[-2:] == ["encoder_wait", "renderer_close"]


def test_video_interrupt_preserves_exception_and_closes_resources(recorder):
    env, path, process, _, events = recorder
    with pytest.raises(KeyboardInterrupt):
        with worker.evaluation_video(env, path) as (capture, info):
            capture()
            raise KeyboardInterrupt
    assert process.stdin.closed and events[-1] == "renderer_close"
    assert "sha256" not in info


def test_renderer_initialization_failure_still_closes(recorder, monkeypatch):
    env, path, _, renderer, events = recorder

    def fail(self):
        raise RuntimeError("GL initialization failed")

    monkeypatch.setattr(renderer, "initialize", fail)
    with pytest.raises(RuntimeError, match="GL initialization"):
        with worker.evaluation_video(env, path):
            pass
    assert events == ["renderer_close"]


def test_stalled_encoder_is_killed_and_renderer_closed(recorder):
    env, path, process, _, events = recorder
    original_wait = process.wait

    def wait(timeout=None):
        if timeout is not None:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return original_wait()

    process.wait = wait
    with pytest.raises(subprocess.TimeoutExpired):
        with worker.evaluation_video(env, path) as (capture, _):
            capture()
    assert "encoder_kill" in events and events[-1] == "renderer_close"


def test_existing_video_is_not_overwritten(recorder):
    env, path, _, _, events = recorder
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        with worker.evaluation_video(env, path):
            pass
    assert path.read_bytes() == b"keep" and events == []


@pytest.mark.parametrize("interrupted", [False, True])
def test_encoder_signal_exit_preserves_keyboard_interrupt_only(recorder, interrupted):
    env, path, process, _, events = recorder

    def wait(timeout=None):
        process.returncode = 255
        return 255

    process.wait = wait
    expected = KeyboardInterrupt if interrupted else RuntimeError
    with pytest.raises(expected):
        with worker.evaluation_video(env, path) as (capture, _):
            capture()
            if interrupted:
                raise KeyboardInterrupt
    assert process.stdin.closed and events[-1] == "renderer_close"
