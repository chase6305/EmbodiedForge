"""Exercise the shared HTTP UI against a live simulation and faulting adapters."""

import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.viewers import create_viewer, web

pytest.importorskip("PIL")
from PIL import Image


@pytest.fixture
def session():
    with VectorEnv(Config(num_envs=3)) as env:
        viewer = create_viewer(
            "web", env.config, env.task.scene, port=0, width=128, height=96
        )
        try:
            yield env, viewer
        finally:
            viewer.close()


def publish(env, viewer):
    viewer.update(
        env.task.scene_update(env.physics.snapshot()),
        env.observe(),
        np.zeros(env.config.num_envs),
    )


def request(viewer, path, command=None, token=True):
    headers = {"X-Session-Token": viewer._token} if token else {}
    payload = None if command is None else json.dumps(command).encode()
    return urlopen(Request(viewer.url + path, data=payload, headers=headers), timeout=3)


def test_http_frame_state_and_main_thread_control(session):
    env, viewer = session
    with request(viewer, "/") as response:
        page = response.read().decode()
    assert "__SESSION_TOKEN__" not in page
    assert "stream.mjpg" in page
    assert 'src="/viewer.js"' in page
    with request(viewer, "/viewer.js") as response:
        assert response.headers["Content-Type"].startswith("text/javascript")
        script = response.read().decode()
        assert "function velocityCommand" in script
        assert viewer._token not in script
    with pytest.raises(HTTPError) as exc:
        request(viewer, "/api/frame")
    assert exc.value.code == 503
    publish(env, viewer)
    with request(viewer, "/snapshot.jpg") as response:
        pixels = np.asarray(Image.open(io.BytesIO(response.read())))
        assert response.headers["Content-Disposition"].startswith("attachment;")
    assert pixels.shape == (96, 96, 3)
    assert pixels.std() > 5
    with request(viewer, "/api/state") as response:
        state = json.load(response)
    assert state["renderer"] == "raster" and state["step"] == 0
    with request(
        viewer, "/api/control", {"action": "pause", "paused": True}
    ) as response:
        assert response.status == 202
    # HTTP threads cannot touch the simulation or the renderer.
    assert not viewer.paused
    assert viewer.is_running()
    assert viewer.paused and not viewer.should_step()
    assert env.observe()["step_id"].tolist() == [0, 0, 0]


def test_queued_steps_and_reset_capture_environment(session):
    env, viewer = session
    for command in [
        {"action": "step"},
        {"action": "step"},
        {"action": "step"},
        {"action": "reset", "env_id": 0},
        {"action": "select", "env_id": 2},
        {"action": "reset", "env_id": 0},
        {"action": "reset", "env_id": 1},
    ]:
        viewer.submit(command)
    viewer.is_running()
    assert viewer.selected_env == 2
    assert [viewer.consume_reset() for _ in range(3)] == [0, 1, None]
    for _ in range(5):
        if viewer.should_step():
            env.step(env.task.expert_action(env.observe()))
    assert env.observe()["step_id"].tolist() == [3, 3, 3]
    viewer.submit({"action": "speed", "value": 2})
    viewer.submit({"action": "pause", "paused": False})
    viewer.is_running()
    assert viewer.playback_speed == 2 and viewer.should_step()


@pytest.mark.parametrize(
    "command",
    [
        [],
        {"action": []},
        {"action": "unknown"},
        {"action": "step", "extra": 1},
        {"action": "pause", "paused": 1},
        {"action": "select", "env_id": True},
        {"action": "reset", "env_id": 3},
        {"action": "speed", "value": float("nan")},
        {"action": "speed", "value": 10**300},
        {"action": "camera", "azimuth": 0, "elevation": 91, "distance": 2},
        {"action": "renderer", "backend": "unknown"},
    ],
)
def test_invalid_http_commands_never_mutate_queue(session, command):
    _, viewer = session
    with pytest.raises(HTTPError) as exc:
        request(viewer, "/api/control", command)
    assert exc.value.code == 400
    assert not viewer._commands


def test_control_token_and_queue_limit(session):
    _, viewer = session
    with pytest.raises(HTTPError) as exc:
        request(viewer, "/api/control", {"action": "stop"}, token=False)
    assert exc.value.code == 403
    for _ in range(128):
        viewer.submit({"action": "step"})
    with pytest.raises(HTTPError) as exc:
        request(viewer, "/api/control", {"action": "step"})
    assert exc.value.code == 429
    with request(viewer, "/api/control", {"action": "stop"}) as response:
        assert response.status == 202
    assert [command["action"] for command in viewer._commands] == ["stop"]
    assert not viewer.is_running()
    assert viewer._steps == 0


def test_control_receipts_acknowledge_only_published_main_thread_state(session):
    env, viewer = session
    publish(env, viewer)
    with request(viewer, "/api/control", {"action": "select", "env_id": 2}) as response:
        first = json.load(response)["control_id"]
    with request(viewer, "/api/state") as response:
        assert json.load(response)["control_id"] < first
    with request(viewer, "/api/control", {"action": "select", "env_id": 1}) as response:
        second = json.load(response)["control_id"]
    assert second > first
    viewer.is_running()
    # Queue consumption alone does not publish a new displayed state.
    with request(viewer, "/api/state") as response:
        assert json.load(response)["control_id"] < first
    publish(env, viewer)
    with request(viewer, "/api/state") as response:
        state = json.load(response)
    assert state["env_id"] == 1 and state["control_id"] == second
    with pytest.raises(ValueError):
        viewer.submit({"action": "step", "_control_id": 9999})
    assert viewer.submit({"action": "pause", "paused": True}) == second + 1


@pytest.mark.parametrize(
    "command",
    [
        {"action": "fps", "value": 0},
        {"action": "fps", "value": 61},
        {"action": "fps", "value": True},
        {"action": "fps", "value": 29.5},
        {"action": "resolution", "width": 1921, "height": 1080},
        {"action": "resolution", "width": 64, "height": 0},
        {"action": "resolution", "width": True, "height": 540},
    ],
)
def test_invalid_display_settings_do_not_queue(session, command):
    _, viewer = session
    with pytest.raises(HTTPError) as exc:
        request(viewer, "/api/control", command)
    assert exc.value.code == 400 and not viewer._commands


def test_fps_change_publishes_without_rebuilding_paused_renderer(session, monkeypatch):
    env, viewer = session
    viewer.paused = True
    publish(env, viewer)
    original, frame_id = viewer.renderer, viewer._frame_id
    monkeypatch.setattr(
        web, "create_frame_renderer", lambda *a: pytest.fail("Unexpected rebuild")
    )
    viewer.submit({"action": "fps", "value": 60})
    viewer.is_running()
    assert viewer.update_due(0)
    publish(env, viewer)
    assert viewer._state["target_fps"] == 60 and viewer.renderer is original
    assert viewer._frame_id == frame_id and viewer._state["render_idle"]


@pytest.mark.parametrize("failure", [False, True])
def test_resolution_switch_preserves_state_and_rolls_back_on_failure(
    session, monkeypatch, failure
):
    env, viewer = session
    viewer.paused = True
    publish(env, viewer)
    original = viewer.renderer
    pose = env.physics.snapshot().position.copy()
    frame_id = viewer._frame_id
    calls = []

    class Candidate:
        camera_control = True

        def render(self, *args):
            if failure:
                raise RuntimeError("Injected framebuffer allocation failure")
            return np.zeros((360, 640, 3), dtype=np.uint8)

        def close(self):
            calls.append("close")

    def create(name, scene, config, width, height):
        calls.append((name, width, height))
        return Candidate()

    monkeypatch.setattr(web, "create_frame_renderer", create)
    viewer.submit({"action": "resolution", "width": 640, "height": 360})
    viewer.is_running()
    assert viewer.update_due(0)
    publish(env, viewer)
    np.testing.assert_array_equal(pose, env.physics.snapshot().position)
    assert calls[0] == ("raster", 640, 360)
    if failure:
        assert viewer.renderer is original and viewer._frame_id == frame_id
        assert (viewer.width, viewer.height) == (128, 96)
        assert viewer._state["error"] and calls[-1] == "close"
    else:
        assert viewer.renderer is not original and viewer._frame_id == frame_id + 1
        assert (viewer._state["width"], viewer._state["height"]) == (640, 360)
        assert (viewer.width, viewer.height) == (640, 360)
        publish(env, viewer)
        assert viewer._frame_id == frame_id + 1  # New resolution is also cached.


def test_render_deadline_includes_render_time_in_frame_period(session, monkeypatch):
    from types import SimpleNamespace

    env, viewer = session
    now = [10.0]
    monkeypatch.setattr(web, "time", SimpleNamespace(monotonic=lambda: now[0]))

    def render(*args):
        now[0] += 0.02  # A 20 ms render must fit inside the 33 ms budget.
        return np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr(viewer.renderer, "render", render)
    viewer.target_fps = 30
    publish(env, viewer)
    assert viewer.next_frame_time == pytest.approx(10 + 1 / 30)
    assert not viewer.update_due(10.025)
    assert viewer.update_due(10.034)


@pytest.mark.parametrize("failure", ["construct", "render", "format", None])
def test_renderer_switch_is_transactional_on_main_thread(session, monkeypatch, failure):
    env, viewer = session
    original = viewer.renderer
    initial = env.observe()["step_id"].copy()
    calls = []
    main_thread = threading.get_ident()

    class Candidate:
        camera_control = True

        def render(self, *args):
            assert threading.get_ident() == main_thread
            calls.append("render")
            if failure == "render":
                raise RuntimeError("render failed")
            return np.full(
                (64, 80, 3), 100, dtype=np.float32 if failure == "format" else np.uint8
            )

        def close(self):
            assert threading.get_ident() == main_thread
            calls.append("close")

    candidate = Candidate()

    def create(*args):
        assert threading.get_ident() == main_thread
        calls.append("create")
        if failure == "construct":
            raise RuntimeError("construct failed")
        return candidate

    monkeypatch.setattr(web, "create_frame_renderer", create)
    viewer._availability["mujoco"]["metadata_ok"] = True
    with request(viewer, "/api/control", {"action": "renderer", "backend": "mujoco"}):
        pass
    assert not calls
    viewer.is_running()
    publish(env, viewer)
    assert viewer._state["ready"]
    assert np.array_equal(env.observe()["step_id"], initial)
    if failure:
        assert viewer.renderer is original
        assert viewer.backend == "raster" and viewer._state["error"]
        assert calls == (
            ["create"] if failure == "construct" else ["create", "render", "close"]
        )
    else:
        assert viewer.renderer is candidate
        assert viewer.backend == "mujoco" and viewer._state["error"] is None
        assert calls == ["create", "render"]


def test_mjpeg_disconnect_and_shutdown_release_waiting_reader(session):
    env, viewer = session
    publish(env, viewer)
    stream = request(viewer, "/stream.mjpg")
    assert stream.readline() == b"--frame\r\n"
    assert stream.readline() == b"Content-Type: image/jpeg\r\n"
    assert stream.readline() == b"\r\n"
    assert Image.open(io.BytesIO(stream.read(len(viewer._frame)))).size == (96, 96)
    assert stream.read(2) == b"\r\n"
    # The following part's complete headers must arrive with the first frame, even
    # when no subsequent simulation update occurs (a newly opened paused tab).
    assert stream.readline() == b"--frame\r\n"
    assert stream.readline() == b"Content-Type: image/jpeg\r\n"
    assert stream.readline() == b"\r\n"
    viewer.paused = True
    publish(env, viewer)
    assert viewer._frame_id == 1
    viewer.submit({"action": "camera_reset"})
    viewer.submit({"action": "select", "env_id": 1})
    viewer.is_running()
    publish(env, viewer)
    assert viewer._frame_id == 2
    assert Image.open(io.BytesIO(stream.read(len(viewer._frame)))).size == (96, 96)
    assert stream.read(2) == b"\r\n"
    assert stream.readline() == b"--frame\r\n"
    assert stream.readline() == b"Content-Type: image/jpeg\r\n"
    assert stream.readline() == b"\r\n"
    viewer.close()
    assert stream.read() == b""
    stream.close()
    assert not viewer.thread.is_alive()
    viewer.close()


def test_camera_commands_and_stop(session):
    env, viewer = session
    viewer.submit({"action": "camera", "azimuth": 30, "elevation": 80, "distance": 2})
    viewer.is_running()
    publish(env, viewer)
    assert viewer._state["camera"]["azimuth"] == 30
    viewer.submit({"action": "camera_reset"})
    viewer.is_running()
    assert viewer.camera == web.OrbitCamera()
    viewer.submit({"action": "stop"})
    assert not viewer.is_running()


def test_paused_frames_are_reused_but_http_state_stays_current(session, monkeypatch):
    env, viewer = session
    viewer.paused = True
    publish(env, viewer)
    initial = dict(viewer._state)
    frame = viewer._frame

    def unexpected(*args):
        pytest.fail("Unchanged paused scene must not render or encode")

    monkeypatch.setattr(viewer.renderer, "render", unexpected)
    monkeypatch.setattr(viewer, "_encode", unexpected)
    viewer.velocity_limits = (1.5, 0.8, 1.2)
    viewer.submit({"action": "velocity", "env_id": 0, "value": [0.5, 0, 0]})
    viewer.submit({"action": "speed", "value": 2})
    viewer.is_running()
    command = viewer.consume_velocity()[1]
    scene = env.task.scene_update(env.physics.snapshot())
    viewer.update(
        scene,
        env.observe(),
        np.ones(3),
        display_state={
            "position": scene.entities["agent"][0].tolist(),
            "velocity": 0.0,
            "reward": 1.0,
            "live": {"command": command},
        },
    )
    with request(viewer, "/api/state") as response:
        state = json.load(response)
    assert state["state_id"] == initial["state_id"] + 1
    assert state["frame_id"] == initial["frame_id"]
    assert state["speed"] == 2 and state["live"]["command"] == command
    assert state["reward"] == 1.0 and state["render_idle"] and state["fps"] == 0
    with request(viewer, "/snapshot.jpg") as response:
        assert response.read() == frame


@pytest.mark.parametrize("change", ["camera", "selection", "reset", "entity", "resume"])
def test_paused_visual_changes_redraw(session, change):
    env, viewer = session
    viewer.paused = True
    scene = env.task.scene_update(env.physics.snapshot())
    observation = env.observe()
    viewer.update(scene, observation, np.zeros(3))
    first = viewer._frame_id
    viewer.update(scene, observation, np.zeros(3))
    assert viewer._frame_id == first
    if change == "camera":
        viewer.submit(
            {"action": "camera", "azimuth": 0, "elevation": 30, "distance": 2}
        )
    elif change == "selection":
        viewer.submit({"action": "select", "env_id": 1})
    elif change == "reset":
        env.reset([0])
        scene, observation = (
            env.task.scene_update(env.physics.snapshot()),
            env.observe(),
        )
    elif change == "entity":
        # SceneUpdate is borrowed; in-place edits must invalidate the owned key.
        scene.entities["target"][0, 0] += 0.1
    elif change == "resume":
        viewer.submit({"action": "pause", "paused": False})
    viewer.is_running()
    viewer.update(scene, observation, np.zeros(3))
    assert viewer._frame_id == first + 1
    assert not viewer._state["render_idle"]


def test_paused_robot_joint_and_time_changes_redraw(session, monkeypatch):
    from embodiedforge.viewers.robot import RobotSceneUpdate

    env, viewer = session
    viewer.paused = True
    calls = []

    def render(scene, *args):
        calls.append(scene.qpos.copy())
        return np.zeros((64, 80, 3), dtype=np.uint8)

    monkeypatch.setattr(viewer.renderer, "render", render)
    scene = RobotSceneUpdate(np.zeros((3, 19)), np.zeros(3))

    def update():
        viewer.update(
            scene,
            env.observe(),
            np.zeros(3),
            display_state={"position": [0, 0, 0], "velocity": 0, "reward": 0},
        )

    update()
    scene.qpos[1, 7] = 0.2
    update()
    assert len(calls) == 1  # Another robot is not displayed.
    scene.qpos[0, 7] = 0.2
    update()
    assert len(calls) == 2
    scene.time[0] = 0.02
    update()
    assert len(calls) == 3


def test_paused_renderer_switch_and_failure_keep_cache_correct(session, monkeypatch):
    env, viewer = session
    viewer.paused = True
    publish(env, viewer)
    original_frame = viewer._frame
    initial_id = viewer._frame_id
    calls = []

    class Candidate:
        camera_control = True

        def render(self, *args):
            calls.append("render")
            return np.full((64, 80, 3), 100, dtype=np.uint8)

        def close(self):
            pass

    def create(name, *args):
        if name == "gl":
            raise RuntimeError("deliberate initialization failure")
        return Candidate()

    monkeypatch.setattr(web, "create_frame_renderer", create)
    for name in ("gl", "mujoco"):
        viewer._availability[name]["metadata_ok"] = True
    viewer.submit({"action": "renderer", "backend": "gl"})
    viewer.is_running()
    publish(env, viewer)
    assert viewer._state["error"] and viewer._frame == original_frame
    assert viewer._frame_id == initial_id
    viewer.submit({"action": "renderer", "backend": "mujoco"})
    viewer.is_running()
    publish(env, viewer)
    assert viewer._state["error"] is None and viewer._frame_id == initial_id + 1
    assert viewer._frame != original_frame
    publish(env, viewer)
    assert calls == ["render"] and viewer._state["render_idle"]


def test_isolated_frame_worker_owns_arrays_and_closes(session):
    from multiprocessing import active_children

    from embodiedforge.viewers.frames import IsolatedFrames, RasterFrames

    env, _ = session
    worker = IsolatedFrames("raster", env.task.scene, env.config, 128, 96)
    pid = worker.process.pid
    direct = RasterFrames(env.task.scene, env.config, 128, 96)
    try:
        args = (
            env.task.scene_update(env.physics.snapshot()),
            env.observe(),
            2,
            web.OrbitCamera(),
        )
        first = worker.render(*args)
        np.testing.assert_array_equal(first, direct.render(*args))
        preserved = first.copy()
        env.reset([2])
        args = (
            env.task.scene_update(env.physics.snapshot()),
            env.observe(),
            2,
            web.OrbitCamera(),
        )
        second = worker.render(*args)
        np.testing.assert_array_equal(second, direct.render(*args))
        np.testing.assert_array_equal(first, preserved)
        assert not np.array_equal(first, second)
    finally:
        direct.close()
        worker.close()
    assert pid not in [process.pid for process in active_children()]
    worker.close()


def test_isolated_worker_crash_does_not_hang(session):
    from embodiedforge.viewers.frames import IsolatedFrames

    env, _ = session
    worker = IsolatedFrames("raster", env.task.scene, env.config, 128, 96)
    worker.process.terminate()
    worker.process.join(timeout=3)
    try:
        with pytest.raises((RuntimeError, OSError)):
            worker.render(
                env.task.scene_update(env.physics.snapshot()),
                env.observe(),
                0,
                web.OrbitCamera(),
            )
    finally:
        worker.close()
    assert worker.closed


def test_isolated_startup_error_is_reported(session):
    from embodiedforge.viewers.frames import IsolatedFrames

    env, _ = session
    with pytest.raises(RuntimeError, match="Unknown frame renderer"):
        IsolatedFrames("nonexistent", env.task.scene, env.config, 128, 96)


def test_replay_controls_follow_and_seek_order(session):
    env, viewer = session
    viewer.replay_frames = (10, 20, 30)
    viewer.submit({"action": "seek", "env_id": 1, "frame": 15})
    viewer.submit({"action": "reset", "env_id": 1})
    viewer.is_running()
    assert viewer.consume_seek() is None and viewer.consume_reset() == 1
    viewer.submit({"action": "reset", "env_id": 2})
    viewer.submit({"action": "seek", "env_id": 2, "frame": 25})
    viewer.submit({"action": "seek", "env_id": 2, "frame": 26})
    viewer.submit({"action": "follow", "enabled": True})
    viewer.is_running()
    assert viewer.consume_seek() == (2, 26)
    assert viewer.consume_reset() is None
    assert viewer.paused and not viewer.should_step()
    viewer.update(
        env.task.scene_update(env.physics.snapshot()),
        env.observe(),
        np.zeros(3),
        display_state={"position": [1.0, 2.0, 3.0], "velocity": 0.0, "reward": None},
    )
    np.testing.assert_allclose(viewer.camera.target, [1.0, 2.0, 3.0])
    viewer.submit({"action": "follow", "enabled": False})
    viewer.is_running()
    previous = viewer.camera.target
    env.reset()
    publish(env, viewer)
    assert viewer.camera.target == previous


def test_live_session_rejects_replay_controls(session):
    _, viewer = session
    for command in [
        {"action": "seek", "env_id": 0, "frame": 0},
        {"action": "follow", "enabled": True},
    ]:
        with pytest.raises(ValueError, match="recorded motion"):
            viewer.submit(command)


@pytest.mark.parametrize(
    "command",
    [
        {"action": "seek", "env_id": 0, "frame": -1},
        {"action": "seek", "env_id": 0, "frame": 10},
        {"action": "seek", "env_id": 0, "frame": True},
        {"action": "follow", "enabled": 1},
    ],
)
def test_invalid_replay_controls(session, command):
    _, viewer = session
    viewer.replay_frames = (10, 20, 30)
    with pytest.raises(ValueError):
        viewer.submit(command)


def test_live_velocity_queue_is_owned_and_reset_ordered(session):
    _, viewer = session
    viewer.velocity_limits = (1.5, 0.8, 1.2)
    value = [0.5, 0.2, 0]
    viewer.submit({"action": "velocity", "env_id": 1, "value": value})
    value[0] = 100
    viewer.is_running()
    assert viewer.consume_velocity() == (1, [0.5, 0.2, 0])
    viewer.submit({"action": "velocity", "env_id": 0, "value": [0.5, 0, 0]})
    viewer.submit({"action": "reset", "env_id": 0})
    viewer.is_running()
    assert viewer.consume_velocity() is None and viewer.consume_reset() == 0
    viewer.submit({"action": "reset", "env_id": 0})
    viewer.submit({"action": "velocity", "env_id": 0, "value": [0.2, 0, 0]})
    viewer.is_running()
    assert viewer.consume_reset() == 0 and viewer.consume_velocity() == (0, [0.2, 0, 0])


@pytest.mark.parametrize(
    "value", [[1.6, 0, 0], [0, float("nan"), 0], [True, 0, 0], [0, 0], None]
)
def test_web_velocity_validation_precedes_queue(session, value):
    _, viewer = session
    viewer.velocity_limits = (1.5, 0.8, 1.2)
    with pytest.raises(ValueError):
        viewer.submit({"action": "velocity", "env_id": 0, "value": value})
    assert not viewer._commands
