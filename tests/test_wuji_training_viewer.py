import sys
from types import ModuleType, SimpleNamespace
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from embodiedforge import recipes
from embodiedforge._wuji_training_viewer import TrainingPreview, training_preview


@pytest.mark.parametrize(
    "flags,enabled",
    [
        ([], False),
        (["--headless"], False),
        (["--no-headless"], True),
        (["--no-headless", "--headless"], False),
    ],
)
def test_cli_display_defaults_and_override(monkeypatch, flags, enabled):
    calls = []
    monkeypatch.setattr(recipes, "execute", calls.append)
    recipes.main(
        [
            "train",
            "--task",
            "wuji-reorient",
            "--output",
            "unused",
            "--updates",
            "1000",
            *flags,
        ]
    )
    assert calls[0].headless is not enabled
    assert calls[0].viewer_port == 8083
    assert calls[0].updates == 1000 and calls[0].timeout is None


@pytest.mark.parametrize(
    "flags",
    [
        ["--visualize"],
        ["--viewer-port", "0"],
        ["--viewer-port", "65536"],
        ["--viewer-fps", "0"],
        ["--viewer-fps", "31"],
    ],
)
def test_invalid_display_options_fail_before_launch(flags):
    with pytest.raises(SystemExit):
        recipes.main(
            [
                "train",
                "--task",
                "wuji-reorient",
                "--output",
                "unused",
                "--updates",
                "1000",
                *flags,
            ]
        )


def test_unsupported_task_rejected_before_output_creation(tmp_path):
    path = tmp_path / "run"
    with pytest.raises(ValueError, match="Wuji"):
        recipes.main(
            [
                "train",
                "--task",
                "cartpole-mpc",
                "--output",
                str(path),
                "--updates",
                "1000",
                "--no-headless",
            ]
        )
    assert not path.exists()


def test_preview_http_wait_frame_and_shutdown():
    preview = TrainingPreview(port=0)
    root = f"http://127.0.0.1:{preview.server.server_port}"
    try:
        assert b"environment 0" in urlopen(root, timeout=2).read()
        with pytest.raises(HTTPError) as error:
            urlopen(root + "/frame.jpg", timeout=2)
        assert error.value.code == 503
        preview.frame = (b"jpeg-frame", 42)
        with urlopen(root + "/frame.jpg", timeout=2) as response:
            assert response.read() == b"jpeg-frame"
            assert response.headers["X-Rollout-Step"] == "42"
        with pytest.raises(OSError):
            TrainingPreview(port=preview.server.server_port)
    finally:
        preview.close()
    assert not preview.thread.is_alive()


def test_hook_preserves_actions_results_and_restores_after_error(monkeypatch):
    class Base:
        def step(self, actions):
            assert actions is action
            return result

    class Wrapper(Base):
        env = object()

    module = ModuleType("wuji_unilab.rl.runtime")
    module.WujiWrapper = Wrapper
    monkeypatch.setitem(sys.modules, "wuji_unilab.rl.runtime", module)
    action, result = object(), object()
    calls = []
    preview = SimpleNamespace(update=calls.append, close=lambda: calls.append("closed"))
    monkeypatch.setattr(
        "embodiedforge._wuji_training_viewer.TrainingPreview", lambda *a: preview
    )
    with pytest.raises(RuntimeError, match="training error"):
        with training_preview({"viewer_port": 8083, "viewer_fps": 10}):
            assert Wrapper().step(action) is result
            raise RuntimeError("training error")
    assert calls == [Wrapper.env, "closed"]
    assert "step" not in Wrapper.__dict__


def test_throttle_skips_snapshot_and_render_imports():
    preview = TrainingPreview.__new__(TrainingPreview)
    preview.steps = 0
    preview.next_frame = float("inf")
    preview.update(object())
    assert preview.steps == 1


def test_headless_training_never_imports_preview(monkeypatch):
    from embodiedforge import _wuji_recipe

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    fake_torch.set_num_threads = lambda _: None
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    cli = ModuleType("wuji_unilab.cli")
    calls = []
    cli.train_main = lambda: calls.append(list(sys.argv))
    monkeypatch.setitem(sys.modules, "wuji_unilab.cli", cli)
    # An accidental preview import must fail, even if the module is installed.
    monkeypatch.setitem(sys.modules, "embodiedforge._wuji_training_viewer", None)
    monkeypatch.setattr(_wuji_recipe, "verify_training", lambda _: "verified")
    monkeypatch.setattr(sys, "argv", [])
    request = dict(
        command="train",
        threads=1,
        upstream_task="WujiHand_Reorient",
        num_envs=32,
        updates=5,
        horizon=40,
        seed=0,
    )
    assert _wuji_recipe.run(request) == "verified"
    assert "training.no_play=true" in calls[0]


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["construct_thread", "before_start", "after_start"])
def test_preview_startup_failure_closes_bound_listener_and_started_thread(
    monkeypatch, failure_type, stage
):
    from embodiedforge import _wuji_training_viewer as module

    failure = failure_type("preview startup failed")
    servers, threads = [], []
    server_factory, thread_factory = module.ThreadingHTTPServer, module.Thread

    def make_server(*args, **kwargs):
        server = server_factory(*args, **kwargs)
        servers.append(server)
        return server

    def make_thread(*args, **kwargs):
        if stage == "construct_thread":
            raise failure
        thread = thread_factory(*args, **kwargs)
        threads.append(thread)
        start = thread.start

        def fail_start():
            if stage == "after_start":
                start()
            raise failure

        monkeypatch.setattr(thread, "start", fail_start)
        return thread

    monkeypatch.setattr(module, "ThreadingHTTPServer", make_server)
    monkeypatch.setattr(module, "Thread", make_thread)
    with pytest.raises(failure_type) as caught:
        TrainingPreview(port=0)
    assert caught.value is failure
    assert servers[0].fileno() == -1
    assert all(not thread.is_alive() for thread in threads)


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("before_stop", [False, True])
def test_preview_close_finishes_cleanup_and_preserves_first_failure(
    monkeypatch, failure_type, before_stop
):
    preview = TrainingPreview(port=0)
    calls = []
    shutdown, server_close = preview.server.shutdown, preview.server.server_close
    failure = failure_type("shutdown failed")

    def fail_shutdown():
        calls.append("shutdown")
        if not before_stop:
            shutdown()
        raise failure

    def fail_socket_close():
        calls.append("socket")
        server_close()
        raise OSError("secondary socket failure")

    def fail_renderer_close():
        calls.append("renderer")
        raise KeyboardInterrupt("secondary renderer failure")

    preview.renderer = SimpleNamespace(close=fail_renderer_close)
    monkeypatch.setattr(preview.server, "shutdown", fail_shutdown)
    monkeypatch.setattr(preview.server, "server_close", fail_socket_close)
    try:
        with pytest.raises(failure_type) as caught:
            preview.close()
        assert caught.value is failure
        assert calls == ["shutdown", "socket", "renderer"]
        assert preview.server.fileno() == -1
        assert not preview.thread.is_alive()
        preview.close()
        assert calls == ["shutdown", "socket", "renderer"]
    finally:
        if preview.thread.is_alive():
            shutdown()
        server_close()
        preview.thread.join(timeout=2)


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [OSError, KeyboardInterrupt, SystemExit])
def test_training_failure_survives_preview_cleanup_failure_and_restores_hook(
    monkeypatch, inherited, failure_type, cleanup_type
):
    class Base:
        def step(self, actions):
            return actions

    class Wrapper(Base):
        pass

    if not inherited:
        Wrapper.step = lambda self, actions: actions
    original = Wrapper.step
    module = ModuleType("wuji_unilab.rl.runtime")
    module.WujiWrapper = Wrapper
    monkeypatch.setitem(sys.modules, module.__name__, module)
    primary = failure_type("primary training failure")
    secondary = cleanup_type("preview cleanup failure")
    calls = []

    def close():
        calls.append("closed")
        assert Wrapper.step is original
        assert ("step" not in Wrapper.__dict__) is inherited
        raise secondary

    monkeypatch.setattr(
        "embodiedforge._wuji_training_viewer.TrainingPreview",
        lambda *args: SimpleNamespace(close=close),
    )
    with pytest.raises(failure_type) as caught:
        with training_preview({"viewer_port": 0, "viewer_fps": 10}):
            raise primary
    assert caught.value is primary
    assert calls == ["closed"]


def test_successful_training_propagates_preview_close_failure_inside_except(
    monkeypatch,
):
    class Wrapper:
        def step(self, actions):
            return actions

    original = Wrapper.step
    module = ModuleType("wuji_unilab.rl.runtime")
    module.WujiWrapper = Wrapper
    monkeypatch.setitem(sys.modules, module.__name__, module)
    failure = OSError("preview close failed")

    def close():
        raise failure

    monkeypatch.setattr(
        "embodiedforge._wuji_training_viewer.TrainingPreview",
        lambda *args: SimpleNamespace(close=close),
    )
    try:
        raise RuntimeError("unrelated handled error")
    except RuntimeError:
        with pytest.raises(OSError) as caught:
            with training_preview({"viewer_port": 0, "viewer_fps": 10}):
                pass
    assert caught.value is failure
    assert Wrapper.step is original


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("secondary_type", [None, KeyboardInterrupt])
def test_preview_interrupted_server_close_still_releases_listener(
    monkeypatch, failure_type, secondary_type
):
    preview = TrainingPreview(port=0)
    original = preview.server.server_close
    primary = failure_type("before closing listener")
    calls = []

    def server_close():
        calls.append("server")
        raise primary

    def renderer_close():
        calls.append("renderer")
        if secondary_type:
            raise secondary_type("secondary renderer cleanup")

    preview.renderer = SimpleNamespace(close=renderer_close)
    monkeypatch.setattr(preview.server, "server_close", server_close)
    try:
        with pytest.raises(failure_type) as caught:
            preview.close()
        assert caught.value is primary
        assert preview.server.fileno() == -1
        assert not preview.thread.is_alive()
        assert calls == ["server", "renderer"]
        preview.close()
        assert calls == ["server", "renderer"]
    finally:
        original()


def test_preview_startup_failure_releases_listener_if_server_close_is_interrupted(
    monkeypatch,
):
    from embodiedforge import _wuji_training_viewer as module

    servers = []
    factory = module.ThreadingHTTPServer
    primary = KeyboardInterrupt("startup interrupted")

    def server(*args, **kwargs):
        instance = factory(*args, **kwargs)
        servers.append(instance)

        def server_close():
            raise OSError("before closing listener")

        monkeypatch.setattr(instance, "server_close", server_close)
        return instance

    def start(*args):
        raise primary

    monkeypatch.setattr(module, "ThreadingHTTPServer", server)
    monkeypatch.setattr(module.Thread, "start", start)
    with pytest.raises(KeyboardInterrupt) as caught:
        TrainingPreview(port=0)
    assert caught.value is primary
    assert len(servers) == 1 and servers[0].fileno() == -1
