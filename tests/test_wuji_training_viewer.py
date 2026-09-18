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
    recipes.main(["train", "--task", "wuji-reorient", "--output", "unused", *flags])
    assert calls[0].headless is not enabled
    assert calls[0].viewer_port == 8083


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
        recipes.main(["train", "--task", "wuji-reorient", "--output", "unused", *flags])


def test_unsupported_task_rejected_before_output_creation(tmp_path):
    path = tmp_path / "run"
    with pytest.raises(ValueError, match="Wuji"):
        recipes.main(
            ["train", "--task", "cartpole-mpc", "--output", str(path), "--no-headless"]
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
