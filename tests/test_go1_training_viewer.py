"""Training previews restore environment hooks and retain primary failures."""

from types import SimpleNamespace

import pytest

from embodiedforge import _go1_training_viewer as module


@pytest.mark.parametrize("instance_step", [False, True])
@pytest.mark.parametrize(
    "failure_type", [None, RuntimeError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize("cleanup_type", [None, OSError, KeyboardInterrupt, SystemExit])
def test_preview_restores_step_and_reports_primary_error(
    monkeypatch, instance_step, failure_type, cleanup_type
):
    class Env:
        def step(self, actions):
            assert actions is action
            return result

    env = Env()
    if instance_step:
        env.step = lambda actions: result
    original = env.step
    action, result = object(), object()
    primary = failure_type("training failed") if failure_type else None
    secondary = cleanup_type("preview cleanup failed") if cleanup_type else None
    calls = []

    def close():
        calls.append("closed")
        assert env.step == original
        assert ("step" in vars(env)) is instance_step
        if secondary is not None:
            raise secondary

    preview = SimpleNamespace(update=calls.append, close=close)
    monkeypatch.setattr(module, "Go1Preview", lambda *args: preview)

    def train():
        with module.training_preview(env, {"viewer_port": 0, "viewer_fps": 10}):
            assert env.step(action) is result
            if primary is not None:
                raise primary

    expected = primary if primary is not None else secondary
    if expected is None:
        train()
    else:
        with pytest.raises(type(expected)) as caught:
            train()
        assert caught.value is expected
    assert calls == [env, "closed"]
    assert env.step == original
    assert ("step" in vars(env)) is instance_step


def test_invalid_environment_does_not_start_preview(monkeypatch):
    monkeypatch.setattr(
        module, "Go1Preview", lambda *args: pytest.fail("Unexpected preview server")
    )
    with pytest.raises(AttributeError):
        with module.training_preview(object(), {"viewer_port": 0, "viewer_fps": 10}):
            pass


def test_nested_preview_restores_outer_hook_before_restoring_class_method(monkeypatch):
    class Env:
        def step(self, actions):
            return actions

    env = Env()
    calls = []

    def make_preview(*args):
        preview_id = len(calls)
        calls.append((preview_id, "created"))
        return SimpleNamespace(
            update=lambda owner: calls.append((preview_id, owner)),
            close=lambda: calls.append((preview_id, "closed")),
        )

    monkeypatch.setattr(module, "Go1Preview", make_preview)
    request = {"viewer_port": 0, "viewer_fps": 10}
    with module.training_preview(env, request):
        outer = env.step
        with module.training_preview(env, request):
            assert env.step(1) == 1
        assert env.step is outer
        assert env.step(2) == 2
    assert "step" not in vars(env)
    assert calls == [
        (0, "created"),
        (1, "created"),
        (0, env),
        (1, env),
        (1, "closed"),
        (0, env),
        (0, "closed"),
    ]
