"""Interrupted renderer cleanup must still reap its owned child process."""

import multiprocessing
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embodiedforge.viewers import frames


def worker_with_fault(stage, error):
    worker = object.__new__(frames.IsolatedFrames)
    worker.closed = False
    alive = [True]
    worker.connection = MagicMock()
    worker.process = MagicMock(pid=123)
    worker.process.is_alive.side_effect = lambda: alive[0]
    worker.process.terminate.side_effect = lambda: alive.__setitem__(0, False)
    worker.process.kill.side_effect = lambda: alive.__setitem__(0, False)
    target = worker.connection if stage in ("send", "close") else worker.process
    getattr(target, stage).side_effect = error
    return worker, alive


@pytest.mark.parametrize("stage", ["send", "join", "terminate", "close"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_cleanup_reaps_worker_after_an_interrupted_operation(stage, error_type):
    error = error_type("cleanup interrupted")
    worker, alive = worker_with_fault(stage, error)
    with pytest.raises(error_type) as caught:
        worker.close()
    assert caught.value is error
    assert not alive[0]
    worker.connection.close.assert_called_once()
    worker.process.close.assert_called_once()
    assert worker.closed
    worker.close()
    worker.process.close.assert_called_once()


@pytest.mark.parametrize("primary_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("cleanup_type", [OSError, KeyboardInterrupt, SystemExit])
def test_render_failure_survives_secondary_cleanup_failure(primary_type, cleanup_type):
    primary = primary_type("render failed")
    worker, alive = worker_with_fault("join", cleanup_type("cleanup failed"))
    worker._receive = MagicMock(side_effect=primary)
    with pytest.raises(primary_type) as caught:
        worker.render(None, {}, 0, frames.OrbitCamera())
    assert caught.value is primary
    assert not alive[0]
    worker.connection.close.assert_called_once()
    worker.process.close.assert_called_once()


@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("child_close_fails", [False, True])
@pytest.mark.parametrize("primary_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_startup_preserves_primary_and_releases_both_pipe_ends(
    monkeypatch, started, child_close_fails, primary_type
):
    primary = primary_type("startup failed")
    worker, _ = worker_with_fault("join", OSError("cleanup failed"))
    child = MagicMock()
    if child_close_fails:
        child.close.side_effect = OSError("child close failed")
    if not started:
        worker.process.pid = None
        worker.process.start.side_effect = primary
    context = SimpleNamespace(
        Pipe=lambda: (worker.connection, child), Process=lambda **kwargs: worker.process
    )
    monkeypatch.setattr(frames.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(
        frames.IsolatedFrames, "_receive", MagicMock(side_effect=primary)
    )
    # If child.close itself fails, that is the original startup error.
    expected = child.close.side_effect if started and child_close_fails else primary
    with pytest.raises(type(expected)) as caught:
        frames.IsolatedFrames("raster", None, None, 64, 64)
    assert caught.value is expected
    worker.connection.close.assert_called_once()
    if started:
        worker.process.close.assert_called_once()


def test_interrupted_join_terminates_and_reaps_a_real_child(monkeypatch):
    context = multiprocessing.get_context("spawn")
    worker = object.__new__(frames.IsolatedFrames)
    worker.closed = False
    worker.connection, child = context.Pipe()
    worker.process = context.Process(target=time.sleep, args=(60,))
    worker.process.start()
    child.close()
    pid = worker.process.pid
    join = worker.process.join
    interrupted = KeyboardInterrupt("initial join interrupted")
    calls = 0

    def interrupt_once(timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise interrupted
        return join(timeout=timeout)

    monkeypatch.setattr(worker.process, "join", interrupt_once)
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker.close()
        assert caught.value is interrupted
        assert worker.connection.closed
        assert pid not in [process.pid for process in multiprocessing.active_children()]
        worker.close()
    finally:
        # Baseline regressions must not leave a child behind when this test fails.
        try:
            if worker.process.is_alive():
                worker.process.kill()
            join(timeout=3)
            worker.process.close()
        except ValueError:
            pass  # The successful path already closed the Process handle.
