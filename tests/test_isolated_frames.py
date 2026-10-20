"""Interrupted renderer cleanup must still reap its owned child process."""

import multiprocessing
import signal
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embodiedforge.viewers import frames


def _ignore_termination(connection):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    connection.send("ready")
    time.sleep(60)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_interrupted_kill_reaps_a_termination_resistant_child(monkeypatch, error_type):
    context = multiprocessing.get_context("spawn")
    worker = object.__new__(frames.IsolatedFrames)
    worker.closed = False
    worker.connection, child = context.Pipe()
    worker.process = context.Process(target=_ignore_termination, args=(child,))
    worker.process.start()
    child.close()
    join, kill = worker.process.join, worker.process.kill
    primary = error_type("kill interrupted before signaling")
    calls = 0

    def interrupt_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise primary
        kill()

    monkeypatch.setattr(worker.process, "kill", interrupt_once)
    # Preserve escalation order while keeping the fault test fast.
    monkeypatch.setattr(worker.process, "join", lambda timeout: join(min(timeout, 0.1)))
    try:
        assert worker.connection.poll(10)
        assert worker.connection.recv() == "ready"
        with pytest.raises(error_type) as caught:
            worker.close()
        assert caught.value is primary
        assert worker.connection.closed
        with pytest.raises(ValueError, match="closed"):
            worker.process.is_alive()
        assert calls == 2
        worker.close()
    finally:
        worker.connection.close()
        try:
            if worker.process.is_alive():
                kill()
            join(5)
            worker.process.close()
        except ValueError:
            pass


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
    assert worker.connection.close.call_count == (2 if stage == "close" else 1)
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


@pytest.mark.parametrize("interrupted_probe", [0, 1, 2, 3])
def test_interrupted_liveness_probe_still_releases_process_handle(interrupted_probe):
    worker, alive = worker_with_fault("join", None)
    primary = KeyboardInterrupt("liveness query interrupted")
    probes = 0

    def is_alive():
        nonlocal probes
        probes += 1
        if probes == interrupted_probe + 1:
            raise primary
        return alive[0]

    worker.process.is_alive.side_effect = is_alive
    with pytest.raises(KeyboardInterrupt) as caught:
        worker.close()
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


def test_interrupted_handle_closes_release_real_descriptors(monkeypatch):
    context = multiprocessing.get_context("spawn")
    worker = object.__new__(frames.IsolatedFrames)
    worker.closed = False
    worker.connection, child = context.Pipe()
    worker.process = context.Process(target=time.sleep, args=(60,))
    worker.process.start()
    child.close()
    join, close_pipe, close_process = (
        worker.process.join,
        worker.connection.close,
        worker.process.close,
    )
    primary = KeyboardInterrupt("join interrupted")

    def once_then(operation, error):
        first = True

        def interrupted(*args, **kwargs):
            nonlocal first
            if first:
                first = False
                raise error
            return operation(*args, **kwargs)

        return interrupted

    monkeypatch.setattr(worker.process, "join", once_then(join, primary))
    monkeypatch.setattr(
        worker.connection,
        "close",
        once_then(close_pipe, OSError("pipe close interrupted")),
    )
    monkeypatch.setattr(
        worker.process,
        "close",
        once_then(close_process, SystemExit("process handle close interrupted")),
    )
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            worker.close()
        assert caught.value is primary
        assert worker.connection.closed
        with pytest.raises(ValueError, match="closed"):
            worker.process.is_alive()
    finally:
        close_pipe()
        try:
            if worker.process.is_alive():
                worker.process.kill()
            join(timeout=3)
            close_process()
        except ValueError:
            pass
