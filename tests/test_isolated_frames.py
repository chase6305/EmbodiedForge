"""Interrupted renderer cleanup must still reap its owned child process."""

import multiprocessing
import signal
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from embodiedforge.viewers import frames
from embodiedforge.viewers._frame_channel import FrameChannel, channel_pair


def _ignore_termination(sock):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    with sock:
        connection = FrameChannel(sock)
        connection.send("ready", deadline=time.monotonic() + 5)
        time.sleep(60)


def _sleep_with_socket(sock):
    with sock:
        time.sleep(60)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_interrupted_kill_reaps_a_termination_resistant_child(monkeypatch, error_type):
    context = multiprocessing.get_context("spawn")
    worker = object.__new__(frames.IsolatedFrames)
    worker.closed = False
    worker.name = "raster"
    worker.connection, child = channel_pair()
    worker.process = context.Process(target=_ignore_termination, args=(child.socket,))
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
        assert worker.connection.receive(deadline=time.monotonic() + 10) == "ready"
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
    worker.name = "raster"
    alive = [True]
    worker.connection = MagicMock(closed=False, failed=False)
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
def test_startup_preserves_primary_and_releases_both_channel_ends(
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
    context = SimpleNamespace(Process=lambda **kwargs: worker.process)
    monkeypatch.setattr(frames, "channel_pair", lambda: (worker.connection, child))
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
    worker.name = "raster"
    worker.connection, child = channel_pair()
    worker.process = context.Process(target=_sleep_with_socket, args=(child.socket,))
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
    worker.name = "raster"
    worker.connection, child = channel_pair()
    worker.process = context.Process(target=_sleep_with_socket, args=(child.socket,))
    worker.process.start()
    child.close()
    join, close_channel, close_process = (
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
        once_then(close_channel, OSError("channel close interrupted")),
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
        close_channel()
        try:
            if worker.process.is_alive():
                worker.process.kill()
            join(timeout=3)
            close_process()
        except ValueError:
            pass


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_process_factory_failure_releases_both_real_channel_ends(
    monkeypatch, error_type
):
    parent, child = channel_pair()
    primary = error_type("process construction failed")

    def process(**kwargs):
        assert kwargs["args"][0] is child.socket
        raise primary

    monkeypatch.setattr(frames, "channel_pair", lambda: (parent, child))
    monkeypatch.setattr(
        frames.multiprocessing,
        "get_context",
        lambda _: SimpleNamespace(Process=process),
    )
    try:
        with pytest.raises(error_type) as caught:
            frames.IsolatedFrames("raster", None, None, 64, 64)
        assert caught.value is primary
        assert parent.closed and child.closed
        assert parent.socket.fileno() == child.socket.fileno() == -1
    finally:
        parent.close()
        child.close()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_channel_factory_failure_preserves_original_error(monkeypatch, error_type):
    primary = error_type("socket pair construction failed")
    process = MagicMock()
    monkeypatch.setattr(frames, "channel_pair", MagicMock(side_effect=primary))
    monkeypatch.setattr(
        frames.multiprocessing,
        "get_context",
        lambda _: SimpleNamespace(Process=process),
    )
    with pytest.raises(error_type) as caught:
        frames.IsolatedFrames("raster", None, None, 64, 64)
    assert caught.value is primary
    process.assert_not_called()


def test_spawn_interrupt_before_popen_publication_still_reaps_owned_child(monkeypatch):
    context = multiprocessing.get_context("spawn")
    parent, child = channel_pair()
    processes, popens, interrupted_states = [], [], []
    primary = KeyboardInterrupt("interrupted after OS spawn, before handle publication")

    def process(**kwargs):
        assert kwargs["target"] is frames._frame_worker
        # Model a renderer stalled in SDK initialization, without importing the
        # SDK. The socket is still transferred by the real spawn implementation.
        value = context.Process(
            target=_sleep_with_socket,
            args=(kwargs["args"][0],),
            name=kwargs["name"],
            daemon=kwargs["daemon"],
        )
        processes.append(value)
        launch, join = value._Popen, value.join

        def interrupted_launch(process_obj):
            # Instance attributes are included in spawn's process pickle. Remove
            # this local injection before the real launch serializes that object.
            del process_obj._Popen
            popen = launch(process_obj)
            popens.append(popen)
            # Install this only after serialization; do not pickle a local lambda.
            # Keep the real escalation waits while shortening the graceful wait.
            monkeypatch.setattr(
                value, "join", lambda timeout: join(0.1 if timeout == 5 else timeout)
            )
            assert value._popen is None and popen.pid is not None
            signal.raise_signal(signal.SIGINT)
            return popen

        monkeypatch.setattr(value, "_Popen", interrupted_launch)
        return value

    def interrupt(signum, frame):
        interrupted_states.append((child.closed, processes[0]._popen is popens[0]))
        raise primary

    previous = signal.signal(signal.SIGINT, interrupt)
    monkeypatch.setattr(frames, "channel_pair", lambda: (parent, child))
    monkeypatch.setattr(
        frames.multiprocessing,
        "get_context",
        lambda _: SimpleNamespace(Process=process),
    )
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            frames.IsolatedFrames("raster", None, None, 64, 64)
        assert caught.value is primary
        assert interrupted_states == [(True, True)]
        assert len(popens) == len(processes) == 1
        assert parent.closed and child.closed
        assert popens[0].poll() is not None
        with pytest.raises(ValueError, match="closed"):
            processes[0].is_alive()
    finally:
        signal.signal(signal.SIGINT, previous)
        parent.close()
        child.close()
        # In the unfixed implementation Process._popen remains None, so its
        # cleanup cannot find this real child. Retain the launch result and reap
        # exactly this test's child without relying on Process bookkeeping.
        for popen in popens:
            try:
                if popen.poll() is None:
                    popen.kill()
                popen.wait(timeout=3)
            finally:
                popen.close()


def test_worker_does_not_report_receive_failure_on_a_failed_open_channel(
    monkeypatch, caplog
):
    primary = ValueError("incomplete renderer frame")
    connection = MagicMock(closed=False, failed=False)
    renderer = MagicMock(camera_control=False)

    def receive(*, deadline):
        assert deadline is None
        # The receive error poisoned the stream; its first descriptor close was
        # interrupted before release. The worker must skip error publication.
        connection.failed = True
        raise primary

    def close():
        assert connection.failed and not connection.closed
        connection.closed = True

    connection.receive.side_effect = receive
    connection.close.side_effect = close
    monkeypatch.setattr(frames, "FrameChannel", lambda sock: connection)
    monkeypatch.setattr(frames, "create_frame_renderer", lambda *a, **kw: renderer)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    frames._frame_worker(object(), "raster", None, None, 64, 64)
    connection.send.assert_called_once()
    assert connection.send.call_args.args == ((True, False),)
    renderer.render.assert_not_called()
    renderer.close.assert_called_once()
    connection.close.assert_called_once()
    assert connection.closed
    failure = next(
        record
        for record in caplog.records
        if "Frame renderer worker failed" in record.getMessage()
    )
    assert failure.exc_info[1] is primary


def test_owner_reaps_worker_without_sending_on_a_failed_open_channel():
    worker, alive = worker_with_fault("send", AssertionError("poisoned stream reused"))
    worker.connection.failed = True
    worker.connection.close.side_effect = lambda: setattr(
        worker.connection, "closed", True
    )
    assert not worker.connection.closed
    worker.close()
    worker.connection.send.assert_not_called()
    worker.connection.close.assert_called_once()
    worker.process.terminate.assert_called_once()
    worker.process.close.assert_called_once()
    assert worker.closed and worker.connection.closed and not alive[0]
    worker.close()
    worker.connection.close.assert_called_once()
