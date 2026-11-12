"""Real stopped peers must not trap renderer exchange or shutdown in socket I/O."""

import multiprocessing
import socket
import struct
import threading
import time
from contextlib import contextmanager

import pytest

from embodiedforge.viewers import frames
from embodiedforge.viewers._frame_channel import FrameChannel, channel_pair


def _stalled_worker(sock, mode, prefix_ready):
    with sock:
        connection = FrameChannel(sock)
        connection.send((True, False), deadline=time.monotonic() + 10)
        if mode.startswith("partial-"):
            # Prequeue only a reply prefix, like the live-policy partial-read
            # regressions. A scheduling delay must not turn this into a test of
            # an entirely silent peer instead of an incomplete frame.
            prefix = (
                b"\x00\x00"
                if mode == "partial-header"
                else struct.pack("!I", 128) + b"x"
            )
            sock.sendall(prefix)
        prefix_ready.set()
        # Keep the peer descriptor alive without reading any remaining request
        # or completing its response. The test owner must terminate this child.
        time.sleep(60)


@contextmanager
def _owner(monkeypatch, mode):
    context = multiprocessing.get_context("spawn")
    owner = object.__new__(frames.IsolatedFrames)
    owner.name = "raster"
    owner.closed = False
    owner.connection, child = channel_pair()
    owner.connection.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    prefix_ready = context.Event()
    owner.process = context.Process(
        target=_stalled_worker, args=(child.socket, mode, prefix_ready)
    )
    join = owner.process.join
    monkeypatch.setattr(frames, "RENDER_TIMEOUT", 0.1)
    monkeypatch.setattr(frames, "CLOSE_SEND_TIMEOUT", 0.1)
    try:
        owner.process.start()
        # Instance attributes are serialized by spawn. Install this local
        # injection only after launch; keep the original join for final cleanup.
        # Shorten only the graceful wait, retaining both escalation waits.
        monkeypatch.setattr(
            owner.process,
            "join",
            lambda timeout: join(0.1 if timeout == 5 else timeout),
        )
        child.close()
        assert owner._receive(10) is False
        assert prefix_ready.wait(timeout=10), "Stopped peer did not finish setup"
        yield owner
    finally:
        child.close()
        owner.connection.close()
        try:
            if owner.process.is_alive():
                owner.process.kill()
            if owner.process.pid is not None:
                join(timeout=3)
            owner.process.close()
        except ValueError:
            pass  # Successful owner cleanup already released the process handle.


def _bounded_call(owner, operation):
    """A baseline regression must fail the test without stranding its own child."""
    finished = threading.Event()
    result, errors = [], []

    def run():
        try:
            result.append(operation())
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()

    caller = threading.Thread(target=run, daemon=True)
    caller.start()
    # The operation has a 0.1 s I/O budget, a shortened graceful wait, and up to
    # two 2 s escalation waits. Leave scheduling margin beyond that whole path.
    completed = finished.wait(timeout=10)
    if not completed:
        # Kill only the deliberately stalled child created by this test. Closing
        # its peer releases a blocked send/read even in the unfixed implementation.
        try:
            owner.process.kill()
        except ValueError:
            pass
        finished.wait(timeout=3)
        if not finished.is_set():
            try:
                owner.connection.socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    caller.join(timeout=3)
    assert not caller.is_alive(), "Test exchange thread did not release its socket"
    assert completed, "Renderer operation exceeded its transport and cleanup deadlines"
    if errors:
        raise errors[0]
    return result[0]


def _assert_reaped(owner):
    assert owner.closed and owner.connection.closed
    assert owner.connection.socket.fileno() == -1
    with pytest.raises(ValueError, match="closed"):
        owner.process.is_alive()
    owner.close()
    with pytest.raises(RuntimeError, match="closed"):
        owner.render(None, {}, 0, frames.OrbitCamera())


@pytest.mark.parametrize("mode", ["partial-header", "partial-body"])
def test_partial_response_deadline_closes_and_reaps_renderer(monkeypatch, mode):
    with _owner(monkeypatch, mode) as owner:
        with pytest.raises(TimeoutError):
            _bounded_call(
                owner, lambda: owner.render(None, {}, 0, frames.OrbitCamera())
            )
        _assert_reaped(owner)


def test_stopped_reader_bounds_render_send_and_reaps_renderer(monkeypatch):
    with _owner(monkeypatch, "stopped-reader") as owner:
        with pytest.raises(TimeoutError):
            _bounded_call(
                owner,
                lambda: owner.render(
                    None, {"payload": b"x" * 1024**2}, 0, frames.OrbitCamera()
                ),
            )
        _assert_reaped(owner)


def test_full_send_buffer_cannot_block_renderer_shutdown(monkeypatch):
    with _owner(monkeypatch, "stopped-reader") as owner:
        sock = owner.connection.socket
        previous = sock.gettimeout()
        sock.setblocking(False)
        try:
            sent = 0
            for _ in range(1024):
                try:
                    sent += sock.send(b"x" * 65536)
                except BlockingIOError:
                    break
            else:
                pytest.fail("Stopped peer never applied socket backpressure")
            assert sent > 0
        finally:
            sock.settimeout(previous)
        _bounded_call(owner, owner.close)
        _assert_reaped(owner)
