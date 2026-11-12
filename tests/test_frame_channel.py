"""Framed renderer messages must not outlive their communication budget."""

import errno
import os
import pickle
import socket
import struct
import sys
from multiprocessing.reduction import ForkingPickler
from threading import Event, Thread
from time import monotonic

import numpy as np
import pytest

from embodiedforge.viewers import _frame_channel as channel


@pytest.fixture
def endpoints():
    pair = channel.channel_pair()
    try:
        yield pair
    finally:
        for endpoint in pair:
            endpoint.close()


def bounded_operation(operation, peer):
    """Close the peer even if a regression blocks a socket operation."""
    results, errors = [], []

    def run():
        try:
            results.append(operation())
        except BaseException as error:
            errors.append(error)

    thread = Thread(target=run, daemon=True)
    thread.start()
    try:
        thread.join(timeout=2)
        timely = not thread.is_alive()
    finally:
        peer.close()
        thread.join(timeout=1)
    assert not thread.is_alive(), "Frame operation survived closing its peer"
    assert timely, "Frame operation ignored its absolute deadline"
    return results, errors


def test_roundtrip_preserves_snapshot_objects_and_owns_array_storage(endpoints):
    sender, receiver = endpoints
    value = {"camera": (-90.0, 45.0), "state": np.arange(48).reshape(4, 12)}
    sender.send(value, deadline=monotonic() + 1)
    received = receiver.receive(deadline=monotonic() + 1)
    assert received["camera"] == value["camera"]
    np.testing.assert_array_equal(received["state"], value["state"])
    received["state"].fill(-1)
    assert (value["state"] >= 0).all()
    receiver.send((True, b"owned frame"), deadline=monotonic() + 1)
    assert sender.receive(deadline=monotonic() + 1) == (True, b"owned frame")


def test_encoded_size_limit_accepts_exact_boundary_and_rejects_before_sending(
    endpoints,
):
    sender, receiver = endpoints
    value = b"small frame"
    limit = len(ForkingPickler.dumps(value))
    sender.LIMIT = receiver.LIMIT = limit
    sender.send(value, deadline=monotonic() + 1)
    assert receiver.receive(deadline=monotonic() + 1) == value
    with pytest.raises(ValueError, match="exceeds"):
        sender.send(value + b"x", deadline=monotonic() + 1)
    assert sender.closed
    # Rejected encoding emits neither a header nor a partial message.
    assert receiver.socket.recv(1) == b""


@pytest.mark.parametrize("declared", [0, 65])
def test_invalid_length_is_rejected_without_waiting_for_or_allocating_a_body(
    endpoints, declared
):
    receiver, peer = endpoints
    receiver.LIMIT = 64
    peer.socket.sendall(struct.pack("!I", declared))
    with pytest.raises(ValueError, match="Invalid frame message length"):
        receiver.receive(deadline=monotonic() + 0.1)
    assert receiver.closed


@pytest.mark.parametrize("fragment", [b"\x00\x00", struct.pack("!I", 64) + b"partial"])
def test_partial_header_or_payload_cannot_block_after_deadline(endpoints, fragment):
    receiver, peer = endpoints
    peer.socket.sendall(fragment)
    results, errors = bounded_operation(
        lambda: receiver.receive(deadline=monotonic() + 0.05), peer
    )
    assert not results
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert receiver.closed


def test_unread_peer_cannot_block_large_send(endpoints):
    sender, peer = endpoints
    sender.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
    results, errors = bounded_operation(
        lambda: sender.send(b"x" * (2 * 1024**2), deadline=monotonic() + 0.05),
        peer,
    )
    assert not results
    assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
    assert sender.closed


def test_slow_progress_does_not_restart_the_receive_budget(endpoints):
    receiver, peer = endpoints
    stop, begun = Event(), Event()
    wire = struct.pack("!I", 100) + b"x" * 100

    def trickle():
        try:
            for byte in wire:
                peer.socket.sendall(bytes([byte]))
                begun.set()
                if stop.wait(0.01):
                    break
        except OSError:
            pass  # Receiver shutdown is expected when its deadline expires.

    producer = Thread(target=trickle, daemon=True)
    producer.start()
    try:
        assert begun.wait(timeout=1)
        results, errors = bounded_operation(
            lambda: receiver.receive(deadline=monotonic() + 0.06), peer
        )
        assert not results
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert receiver.closed
    finally:
        stop.set()
        peer.close()
        producer.join(timeout=1)
    assert not producer.is_alive()


def test_successful_short_reads_also_check_elapsed_time(monkeypatch):
    clock = [0.0]
    wire = bytearray(struct.pack("!I", 100) + b"x" * 100)

    class FragmentedSocket:
        def setblocking(self, blocking):
            assert not blocking

        def settimeout(self, timeout):
            assert 0 < timeout <= 0.1

        def recv_into(self, view):
            view[0] = wire.pop(0)
            clock[0] += 0.02
            return 1

        def close(self):
            pass

    monkeypatch.setattr(channel, "monotonic", lambda: clock[0])
    receiver = channel.FrameChannel(FragmentedSocket())
    with pytest.raises(TimeoutError):
        receiver.receive(deadline=0.1)
    assert receiver.closed and wire


def test_encoding_time_is_charged_before_any_bytes_are_sent(endpoints, monkeypatch):
    sender, receiver = endpoints
    clock = [0.0]
    monkeypatch.setattr(channel, "monotonic", lambda: clock[0])

    class SlowValue:
        def __reduce_ex__(self, protocol):
            clock[0] = 2.0
            return int, (1,)

    with pytest.raises(TimeoutError):
        sender.send(SlowValue(), deadline=1.0)
    assert sender.closed
    assert receiver.socket.recv(1) == b""


def test_decoding_time_is_charged_before_returning_a_value(endpoints, monkeypatch):
    sender, receiver = endpoints
    sender.send("value", deadline=monotonic() + 1)
    clock = [0.0]
    monkeypatch.setattr(channel, "monotonic", lambda: clock[0])
    decode = ForkingPickler.loads

    def delayed_decode(payload):
        value = decode(payload)
        clock[0] = 2.0
        return value

    monkeypatch.setattr(channel.ForkingPickler, "loads", delayed_decode)
    with pytest.raises(TimeoutError):
        receiver.receive(deadline=1.0)
    assert receiver.closed


@pytest.mark.parametrize("wire", [b"", struct.pack("!I", 8) + b"short"])
def test_peer_eof_closes_the_channel(endpoints, wire):
    receiver, peer = endpoints
    if wire:
        peer.socket.sendall(wire)
    peer.close()
    with pytest.raises(EOFError):
        receiver.receive(deadline=monotonic() + 1)
    assert receiver.closed


def test_invalid_pickle_closes_the_channel(endpoints):
    receiver, peer = endpoints
    peer.socket.sendall(struct.pack("!I", 1) + b"!")
    with pytest.raises(pickle.UnpicklingError):
        receiver.receive(deadline=monotonic() + 1)
    assert receiver.closed


@pytest.mark.parametrize("retry", ["close", "send", "receive"])
def test_codec_failure_preserves_primary_error_and_allows_close_retry(caplog, retry):
    primary = KeyboardInterrupt("serialization interrupted")

    class BadValue:
        def __reduce_ex__(self, protocol):
            raise primary

    class InterruptedClose:
        attempts = 0

        def setblocking(self, blocking):
            pass

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("descriptor close interrupted")

    endpoint = channel.FrameChannel(InterruptedClose())
    with pytest.raises(KeyboardInterrupt) as caught:
        endpoint.send(BadValue(), deadline=monotonic() + 1)
    assert caught.value is primary
    assert not endpoint.closed and endpoint.failed
    assert "preserving its error" in caplog.text
    if retry == "send":
        with pytest.raises(RuntimeError, match="channel failed"):
            endpoint.send(b"never written", deadline=monotonic() + 1)
    elif retry == "receive":
        with pytest.raises(RuntimeError, match="channel failed"):
            endpoint.receive(deadline=monotonic() + 1)
    endpoint.close()
    assert endpoint.closed and endpoint.socket.attempts == 2
    endpoint.close()
    assert endpoint.socket.attempts == 2


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux descriptor limit"
)
def test_receive_timeout_accepts_socket_descriptor_above_fd_setsize(endpoints):
    import fcntl
    import resource

    minimum = 1024
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit <= minimum:
        pytest.skip("Current descriptor limit cannot allocate a high socket descriptor")
    original, peer = endpoints
    try:
        descriptor = fcntl.fcntl(original.socket.fileno(), fcntl.F_DUPFD, minimum)
    except OSError as error:
        if error.errno in (errno.EMFILE, errno.ENFILE, errno.EINVAL):
            pytest.skip(f"A high descriptor is unavailable: {error}")
        raise
    try:
        os.set_inheritable(descriptor, False)
        high_socket = socket.socket(
            original.socket.family,
            original.socket.type,
            original.socket.proto,
            fileno=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise
    original.close()
    receiver = channel.FrameChannel(high_socket)
    try:
        assert receiver.socket.fileno() >= minimum
        results, errors = bounded_operation(
            lambda: receiver.receive(deadline=monotonic() + 0.05), peer
        )
        assert not results
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError)
        assert receiver.closed
    finally:
        receiver.close()


def test_second_wrapper_failure_releases_both_raw_sockets(monkeypatch):
    pair = socket.socketpair()
    original = channel.FrameChannel
    primary = RuntimeError("second wrapper failed")
    calls = 0

    def wrap(sock):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise primary
        return original(sock)

    monkeypatch.setattr(channel.socket, "socketpair", lambda: pair)
    monkeypatch.setattr(channel, "FrameChannel", wrap)
    try:
        with pytest.raises(RuntimeError) as caught:
            channel.channel_pair()
        assert caught.value is primary
        assert all(endpoint.fileno() == -1 for endpoint in pair)
    finally:
        for endpoint in pair:
            endpoint.close()
