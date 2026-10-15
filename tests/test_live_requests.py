"""A failed exchange must never reuse a partially consumed response stream."""

import socket
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from embodiedforge._live_channel import JsonChannel
from embodiedforge.go1_live import LivePolicyProcess


@pytest.fixture
def session():
    owner = object.__new__(LivePolicyProcess)
    owner.closed = False
    owner.directory = tempfile.TemporaryDirectory(prefix="ef-live-request-test-")
    owner.socket, peer = socket.socketpair()
    owner.socket.settimeout(0.02)
    owner.channel = JsonChannel(owner.socket)
    owner.process = MagicMock()
    owner.process.poll.return_value = 0
    owner.state = {"step": [0]}
    try:
        yield owner, peer
    finally:
        owner.close()
        peer.close()


@pytest.mark.parametrize(
    "prefix", [b"", b"\x00\x00", struct.pack("!I", 40) + b'{"ok":']
)
def test_response_timeout_closes_session_and_prevents_stale_reply(session, prefix):
    owner, peer = session
    private = Path(owner.directory.name)
    peer.sendall(prefix)
    with pytest.raises(TimeoutError):
        owner.request("step")
    assert owner.closed
    assert owner.socket.fileno() == -1
    assert not private.exists()
    assert owner.state == {"step": [0]}
    with pytest.raises(RuntimeError, match="closed"):
        owner.request("step")


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        1,
        {},
        {"ok": 1},
        {"ok": True},
        {"ok": True, "state": []},
        {"ok": False},
        {"ok": False, "error": None},
    ],
)
def test_malformed_response_closes_session(session, value):
    owner, peer = session
    JsonChannel(peer).send(value)
    with pytest.raises(ValueError, match="response"):
        owner.request("step")
    assert owner.closed
    assert owner.state == {"step": [0]}


@pytest.mark.parametrize("body", [b"\xff", b"{", b""])
def test_corrupt_response_closes_session(session, body):
    owner, peer = session
    peer.sendall(struct.pack("!I", len(body)) + body)
    with pytest.raises(ValueError):
        owner.request("step")
    assert owner.closed


def test_worker_rejection_preserves_session_and_accepts_next_response(session):
    owner, peer = session
    channel = JsonChannel(peer)
    channel.send({"ok": False, "error": "Unknown live policy operation"})
    with pytest.raises(RuntimeError, match="Unknown live policy operation"):
        owner.request("invalid")
    assert not owner.closed
    channel.send({"ok": True, "state": {"step": [1]}})
    assert owner.request("step") == {"step": [1]}
    assert channel.receive() == {"action": "invalid"}
    assert channel.receive() == {"action": "step"}


@pytest.mark.parametrize("failure", ["timeout", "eof", "invalid"])
def test_failed_response_reaps_real_worker_and_private_files(monkeypatch, failure):
    owner = object.__new__(LivePolicyProcess)
    owner.closed = False
    owner.directory = tempfile.TemporaryDirectory(prefix="ef-live-request-child-")
    private = Path(owner.directory.name)
    owner.socket, child = socket.socketpair()
    owner.socket.settimeout(0.2)
    owner.channel = JsonChannel(owner.socket)
    owner.state = {"step": [0]}
    script = """
import socket, sys, time
sock = socket.socket(fileno=int(sys.argv[1]))
sock.recv(4096)
if sys.argv[2] == 'timeout':
    sock.sendall(b'\\x00\\x00')
elif sys.argv[2] == 'eof':
    sock.shutdown(socket.SHUT_WR)
else:
    sock.sendall(b'\\x00\\x00\\x00\\x01{')
time.sleep(60)
"""
    owner.process = subprocess.Popen(
        [sys.executable, "-c", script, str(child.fileno()), failure],
        pass_fds=(child.fileno(),),
    )
    child.close()
    wait = owner.process.wait
    monkeypatch.setattr(owner.process, "wait", lambda timeout: wait(min(timeout, 0.1)))
    try:
        with pytest.raises((TimeoutError, EOFError, ValueError)):
            owner.request("step")
        assert owner.closed
        assert owner.process.poll() is not None
        assert owner.socket.fileno() == -1
        assert not private.exists()
    finally:
        if owner.process.poll() is None:
            owner.process.kill()
            wait(5)
        owner.socket.close()
        owner.directory.cleanup()


@pytest.mark.parametrize("value", [float("nan"), object(), "x" * JsonChannel.LIMIT])
def test_local_serialization_rejection_preserves_stream(session, value):
    owner, peer = session
    with pytest.raises((TypeError, ValueError)):
        owner.request("velocity", value=value)
    assert not owner.closed
    JsonChannel(peer).send({"ok": True, "state": {"step": [1]}})
    assert owner.request("step") == {"step": [1]}
    assert JsonChannel(peer).receive() == {"action": "step"}


@pytest.mark.parametrize("stage", ["send", "receive"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_exchange_failure_preserves_primary_and_closes(
    session, monkeypatch, stage, error_type
):
    owner, _ = session
    primary = error_type("exchange interrupted")
    monkeypatch.setattr(owner.channel, stage, MagicMock(side_effect=primary))
    close = owner.close

    def close_then_fail():
        close()
        raise RuntimeError("secondary cleanup failure")

    monkeypatch.setattr(owner, "close", close_then_fail)
    try:
        with pytest.raises(error_type) as caught:
            owner.request("step")
        assert caught.value is primary
        assert owner.closed
    finally:
        monkeypatch.setattr(owner, "close", close)
