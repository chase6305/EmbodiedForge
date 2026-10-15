"""Live policy shutdown releases its child, socket and private files after faults."""

import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from embodiedforge import go1_live


@pytest.mark.parametrize("stage", ["settimeout", "asset-cache"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_prelaunch_failure_closes_both_real_sockets_with_retained_traceback(
    tmp_path, monkeypatch, stage, error_type
):
    from embodiedforge.recipes import sha256

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"snapshot")
    manifest = {
        "result": {
            "checkpoint_sha256": sha256(checkpoint),
            "runtime": {"executable": sys.executable},
        }
    }
    monkeypatch.setattr(go1_live, "checkpoint_input", lambda *a: (checkpoint, manifest))
    monkeypatch.setattr(go1_live, "recorded_go1_assets", lambda *a: None)
    monkeypatch.setattr(go1_live, "copy_go1_implementation", lambda *a: None)
    temporary_directory = tempfile.TemporaryDirectory
    monkeypatch.setattr(
        go1_live.tempfile,
        "TemporaryDirectory",
        lambda **kwargs: temporary_directory(dir=tmp_path, **kwargs),
    )
    parent, child = socket.socketpair()
    primary = error_type("prelaunch failed")

    class ParentSocket:
        def settimeout(self, timeout):
            if stage == "settimeout":
                raise primary
            parent.settimeout(timeout)

        def close(self):
            parent.close()

    monkeypatch.setattr(go1_live.socket, "socketpair", lambda: (ParentSocket(), child))
    if stage == "asset-cache":
        monkeypatch.setattr(
            go1_live, "reuse_asset_cache", MagicMock(side_effect=primary)
        )
    launch = MagicMock(side_effect=AssertionError("Must fail before launch"))
    monkeypatch.setattr(go1_live.subprocess, "Popen", launch)
    before = set(tmp_path.glob("embodiedforge-go1-live-*"))
    try:
        with pytest.raises(BaseException) as caught:
            go1_live.LivePolicyProcess(tmp_path)
        assert caught.value is primary
        # Keep caught/primary and their tracebacks alive: GC must not be needed
        # to release the inherited child end retained in the constructor frame.
        assert parent.fileno() == child.fileno() == -1
        assert set(tmp_path.glob("embodiedforge-go1-live-*")) == before
        launch.assert_not_called()
    finally:
        parent.close()
        child.close()


def session_with_fault(stage, error):
    owner = object.__new__(go1_live.LivePolicyProcess)
    owner.closed = False
    owner.socket = MagicMock()
    owner.directory = MagicMock()
    owner.process = MagicMock()
    alive = [True]
    owner.process.poll.side_effect = lambda: None if alive[0] else 0

    def wait(timeout):
        if alive[0]:
            raise subprocess.TimeoutExpired("test worker", timeout)
        return 0

    owner.process.wait.side_effect = wait
    owner.process.terminate.side_effect = lambda: alive.__setitem__(0, stage == "kill")
    owner.process.kill.side_effect = lambda: alive.__setitem__(0, False)
    target = (
        owner.directory
        if stage == "cleanup"
        else owner.socket
        if stage in ("sendall", "close", "settimeout")
        else owner.process
    )
    operation = getattr(target, stage)
    original = operation.side_effect
    calls = 0

    def interrupt_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise error
        if callable(original):
            return original(*args, **kwargs)

    operation.side_effect = interrupt_once
    return owner, alive


@pytest.mark.parametrize(
    "stage",
    ["poll", "wait", "terminate", "kill", "close", "cleanup", "sendall", "settimeout"],
)
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_shutdown_continues_after_one_cleanup_fault(stage, error_type):
    primary = error_type("interrupted cleanup")
    owner, alive = session_with_fault(stage, primary)
    with pytest.raises(error_type) as caught:
        owner.close()
    assert caught.value is primary
    assert not alive[0]
    assert owner.socket.close.call_count == (2 if stage == "close" else 1)
    assert owner.directory.cleanup.call_count == (2 if stage == "cleanup" else 1)
    owner.close()
    assert owner.directory.cleanup.call_count == (2 if stage == "cleanup" else 1)


def test_first_cleanup_error_survives_secondary_failures():
    primary = KeyboardInterrupt("wait interrupted")
    owner, alive = session_with_fault("wait", primary)
    owner.socket.close.side_effect = OSError("socket close failed")
    owner.directory.cleanup.side_effect = RuntimeError("directory cleanup failed")
    with pytest.raises(KeyboardInterrupt) as caught:
        owner.close()
    assert caught.value is primary
    assert not alive[0]
    assert owner.socket.close.call_count == owner.directory.cleanup.call_count == 2


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_interrupted_wait_reaps_real_worker_and_releases_socket_and_files(
    monkeypatch, error_type
):
    owner = object.__new__(go1_live.LivePolicyProcess)
    owner.closed = False
    owner.directory = tempfile.TemporaryDirectory(prefix="ef-live-cleanup-test-")
    directory = Path(owner.directory.name)
    (directory / "checkpoint.pt").write_bytes(b"private snapshot")
    owner.socket, child = socket.socketpair()
    child.close()
    owner.process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    wait = owner.process.wait
    primary = error_type("wait interrupted")
    calls = 0

    def interrupt_once(timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise primary
        return wait(timeout=timeout)

    monkeypatch.setattr(owner.process, "wait", interrupt_once)
    try:
        with pytest.raises(error_type) as caught:
            owner.close()
        assert caught.value is primary
        assert owner.process.poll() is not None
        assert owner.socket.fileno() == -1
        assert not directory.exists()
        owner.close()
    finally:
        if owner.process.poll() is None:
            owner.process.kill()
            wait(timeout=5)
        owner.socket.close()
        owner.directory.cleanup()


@pytest.mark.parametrize("cleanup_type", [OSError, KeyboardInterrupt, SystemExit])
def test_startup_preserves_launch_error_when_child_socket_close_also_fails(
    tmp_path, monkeypatch, cleanup_type
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    from embodiedforge.recipes import sha256

    manifest = {
        "result": {
            "checkpoint_sha256": sha256(checkpoint),
            "runtime": {"executable": sys.executable},
        }
    }
    monkeypatch.setattr(
        go1_live, "checkpoint_input", lambda *args: (checkpoint, manifest)
    )
    monkeypatch.setattr(go1_live, "recorded_go1_assets", lambda *args: None)
    monkeypatch.setattr(go1_live, "copy_go1_implementation", lambda *args: None)
    parent, child = MagicMock(), MagicMock()
    monkeypatch.setattr(go1_live.socket, "socketpair", lambda: (parent, child))
    child.close.side_effect = cleanup_type("child socket interrupted")
    primary = RuntimeError("Popen failed")
    monkeypatch.setattr(go1_live.subprocess, "Popen", MagicMock(side_effect=primary))
    before = set(Path(tempfile.gettempdir()).glob("embodiedforge-go1-live-*"))
    with pytest.raises(RuntimeError) as caught:
        go1_live.LivePolicyProcess(tmp_path)
    assert caught.value is primary
    parent.close.assert_called_once()
    assert set(Path(tempfile.gettempdir()).glob("embodiedforge-go1-live-*")) == before


@pytest.mark.parametrize("cleanup_type", [OSError, KeyboardInterrupt, SystemExit])
def test_startup_preserves_primary_when_session_cleanup_also_fails(
    tmp_path, monkeypatch, cleanup_type
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    from embodiedforge.recipes import sha256

    manifest = {
        "result": {
            "checkpoint_sha256": sha256(checkpoint),
            "runtime": {"executable": sys.executable},
        }
    }
    monkeypatch.setattr(
        go1_live, "checkpoint_input", lambda *args: (checkpoint, manifest)
    )
    monkeypatch.setattr(go1_live, "recorded_go1_assets", lambda *args: None)
    primary = ValueError("snapshot invalid")
    monkeypatch.setattr(
        go1_live, "copy_go1_implementation", MagicMock(side_effect=primary)
    )
    close = go1_live.LivePolicyProcess.close

    def close_then_fail(owner):
        close(owner)
        raise cleanup_type("cleanup interrupted")

    monkeypatch.setattr(go1_live.LivePolicyProcess, "close", close_then_fail)
    with pytest.raises(ValueError) as caught:
        go1_live.LivePolicyProcess(tmp_path)
    assert caught.value is primary


@pytest.mark.parametrize("stage", ["kill", "socket", "directory"])
def test_real_handles_close_after_an_interrupt_before_release(monkeypatch, stage):
    owner = object.__new__(go1_live.LivePolicyProcess)
    owner.closed = False
    owner.directory = tempfile.TemporaryDirectory(prefix="ef-live-handles-test-")
    directory = Path(owner.directory.name)
    owner.socket, peer = socket.socketpair()
    peer.close()
    script = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    owner.process = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE
    )
    assert owner.process.stdout.readline() == b"ready\n"
    kill, wait, cleanup = (
        owner.process.kill,
        owner.process.wait,
        owner.directory.cleanup,
    )
    sock = owner.socket
    primary = KeyboardInterrupt("release interrupted")
    calls = 0

    def interrupt_once(operation):
        def run(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise primary
            return operation(*args, **kwargs)

        return run

    if stage == "kill":
        # Skip the initial graceful timeout while still testing real terminate,
        # kill and wait operations against a SIGTERM-resistant subprocess.
        def fast_wait(timeout):
            return wait(timeout=0.01 if timeout == 5 else timeout)

        monkeypatch.setattr(owner.process, "wait", fast_wait)
        monkeypatch.setattr(owner.process, "kill", interrupt_once(kill))
    else:
        kill()
        wait(timeout=5)
        if stage == "directory":
            monkeypatch.setattr(owner.directory, "cleanup", interrupt_once(cleanup))
        else:

            class SocketProxy:
                close = staticmethod(interrupt_once(sock.close))

            owner.socket = SocketProxy()
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            owner.close()
        assert caught.value is primary
        assert owner.process.poll() is not None
        assert sock.fileno() == -1
        assert not directory.exists()
    finally:
        if owner.process.poll() is None:
            kill()
            wait(timeout=5)
        owner.process.stdout.close()
        sock.close()
        cleanup()


@pytest.mark.parametrize("launched", [False, True])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_startup_socket_interrupt_releases_both_real_descriptors(
    tmp_path, monkeypatch, launched, error_type
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    from embodiedforge.recipes import sha256

    manifest = {
        "result": {
            "checkpoint_sha256": sha256(checkpoint),
            "runtime": {"executable": sys.executable},
        }
    }
    monkeypatch.setattr(
        go1_live, "checkpoint_input", lambda *args: (checkpoint, manifest)
    )
    monkeypatch.setattr(go1_live, "recorded_go1_assets", lambda *args: None)
    monkeypatch.setattr(go1_live, "copy_go1_implementation", lambda *args: None)
    parent, child = socket.socketpair()
    primary = RuntimeError("launch failed")
    interrupted = error_type("child socket close interrupted")
    calls = 0

    class ChildSocket:
        fileno = staticmethod(child.fileno)

        @staticmethod
        def close():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise interrupted
            child.close()

    monkeypatch.setattr(go1_live.socket, "socketpair", lambda: (parent, ChildSocket()))
    process = MagicMock()
    process.poll.return_value = 0
    launch = (
        MagicMock(return_value=process) if launched else MagicMock(side_effect=primary)
    )
    monkeypatch.setattr(go1_live.subprocess, "Popen", launch)
    expected = interrupted if launched else primary
    try:
        with pytest.raises(type(expected)) as caught:
            go1_live.LivePolicyProcess(tmp_path)
        assert caught.value is expected
        assert parent.fileno() == child.fileno() == -1
        assert calls == 2
        if launched:
            process.wait.assert_called_once_with(timeout=5)
    finally:
        parent.close()
        child.close()
