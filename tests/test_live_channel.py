"""IPC framing must be bounded and preserve messages across partial reads."""

import socket
import struct
import threading

import pytest

from embodiedforge._live_channel import JsonChannel


def test_partial_json_reads_and_multiple_messages():
    left, right = socket.socketpair()
    left.settimeout(2)
    with left, right:
        payload = '{"label":"策略","step":3}'.encode()

        def sender():
            for byte in struct.pack("!I", len(payload)) + payload:
                right.sendall(bytes([byte]))
            JsonChannel(right).send({"next": 4})

        thread = threading.Thread(target=sender)
        thread.start()
        reader = JsonChannel(left)
        assert reader.receive() == {"label": "策略", "step": 3}
        assert reader.receive() == {"next": 4}
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("size", [0, JsonChannel.LIMIT + 1])
def test_invalid_length_rejected_before_payload_read(size):
    left, right = socket.socketpair()
    with left, right:
        right.sendall(struct.pack("!I", size))
        with pytest.raises(ValueError, match="length"):
            JsonChannel(left).receive()


def test_disconnect_and_nonfinite_send():
    left, right = socket.socketpair()
    with left:
        right.close()
        with pytest.raises(EOFError):
            JsonChannel(left).receive()
        with pytest.raises(ValueError):
            JsonChannel(left).send({"bad": float("nan")})


def test_policy_launcher_preserves_venv_symlink_and_cleans_failed_startup(
    tmp_path, monkeypatch
):
    import os
    import sys
    from pathlib import Path

    from embodiedforge import go1_live
    from embodiedforge.recipes import sha256

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fake weights: worker deliberately rejects startup")
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    manifest = {
        "result": {
            "checkpoint_sha256": sha256(checkpoint),
            "runtime": {"executable": str(executable)},
        }
    }
    monkeypatch.setattr(
        go1_live, "checkpoint_input", lambda *args: (checkpoint, manifest)
    )
    captured = {}

    class Process:
        def poll(self):
            return 1

        def wait(self, timeout):
            captured["waited"] = True
            return 1

    def launch(command, **kwargs):
        captured["command"] = command
        duplicate = socket.socket(fileno=os.dup(kwargs["pass_fds"][0]))

        def worker():
            with duplicate:
                channel = JsonChannel(duplicate)
                captured["request"] = channel.receive()
                channel.send({"ok": False, "error": "Injected startup failure"})

        thread = threading.Thread(target=worker)
        captured["thread"] = thread
        thread.start()
        return Process()

    monkeypatch.setattr(go1_live.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="Injected startup failure"):
        go1_live.LivePolicyProcess(tmp_path)
    captured["thread"].join(timeout=2)
    assert captured["command"][0] == str(executable)
    assert captured["waited"]
    assert not Path(captured["request"]["model_path"]).parent.exists()
