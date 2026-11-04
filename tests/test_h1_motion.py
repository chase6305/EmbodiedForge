import builtins
import json
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge import _h1_motion as h1_motion
from embodiedforge._h1_motion import MotionRecorder, load_motion, render_motion
from embodiedforge.runtime import artifacts


def recorder():
    return MotionRecorder(
        {
            "schema": 1,
            "case": "custom",
            "seed": 0,
            "env_id": 0,
            "velocity_command": [0.5, 0, 0],
            "body_names": ["pelvis", "left_ankle"],
            "joint_names": ["knee"],
            "edges": [[0, 1]],
            "quaternion_order": "xyzw",
        }
    )


def append(motion, time, *, terminated=False):
    motion.append(
        time, [[0, 0, 1], [0, 0, 0.1]], [[0, 0, 0, 1]] * 2, [0.2], terminated, False
    )


def test_recording_stops_at_terminal_frame_and_owns_its_arrays(tmp_path):
    motion = recorder()
    positions = np.ones((2, 3))
    motion.append(0.02, positions, [[0, 0, 0, 1]] * 2, [0.2], False, False)
    positions[:] = 99
    append(motion, 0.04, terminated=True)
    append(motion, 0.06)
    path = tmp_path / "motion.npz"
    motion.save(path)
    loaded = load_motion(path)
    assert loaded["time"] == [0.02, 0.04]
    assert loaded["positions"][0] == [[1, 1, 1]] * 2
    assert loaded["terminated"]
    with np.load(path, allow_pickle=False) as data:
        assert data["joints"].shape == (2, 1)
        assert data["quaternions"].shape == (2, 2, 4)


@pytest.mark.parametrize("stage", ["write", "close", "replace"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_failed_save_preserves_published_motion_and_allows_retry(
    tmp_path, monkeypatch, stage, error_type
):
    motion = recorder()
    append(motion, 0.02)
    path = tmp_path / "motion.npz"
    temporary = tmp_path / "motion.npz.attempt.tmp"
    motion.save(path)
    published = path.read_bytes()
    append(motion, 0.04)
    primary = error_type("motion publication failed")

    def fail_write(stream, **arrays):
        stream.write(b"partial archive")
        raise primary

    def fail_replace(source, target):
        assert source == str(temporary) and target == str(path)
        with np.load(source, allow_pickle=False) as archive:
            assert archive["time"].tolist() == [0.02, 0.04]
        raise primary

    original_open = builtins.open

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *exc):
            self.stream.close()
            raise primary

    def open_with_failing_close(target, *args, **kwargs):
        stream = original_open(target, *args, **kwargs)
        return FailingClose(stream) if target == str(temporary) else stream

    with monkeypatch.context() as patch:
        patch.setattr(h1_motion, "uuid4", lambda: SimpleNamespace(hex="attempt"))
        if stage == "write":
            patch.setattr(np, "savez_compressed", fail_write)
        elif stage == "close":
            patch.setattr(builtins, "open", open_with_failing_close)
        else:
            patch.setattr(h1_motion.os, "replace", fail_replace)
        with pytest.raises(error_type) as caught:
            motion.save(path)
        assert caught.value is primary
    assert path.read_bytes() == published
    assert not temporary.exists()
    motion.save(path)
    assert load_motion(path)["time"] == [0.02, 0.04]
    assert not temporary.exists()


@pytest.mark.parametrize("published", [False, True])
def test_save_never_removes_or_overwrites_an_existing_temporary(
    tmp_path, monkeypatch, published
):
    motion = recorder()
    append(motion, 0.02)
    path = tmp_path / "motion.npz"
    if published:
        motion.save(path)
    previous = path.read_bytes() if published else None
    temporary = tmp_path / "motion.npz.existing.tmp"
    temporary.write_bytes(b"another attempt owns this file")
    monkeypatch.setattr(h1_motion, "uuid4", lambda: SimpleNamespace(hex="existing"))
    with pytest.raises(FileExistsError):
        motion.save(path)
    assert temporary.read_bytes() == b"another attempt owns this file"
    assert (path.read_bytes() if path.exists() else None) == previous


@pytest.mark.parametrize(
    "cleanup_error",
    [PermissionError("cleanup denied"), KeyboardInterrupt(), SystemExit(3)],
)
def test_cleanup_failure_does_not_replace_the_motion_save_error(
    tmp_path, monkeypatch, caplog, cleanup_error
):
    motion = recorder()
    append(motion, 0.02)
    path = tmp_path / "motion.npz"
    motion.save(path)
    published = path.read_bytes()
    primary = OSError("motion write failed")

    def fail_write(stream, **arrays):
        stream.write(b"partial archive")
        raise primary

    def fail_cleanup(target):
        assert target == str(tmp_path / "motion.npz.attempt.tmp")
        raise cleanup_error

    with monkeypatch.context() as patch:
        patch.setattr(h1_motion, "uuid4", lambda: SimpleNamespace(hex="attempt"))
        patch.setattr(np, "savez_compressed", fail_write)
        patch.setattr(artifacts.os, "unlink", fail_cleanup)
        with pytest.raises(OSError) as caught:
            motion.save(path)
        assert caught.value is primary
    assert path.read_bytes() == published
    assert "Could not remove partial H1 motion" in caplog.text


def test_cancellation_after_rename_preserves_another_writers_temporary(
    tmp_path, monkeypatch
):
    first, second = recorder(), recorder()
    append(first, 0.02)
    append(second, 0.02)
    append(second, 0.04)
    path = tmp_path / "motion.npz"
    first_temporary = str(tmp_path / "motion.npz.first.tmp")
    second_temporary = str(tmp_path / "motion.npz.second.tmp")
    tokens = iter(("first", "second"))
    monkeypatch.setattr(h1_motion, "uuid4", lambda: SimpleNamespace(hex=next(tokens)))
    renamed, second_created, finish_second = (threading.Event() for _ in range(3))
    failures = []
    primary = KeyboardInterrupt("cancelled after successful rename")
    original_save, original_replace = np.savez_compressed, os.replace

    def pause_second_write(stream, **arrays):
        if stream.name == second_temporary:
            second_created.set()
            assert finish_second.wait(3), "second writer was not released"
        original_save(stream, **arrays)

    def replace_then_interrupt(source, target):
        result = original_replace(source, target)
        if source == first_temporary:
            renamed.set()
            assert second_created.wait(3), "second writer did not claim its file"
            raise primary
        return result

    def second_writer():
        try:
            assert renamed.wait(3), "first writer did not publish"
            second.save(path)
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(np, "savez_compressed", pause_second_write)
    monkeypatch.setattr(h1_motion.os, "replace", replace_then_interrupt)
    worker = threading.Thread(target=second_writer)
    worker.start()
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            first.save(path)
        assert caught.value is primary
        assert not os.path.exists(first_temporary)
        assert os.path.isfile(second_temporary)
        assert load_motion(path)["time"] == [0.02]
    finally:
        finish_second.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and not failures
    assert load_motion(path)["time"] == [0.02, 0.04]
    assert list(tmp_path.iterdir()) == [path]


def test_recording_rejects_nonfinite_and_unordered_data():
    motion = recorder()
    append(motion, 0.02)
    with pytest.raises(ValueError, match="timestamps"):
        append(motion, 0.01)
    with pytest.raises(ValueError, match="Non-finite"):
        motion.append(0.04, [[np.nan, 0, 0]] * 2, [[0, 0, 0, 1]] * 2, [0], False, False)


def test_loader_rejects_frames_after_auto_reset(tmp_path):
    motion = recorder()
    append(motion, 0.02)
    append(motion, 0.04)
    path = tmp_path / "motion.npz"
    motion.save(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays["terminated"] = np.array([True, False])
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="after a terminal"):
        load_motion(path)


def test_replay_escapes_embedded_labels_and_preserves_existing_output(tmp_path):
    motion = recorder()
    attack = '</script><script>alert("unexpected")</script>'
    motion.metadata["case"] = attack
    append(motion, 0.02)
    path, output = tmp_path / "motion.npz", tmp_path / "motion.html"
    motion.save(path)
    render_motion(path, output)
    html = output.read_text()
    assert attack not in html
    payload = html.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(payload)["metadata"]["case"] == attack
    with pytest.raises(FileExistsError):
        render_motion(path, output)


def test_invalid_topology_is_rejected(tmp_path):
    motion = recorder()
    motion.metadata["edges"] = [[0, 3]]
    append(motion, 0.02)
    path = tmp_path / "motion.npz"
    motion.save(path)
    with pytest.raises(ValueError, match="connection"):
        load_motion(path)


def test_dynamic_motion_schedule_covers_frames_without_gaps(tmp_path):
    motion = recorder()
    motion.metadata["velocity_command"] = None
    motion.metadata["command_schedule"] = [
        {"segment": "stand", "start_step": 0, "end_step": 1, "command": [0, 0, 0]},
        {"segment": "forward", "start_step": 1, "end_step": 2, "command": [0.5, 0, 0]},
    ]
    append(motion, 0.02)
    append(motion, 0.04)
    path = tmp_path / "valid.npz"
    motion.save(path)
    assert len(load_motion(path)["metadata"]["command_schedule"]) == 2
    motion.metadata["command_schedule"][1]["start_step"] = 2
    path = tmp_path / "gap.npz"
    motion.save(path)
    with pytest.raises(ValueError, match="command schedule"):
        load_motion(path)
