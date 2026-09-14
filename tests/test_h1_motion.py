import json

import numpy as np
import pytest

from embodiedforge._h1_motion import MotionRecorder, load_motion, render_motion


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
