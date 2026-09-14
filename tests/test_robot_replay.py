"""Named articulation mapping, asset mismatch detection and replay semantics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge._h1_motion import MotionRecorder
from embodiedforge.robot_replay import ReplaySession, RobotMotion

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("PIL")


@pytest.fixture
def recording(tmp_path):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="root" pos="0 0 1"><freejoint/><geom size=".1"/>
      <body name="arm" pos=".2 0 0"><joint name="shoulder" axis="0 0 1"/><geom size=".05"/>
        <body name="hand" pos=".2 0 0"><joint name="wrist" axis="0 1 0"/><geom size=".03"/></body>
      </body>
    </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)
    recorder = MotionRecorder(
        {
            "body_names": ["root", "arm", "hand"],
            # Intentionally different from the model's joint ordering.
            "joint_names": ["wrist", "shoulder"],
            "quaternion_order": "xyzw",
            "edges": [[0, 1], [1, 2]],
            "velocity_command": [0.5, 0, 0],
        }
    )
    expected = []
    for i, stamp in enumerate([0.02, 0.04, 0.09, 0.12]):
        data.qpos[:3] = [0.1 * i, 0, 1]
        data.qpos[7:] = [0.2 * i, -0.1 * i]
        mujoco.mj_forward(model, data)
        recorder.append(
            stamp,
            data.xpos[1:],
            np.roll(data.xquat[1:], -1, axis=-1),
            data.qpos[[8, 7]],
            i == 3,
            False,
        )
        expected.append(data.qpos.copy())
    path = tmp_path / "motion.npz"
    recorder.save(path)
    return path, model, np.stack(expected)


def rewrite(path, change):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in archive.files}
    meta = json.loads(str(values["metadata"]))
    change(values, meta)
    values["metadata"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **values)


def test_named_joint_mapping_and_full_fk_validation(recording):
    path, model, expected = recording
    motion = RobotMotion(path, model)
    np.testing.assert_allclose(motion.qpos, expected, atol=1e-7)
    assert motion.max_position_error < 1e-6
    assert motion.terminated and not motion.truncated
    assert motion.details(3)["terminal"] == "终止"
    assert motion.details(1)["command"] == [0.5, 0, 0]


@pytest.mark.parametrize(
    "corruption,error",
    [
        ("joint", "Missing or incompatible"),
        ("duplicate", "unique nonempty"),
        ("root", "root body"),
        ("quaternion", "unit norm"),
        ("ordering", "explicit xyzw"),
        ("pose", "pose mismatch at frame 2"),
        ("rotation", "pose mismatch at frame 2"),
    ],
)
def test_bad_recording_or_incompatible_asset_rejected(recording, corruption, error):
    path, model, _ = recording

    def change(arrays, meta):
        if corruption == "joint":
            meta["joint_names"][0] = "unknown"
        elif corruption == "duplicate":
            meta["joint_names"][0] = "shoulder"
        elif corruption == "root":
            meta["body_names"][0] = "other"
        elif corruption == "ordering":
            meta["quaternion_order"] = "wxyz"
        elif corruption == "quaternion":
            arrays["quaternions"][1, 0] *= 2
        elif corruption == "pose":
            arrays["positions"][2, 1, 0] += 0.1
        elif corruption == "rotation":
            arrays["quaternions"][2, 1] = [1, 0, 0, 0]

    rewrite(path, change)
    with pytest.raises(ValueError, match=error):
        RobotMotion(path, model)


def test_replay_uses_recorded_timestamps_and_preserves_terminal(recording):
    path, model, _ = recording
    motion = RobotMotion(path, model)
    session = ReplaySession([motion, motion])
    session.advance(0.021)
    assert session.indices.tolist() == [1, 1]
    session.advance(0.04)
    assert session.indices.tolist() == [1, 1]  # next frame is .09, not .06
    session.advance()
    assert session.indices.tolist() == [2, 2]
    session.seek(0, 0)
    assert session.indices.tolist() == [0, 2]
    session.advance(100)
    assert session.indices.tolist() == [3, 3] and session.ended
    session.advance()
    assert session.indices.tolist() == [3, 3]
    session.seek(1, 0)
    assert not session.ended and session.indices.tolist() == [3, 0]


def test_replay_publishes_selected_robot_and_no_invented_reward(recording):
    path, model, _ = recording
    motion = RobotMotion(path, model)
    session = ReplaySession([motion, motion])
    session.seek(1, 2)
    calls = []
    viewer = SimpleNamespace(
        selected_env=1, update=lambda *args, **kw: calls.append((args, kw))
    )
    session.update(viewer)
    args, kwargs = calls[0]
    np.testing.assert_allclose(args[0].qpos[1], motion.qpos[2])
    assert kwargs["display_state"]["reward"] is None
    assert kwargs["display_state"]["replay"]["frame"] == 2


@pytest.mark.parametrize(
    "env_id,frame", [(True, 0), (2, 0), (0, True), (0, -1), (0, 4)]
)
def test_invalid_seek(recording, env_id, frame):
    path, model, _ = recording
    session = ReplaySession([RobotMotion(path, model)])
    with pytest.raises(ValueError):
        session.seek(env_id, frame)


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
def test_replay_rejects_invalid_clock_interval(recording, seconds):
    path, model, _ = recording
    playback = ReplaySession([RobotMotion(path, model)])
    with pytest.raises(ValueError, match="finite and nonnegative"):
        playback.advance(seconds)


def test_native_visual_primitives_produce_finite_indexed_meshes():
    pytest.importorskip("newton")
    from embodiedforge.viewers.robot import RobotFrames

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <geom type="sphere" size=".1"/>
      <geom type="box" size=".1 .2 .3"/>
      <geom type="capsule" size=".1 .2"/>
      <geom type="cylinder" size=".1 .2"/>
      <geom type="ellipsoid" size=".1 .2 .3"/>
    </worldbody></mujoco>""")
    renderer = RobotFrames.__new__(RobotFrames)
    renderer.model, renderer.mujoco = model, mujoco
    for i in range(model.ngeom):
        vertices, indices = renderer._geometry(i)
        assert vertices.shape[1] == 3 and np.isfinite(vertices).all()
        assert indices.size % 3 == 0
        assert 0 <= indices.min() <= indices.max() < len(vertices)
