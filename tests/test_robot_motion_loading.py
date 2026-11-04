"""Robot replay validates one archive without constructing offline replay JSON."""

import hashlib
from collections import Counter

import numpy as np
import pytest

from embodiedforge import _h1_motion, robot_replay

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("PIL")


@pytest.mark.parametrize("fortran", [False, True])
def test_robot_motion_reads_each_member_once_without_json_conversion(
    tmp_path, monkeypatch, fortran
):
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root" pos="0 0 1"><freejoint/><geom size=".1"/>
        <body name="arm" pos=".2 0 0"><joint name="hinge" axis="0 0 1"/>
          <geom size=".05"/></body>
      </body></worldbody></mujoco>""")
    state = mujoco.MjData(model)
    recorder = _h1_motion.MotionRecorder(
        {
            "body_names": ["root", "arm"],
            "joint_names": ["hinge"],
            "edges": [[0, 1]],
            "quaternion_order": "xyzw",
        }
    )
    expected_qpos = []
    for index in range(3):
        state.qpos[0], state.qpos[-1] = index * 0.1, index * 0.2
        mujoco.mj_forward(model, state)
        recorder.append(
            (index + 1) * 0.02,
            state.xpos[1:],
            np.roll(state.xquat[1:], -1, axis=-1),
            state.qpos[-1:],
            index == 2,
            False,
        )
        expected_qpos.append(state.qpos.copy())
    path = tmp_path / "motion.npz"
    recorder.save(path)
    if fortran:
        with np.load(path, allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        for key in ("positions", "quaternions", "joints"):
            fields[key] = np.asfortranarray(fields[key])
        np.savez_compressed(path, **fields)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original_load = np.load
    with original_load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__
    archive_opens, member_reads = [], []

    def load(*args, **kwargs):
        archive_opens.append(args[0])
        return original_load(*args, **kwargs)

    def getitem(self, key):
        member_reads.append(key)
        return original_getitem(self, key)

    def reject_json_rounding(*args, **kwargs):
        raise AssertionError("Robot replay must not build offline JSON positions")

    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(archive_type, "__getitem__", getitem)
    monkeypatch.setattr(np, "round", reject_json_rounding)
    motion = robot_replay.RobotMotion(path, model)
    assert archive_opens == [path]
    assert Counter(member_reads) == {
        key: 1
        for key in (
            "metadata",
            "time",
            "positions",
            "quaternions",
            "joints",
            "terminated",
            "truncated",
        )
    }
    np.testing.assert_allclose(motion.qpos, expected_qpos, atol=1e-7)
    assert motion.time.flags.owndata and motion.time.flags.c_contiguous
    assert motion.positions.flags.owndata and motion.positions.flags.c_contiguous
    assert motion.terminated and not motion.truncated
    assert robot_replay.load_motion is _h1_motion.load_motion
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
