"""Recorded poses and diagnostic history remain independent of borrowed inputs."""

import numpy as np
import pytest

from embodiedforge._h1_motion import MotionRecorder
from embodiedforge._wuji_diagnostics import TrialDiagnostics


@pytest.mark.parametrize("bodies", [2, 17])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed", "list"])
def test_motion_frames_own_contiguous_float32_storage(bodies, dtype, layout):
    rng = np.random.default_rng(41)
    inputs = [
        rng.normal(size=shape).astype(dtype)
        for shape in ((bodies, 3), (bodies, 4), (5,))
    ]
    if layout == "F":
        inputs = [np.asfortranarray(value) for value in inputs]
    elif layout == "strided":
        inputs = [np.repeat(value, 2, axis=0)[::2] for value in inputs]
    elif layout == "reversed":
        inputs = [value[::-1] for value in inputs]
    elif layout == "list":
        inputs = [value.tolist() for value in inputs]
    expected = [np.asarray(value, np.float32).copy() for value in inputs]
    for value in inputs:
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    recorder = MotionRecorder(
        {"body_names": [str(i) for i in range(bodies)], "joint_names": list("abcde")}
    )
    recorder.append(0.02, *inputs, False, False)
    for actual, wanted in zip(recorder.frames[0][1:4], expected, strict=True):
        assert (
            actual.dtype == np.float32
            and actual.flags.owndata
            and actual.flags.c_contiguous
        )
        np.testing.assert_array_equal(actual, wanted)
        for original in inputs:
            if isinstance(original, np.ndarray):
                assert not np.shares_memory(actual, original)
    for value in inputs:
        if isinstance(value, np.ndarray):
            value.flags.writeable = True
            value[:] = 91
        elif isinstance(value[0], list):
            value[0][0] = 91
        else:
            value[0] = 91
    for actual, wanted in zip(recorder.frames[0][1:4], expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64])
@pytest.mark.parametrize("layout", ["C", "strided", "reversed", "readonly"])
def test_diagnostics_snapshot_lists_and_previous_action_own_inputs(dtype, layout):
    origin = np.array([1, 2, 3], dtype=dtype)
    quaternion = np.array([1, 0, 0, 0], dtype=dtype)
    goal = np.array([0, 1, 0, 0], dtype=dtype)
    audit = TrialDiagnostics()
    audit.start(origin, quaternion, goal)
    origin[:], quaternion[:], goal[:] = 9, 9, 9
    assert audit.current["initial_position"] == [1.0, 2.0, 3.0]
    assert audit.current["initial_quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert audit.current["goal"] == [0.0, 1.0, 0.0, 0.0]
    action = (
        np.arange(40, dtype=dtype)[::2]
        if layout == "strided"
        else np.arange(20, dtype=dtype)
    )
    if layout == "reversed":
        action = action[::-1]
    if layout == "readonly":
        action.flags.writeable = False
    expected = action.astype(float)
    audit.update([1, 2, 3], [1, 0, 0, 0], action)
    assert (
        audit.previous_action.dtype == np.float64
        and audit.previous_action.flags.owndata
    )
    assert not np.shares_memory(audit.previous_action, action)
    action.flags.writeable = True
    action[:] = 99
    np.testing.assert_array_equal(audit.previous_action, expected)
    audit.update([1, 2, 3], [1, 0, 0, 0], np.zeros(20, dtype=dtype))
    assert audit.current["action_delta_square_sum"] == float(np.square(expected).sum())
    audit.finish()
    assert audit.trials[0]["steps"] == 2


@pytest.mark.parametrize("layout", ["strided", "reversed"])
def test_diagnostic_start_preserves_contiguous_quaternion_reduction(layout):
    from embodiedforge._wuji_diagnostics import quaternion_angle

    rng = np.random.default_rng(391)
    for _ in range(32):
        quaternions = [rng.normal(size=8)[::2] for _ in range(2)]
        if layout == "reversed":
            quaternions = [value[::-1] for value in quaternions]
        expected = quaternion_angle(*(value.copy() for value in quaternions))
        audit = TrialDiagnostics()
        audit.start([0, 0, 0], *quaternions)
        assert audit.current["initial_error_rad"] == expected
