"""Long motion scoring preserves analytic errors and never writes its inputs."""

import math

import numpy as np
import pytest

from embodiedforge._motion_clips import MotionClips
from embodiedforge.motion import compare_clips


def library(lengths, *, dtype, bodies=4, joints=17, rollout=False):
    frames = sum(lengths)
    metadata = {
        "schema": "ef-motion-v1",
        "robot": "scoring-fixture",
        "joint_names": [f"joint-{i}" for i in range(joints)],
        "body_names": [f"body-{i}" for i in range(bodies)],
        "quaternion_order": "wxyz",
        "coordinate_frame": "world",
    }
    arrays = {
        "fps": np.array(50.0),
        "motion_lengths": np.array(lengths),
        "motion_names": np.array([f"clip-{i}" for i in range(len(lengths))]),
        "object_names": np.array(["box"] * len(lengths)),
    }
    for name, shape in {
        "joint_pos": (frames, joints),
        "joint_vel": (frames, joints),
        "body_pos_w": (frames, bodies, 3),
        "body_quat_w": (frames, bodies, 4),
        "body_lin_vel_w": (frames, bodies, 3),
        "body_ang_vel_w": (frames, bodies, 3),
        "object_pos_w": (frames, 3),
        "object_quat_w": (frames, 4),
        "object_lin_vel_w": (frames, 3),
        "object_ang_vel_w": (frames, 3),
        "contact_label": (frames, bodies),
    }.items():
        arrays[name] = np.zeros(shape, dtype=dtype)
        if "quat" in name:
            arrays[name][..., 0] = 1
    if rollout:
        metadata["reference_sha256"] = "reference-digest"
        for name in ("terminated", "truncated", "clip_end"):
            arrays[name] = np.zeros(frames, dtype=bool)
        arrays["clip_end"][np.cumsum(lengths) - 1] = True
    result = MotionClips(arrays, metadata)
    result.sha256 = "rollout-digest" if rollout else "reference-digest"
    return result


def layout(value, order):
    if order == "fortran":
        result = np.asfortranarray(value)
    elif order == "reversed":
        result = value[::-1].copy()[::-1]
    elif order == "strided":
        result = np.repeat(value, 2, axis=0)[::2]
    else:
        result = value.copy()
    result.flags.writeable = False
    return result


@pytest.mark.parametrize("dtype", ["float32", "float64", ">f4"])
@pytest.mark.parametrize("order", ["contiguous", "fortran", "reversed", "strided"])
@pytest.mark.parametrize("frames", [4095, 4096, 4097])
def test_long_scores_match_independent_position_and_angle_oracles(dtype, order, frames):
    reference = library([frames], dtype=dtype)
    rollout = library([frames], dtype=dtype, rollout=True)
    rng = np.random.default_rng(frames)
    expected = {}
    for field, label in (
        ("joint_pos", "joint_rmse_rad"),
        ("body_pos_w", "body_position_rmse_m"),
        ("object_pos_w", "object_position_rmse_m"),
    ):
        values = rollout.arrays[field]
        values[:] = rng.normal(scale=0.1, size=values.shape)
        # Python's compensated sum is independent of NumPy's block boundaries.
        mean_square = math.fsum(float(x) ** 2 for x in values.flat)
        denominator = values.size if field == "joint_pos" else values.size // 3
        expected[label] = math.sqrt(mean_square / denominator)
    for field, label in (
        ("body_quat_w", "body_rotation_rmse_rad"),
        ("object_quat_w", "object_rotation_rmse_rad"),
    ):
        values = rollout.arrays[field]
        angles = rng.uniform(0, np.pi, values.shape[:-1])
        values[..., 0] = np.cos(angles / 2)
        values[..., 3] = np.sin(angles / 2)
        values[::3] *= -1
        # Recover the represented angle directly from its axis-angle coordinates.
        represented = [
            2 * math.atan2(abs(float(q[3])), abs(float(q[0])))
            for q in values.reshape(-1, 4)
        ]
        expected[label] = math.sqrt(
            math.fsum(x * x for x in represented) / len(represented)
        )
    before = {}
    for side, motion in enumerate((reference, rollout)):
        for key in motion.frame_fields:
            motion.arrays[key] = layout(motion.arrays[key], order)
            before[side, key] = motion.arrays[key].tobytes()
    report = compare_clips(reference, rollout)
    assert report["status"] == "complete" and report["success_rate"] == 1
    for key, value in expected.items():
        assert report["clips"][0][key] == pytest.approx(value, rel=2e-14, abs=1e-16)
        assert report["metrics_all"][key] == report["metrics_success"][key]
    for side, motion in enumerate((reference, rollout)):
        for key in motion.frame_fields:
            assert motion.arrays[key].tobytes() == before[side, key]
            assert not motion.arrays[key].flags.writeable


@pytest.mark.parametrize("angle", [0.0, 1e-12, 1e-9, 1e-5, np.pi - 1e-9, np.pi])
@pytest.mark.parametrize("sign", [-1, 1])
def test_long_rotation_precision_and_equivalent_quaternion_signs(angle, sign):
    reference = library([16385], dtype="float64", bodies=1, joints=1)
    rollout = library([16385], dtype="float64", bodies=1, joints=1, rollout=True)
    for field in ("body_quat_w", "object_quat_w"):
        reference.arrays[field] *= 1.0004
        values = rollout.arrays[field]
        values[..., 0] = sign * np.cos(angle / 2) * 0.9996
        values[..., 3] = sign * np.sin(angle / 2) * 0.9996
    report = compare_clips(reference, rollout)
    for key in ("body_rotation_rmse_rad", "object_rotation_rmse_rad"):
        assert report["metrics_all"][key] == pytest.approx(angle, rel=2e-14, abs=1e-25)


def test_long_and_short_clips_keep_equal_weights_and_name_alignment():
    lengths = [16385, 1]
    reference = library(lengths, dtype="float64", bodies=1, joints=1)
    rollout = library(lengths[::-1], dtype="float64", bodies=1, joints=1, rollout=True)
    rollout.names.reverse()
    rollout.arrays["joint_pos"][0] = 2
    rollout.arrays["joint_pos"][1:] = 4
    rollout.arrays["terminated"][-1] = True
    report = compare_clips(reference, rollout)
    assert [row["status"] for row in report["clips"]] == ["failed", "success"]
    assert report["metrics_all"]["joint_rmse_rad"] == 3
    assert report["metrics_success"]["joint_rmse_rad"] == 2


def test_large_finite_position_error_that_overflows_still_rejects_metrics():
    reference = library([4097], dtype="float64")
    rollout = library([4097], dtype="float64", rollout=True)
    rollout.arrays["joint_pos"][-1] = 1e200
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="Nonfinite tracking"),
    ):
        compare_clips(reference, rollout)


@pytest.mark.parametrize("bodies,joints", [(1, 65537), (16385, 1)])
def test_frame_wider_than_block_target_is_scored_completely(bodies, joints):
    reference = library([3], dtype="float64", bodies=bodies, joints=joints)
    rollout = library([3], dtype="float64", bodies=bodies, joints=joints, rollout=True)
    rollout.arrays["joint_pos"][:] = 2
    rollout.arrays["body_pos_w"][:] = [1, 2, 2]
    rollout.arrays["object_pos_w"][:] = [2, 3, 6]
    rollout.arrays["body_quat_w"][:] = [0, 0, 0, -1]
    report = compare_clips(reference, rollout)
    assert report["metrics_all"] == pytest.approx(
        {
            "joint_rmse_rad": 2,
            "body_position_rmse_m": 3,
            "object_position_rmse_m": 7,
            "body_rotation_rmse_rad": np.pi,
            "object_rotation_rmse_rad": 0,
        },
        rel=2e-14,
        abs=1e-25,
    )


@pytest.mark.parametrize("field", ["joint_pos", "body_pos_w", "object_pos_w"])
@pytest.mark.parametrize("side", ["reference", "rollout", "both"])
@pytest.mark.parametrize("mask_kind", ["partial", "tail"])
@pytest.mark.parametrize("order", ["contiguous", "fortran"])
def test_masked_subclasses_keep_whole_field_means(field, side, mask_kind, order):
    reference = library([21846], dtype="float64")
    rollout = library([21846], dtype="float64", rollout=True)
    rollout.arrays[field][:] = 2
    snapshots = []
    for name, motion in (("reference", reference), ("rollout", rollout)):
        if side != "both" and side != name:
            continue
        mask = np.zeros(motion.arrays[field].shape, dtype=bool)
        if mask_kind == "partial":
            mask[::3] = True
        else:
            # An entirely masked final block must not poison the whole mean.
            mask[-4097:] = True
        value = np.ma.array(
            layout(motion.arrays[field], order), mask=layout(mask, order), copy=False
        )
        motion.arrays[field] = value
        MotionClips(motion.arrays, motion.metadata)  # Accepted public constructor.
        snapshots.append((value, value.data.tobytes(), value.mask.tobytes()))
    report = compare_clips(reference, rollout)
    key = {
        "joint_pos": "joint_rmse_rad",
        "body_pos_w": "body_position_rmse_m",
        "object_pos_w": "object_position_rmse_m",
    }[field]
    expected = 2 if field == "joint_pos" else math.sqrt(12)
    assert report["metrics_all"][key] == pytest.approx(expected, rel=2e-14)
    for value, data, mask in snapshots:
        assert value.data.tobytes() == data and value.mask.tobytes() == mask
        assert not value.flags.writeable and not value.mask.flags.writeable
