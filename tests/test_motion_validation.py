"""Large reference inputs keep field validation and caller ownership intact."""

import numpy as np
import pytest

from embodiedforge._motion_clips import MotionClips

FIELDS = {
    "joint_pos": (17,),
    "joint_vel": (17,),
    "body_pos_w": (4, 3),
    "body_quat_w": (4, 4),
    "body_lin_vel_w": (4, 3),
    "body_ang_vel_w": (4, 3),
    "object_pos_w": (3,),
    "object_quat_w": (4,),
    "object_lin_vel_w": (3,),
    "object_ang_vel_w": (3,),
    "contact_label": (4,),
}


def motion_arrays(frames, dtype="float32", *, bodies=4, joints=17):
    metadata = {
        "schema": "ef-motion-v1",
        "robot": "validation-fixture",
        "joint_names": [f"joint-{i}" for i in range(joints)],
        "body_names": [f"body-{i}" for i in range(bodies)],
        "quaternion_order": "wxyz",
        "coordinate_frame": "world",
    }
    arrays = {
        "fps": np.array(50.0),
        "motion_lengths": np.array([frames]),
        "motion_names": np.array(["clip"]),
        "object_names": np.array(["box"]),
    }
    for field, tail in FIELDS.items():
        if field.startswith("joint_"):
            tail = (joints,)
        elif field.startswith("body_"):
            tail = (bodies, tail[-1])
        elif field == "contact_label":
            tail = (bodies,)
        seed = [1, 0, 0, 0] if "quat" in field else 0
        arrays[field] = np.broadcast_to(np.array(seed, dtype=dtype), (frames, *tail))
    return arrays, metadata


def arrange(value, order):
    if order == "fortran":
        return np.array(value, order="F", copy=True)
    if order == "reversed":
        return value[::-1].copy()[::-1]
    if order == "strided":
        return np.repeat(value, 2, axis=0)[::2]
    return value.copy()


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("dtype", ["float32", "float64", ">f4"])
@pytest.mark.parametrize("order", ["contiguous", "fortran", "reversed", "strided"])
def test_large_fields_reject_nonfinite_tail_without_mutating_inputs(
    field, dtype, order
):
    frames = 262144 // int(np.prod(FIELDS[field])) + 2
    arrays, metadata = motion_arrays(frames, dtype)
    value = arrange(arrays[field], order)
    value.flags.writeable = False
    arrays[field] = value
    before = value.tobytes()
    result = MotionClips(arrays, metadata)
    assert result.arrays is arrays and result.metadata is metadata
    assert result.starts.tolist() == [0, frames]
    assert value.tobytes() == before and not value.flags.writeable
    for invalid in (np.nan, np.inf, -np.inf):
        bad = arrange(value, order)
        bad.flat[-1] = invalid
        bad.flags.writeable = False
        arrays[field] = bad
        before = bad.tobytes()
        with pytest.raises(ValueError, match=f"{field} must be finite numeric"):
            MotionClips(arrays, metadata)
        assert bad.tobytes() == before and not bad.flags.writeable


@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "float64", ">f8", "longdouble"]
)
@pytest.mark.parametrize("order", ["contiguous", "fortran", "reversed", "strided"])
@pytest.mark.parametrize("field", ["body_quat_w", "object_quat_w"])
def test_large_unit_quaternion_tolerance_keeps_boundary_decisions(dtype, order, field):
    frames = 262144 // int(np.prod(FIELDS[field])) + 2
    arrays, metadata = motion_arrays(frames, dtype)
    scalar = np.dtype(dtype).type
    tolerance = np.longdouble(1e-3)
    for edge in (scalar(0.999), scalar(1.001)):
        for target in (scalar(0), scalar(1), scalar(2)):
            scale = np.nextafter(edge, target)
            value = arrange(arrays[field], order)
            value[-1] = 0
            value[-1, ..., 0] = scale
            value.flags.writeable = False
            candidate = {**arrays, field: value}
            expected = abs(np.longdouble(scale) - 1) <= tolerance
            if expected:
                MotionClips(candidate, metadata)
            else:
                with pytest.raises(ValueError, match=f"{field} must contain unit"):
                    MotionClips(candidate, metadata)
            assert not value.flags.writeable


@pytest.mark.parametrize("dtype", ["int8", "uint64", ">i8", "float32"])
def test_large_contact_domain_checks_the_final_element(dtype):
    arrays, metadata = motion_arrays(65538, dtype)
    value = arrays["contact_label"].copy()
    value[::2] = 1
    MotionClips({**arrays, "contact_label": value}, metadata)
    if np.dtype(dtype).kind != "u":
        value[::3] = -1
        MotionClips({**arrays, "contact_label": value}, metadata)
    value[-1, -1] = 2
    value.flags.writeable = False
    with pytest.raises(ValueError, match="contact_label must use"):
        MotionClips({**arrays, "contact_label": value}, metadata)


@pytest.mark.parametrize("frames", [1, 65538])
@pytest.mark.parametrize("dtype", ["uint64", ">u8"])
def test_unsigned_contacts_accept_zero_one_and_reject_large_integers(frames, dtype):
    arrays, metadata = motion_arrays(frames)
    labels = np.zeros((frames, 4), dtype=dtype)
    labels[::2] = 1
    MotionClips({**arrays, "contact_label": labels}, metadata)
    for invalid in (2, 2**63, 2**64 - 1):
        labels[-1, -1] = invalid
        with pytest.raises(ValueError, match="contact_label must use"):
            MotionClips({**arrays, "contact_label": labels}, metadata)


def test_large_masked_finite_reduction_preserves_hidden_payload_semantics():
    arrays, metadata = motion_arrays(16386)
    value = arrays["joint_pos"].copy()
    mask = np.zeros(value.shape, dtype=bool)
    value[-1] = np.nan
    mask[-1] = True
    value.flags.writeable = False
    mask.flags.writeable = False
    masked = np.ma.array(value, mask=mask, copy=False)
    masked.flags.writeable = False
    arrays["joint_pos"] = masked
    MotionClips(arrays, metadata)
    assert np.isnan(masked.data[-1]).all() and masked.mask[-1].all()
    assert not masked.data.flags.writeable and not masked.mask.flags.writeable


def test_large_subclass_all_reduction_is_not_replaced():
    class RejectAll(np.ndarray):
        def all(self, *args, **kwargs):
            return False

    arrays, metadata = motion_arrays(16386)
    arrays["joint_pos"] = arrays["joint_pos"].view(RejectAll)
    with pytest.raises(ValueError, match="joint_pos must be finite"):
        MotionClips(arrays, metadata)


@pytest.mark.parametrize("bodies,joints", [(65537, 1), (1, 262145)])
def test_single_frame_wider_than_validation_target_is_still_checked(bodies, joints):
    arrays, metadata = motion_arrays(1, bodies=bodies, joints=joints)
    library = MotionClips(arrays, metadata)
    assert library.starts.tolist() == [0, 1]
    field = "body_quat_w" if bodies > 1 else "joint_pos"
    invalid = arrays[field].copy()
    invalid.flat[-1] = 2 if "quat" in field else np.nan
    invalid.flags.writeable = False
    with pytest.raises(ValueError, match=f"{field} must"):
        MotionClips({**arrays, field: invalid}, metadata)


@pytest.mark.parametrize("field", ["body_quat_w", "object_quat_w"])
@pytest.mark.parametrize("dtype", ["float16", ">f2"])
@pytest.mark.parametrize("frames", [65535, 65536, 65537])
@pytest.mark.parametrize(
    "order", ["contiguous", "fortran", "fortran-strided", "fortran-reversed"]
)
@pytest.mark.parametrize("payload", ["ecb567a834b949b9", "d5a61339d02f1fba"])
def test_half_precision_norm_keeps_whole_field_tolerance_decisions(
    tmp_path, field, dtype, frames, order, payload
):
    arrays, metadata = motion_arrays(frames, dtype, bodies=1, joints=1)
    value = arrays[field].copy()
    value[-1] = np.frombuffer(bytes.fromhex(payload), dtype="<f2")
    if order == "fortran":
        value = np.array(value, order="F", copy=True)
    elif order == "fortran-strided":
        storage = np.empty((2 * frames, *value.shape[1:]), dtype=dtype, order="F")
        storage[::2] = value
        value = storage[::2]
    elif order == "fortran-reversed":
        value = np.array(value[::-1], order="F", copy=True)[::-1]
    # NumPy half reductions can round differently with a contiguous reduction
    # axis. The installed NumPy's whole-field norm defines the existing format.
    accepted = np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-3, rtol=0)
    value.flags.writeable = False
    arrays[field] = value
    before = value.tobytes()
    if accepted:
        library = MotionClips(arrays, metadata)
        if frames == 65537 and order == "fortran":
            path = tmp_path / "half-reference.npz"
            library.save(path)
            loaded = MotionClips.load(path)
            assert loaded.arrays[field].dtype == np.dtype(dtype)
            assert loaded.arrays[field].tobytes() == before
    else:
        with pytest.raises(ValueError, match=f"{field} must contain unit"):
            MotionClips(arrays, metadata)
    assert value.tobytes() == before and not value.flags.writeable
