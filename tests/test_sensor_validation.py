"""Large depth checks preserve finite/mask rejection and cache atomicity."""

from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge.core import Config, RenderBatch, SceneUpdate, StateSnapshot
from embodiedforge.sensors import SensorPipeline, _matching_depth_validity, _valid_depth


@pytest.fixture
def large_depth():
    depth = np.ones((1, 1, 2049, 2049), dtype=np.float32)
    depth[..., ::3, ::5] = 0
    return depth


@pytest.mark.parametrize("layout", ["C", "F", "reversed", "mixed"])
def test_large_strided_depth_and_validity_match_numpy(large_depth, layout):
    depth = large_depth
    if layout == "F":
        depth = np.asfortranarray(depth)
    elif layout == "reversed":
        depth = depth[..., ::-1, ::-1]
    valid = depth > 0
    if layout == "mixed":
        valid = np.asfortranarray(valid)
    assert _valid_depth(depth)
    assert _matching_depth_validity(depth, valid)
    valid[-1, -1, -1, -1] = not valid[-1, -1, -1, -1]
    assert not _matching_depth_validity(depth, valid)


@pytest.mark.parametrize("index", [0, 262143, 262144, -1])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -1.0])
def test_invalid_depth_at_chunk_edges_is_rejected(large_depth, index, invalid):
    large_depth.reshape(-1)[index] = invalid
    assert not _valid_depth(large_depth)


def test_subclass_validation_retains_numpy_dispatch(large_depth, monkeypatch):
    class Images(np.ndarray):
        pass

    def unexpected(*args, **kwargs):
        pytest.fail("Subclasses must retain their NumPy validation dispatch")

    monkeypatch.setattr(np, "nditer", unexpected)
    depth = large_depth.view(Images)
    assert _valid_depth(depth)
    assert _matching_depth_validity(depth, depth > 0)


@pytest.mark.parametrize("fault", ["nonfinite", "negative", "validity", "both"])
def test_large_invalid_batch_does_not_change_any_sensor_cache(large_depth, fault):
    config = Config(num_envs=1, image_size=2049, channels=("depth",))
    batch = RenderBatch(
        {"depth": large_depth, "depth_valid": large_depth > 0},
        np.zeros(1),
        np.zeros(1, dtype=np.int64),
        {"camera": {"width": 2049}},
    )
    pipeline = SensorPipeline(
        config, SimpleNamespace(sync=lambda _: None, render=lambda _: batch)
    )
    state = StateSnapshot(np.zeros((1, 2)), np.zeros((1, 2)), batch.time, batch.version)
    update = SceneUpdate(state, {})
    pipeline.sample(np.array([0]), update)
    previous = pipeline.observe(np.zeros(1))
    calibration = pipeline.calibration.copy()
    if fault in ("nonfinite", "both"):
        large_depth.reshape(-1)[-1] = np.nan
    if fault == "negative":
        large_depth.reshape(-1)[-1] = -1
    if fault in ("validity", "both"):
        batch.images["depth_valid"].reshape(-1)[0] = True
    message = "validity" if fault == "validity" else "finite and nonnegative"
    with pytest.raises(ValueError, match=message):
        pipeline.sample(np.array([0]), update)
    for name, values in previous.items():
        np.testing.assert_array_equal(pipeline.observe(np.zeros(1))[name], values)
    assert pipeline.calibration == calibration


@pytest.mark.parametrize("size", [0, 8191, 8192, 8193, 4194304, 4194305])
@pytest.mark.parametrize("layout", ["C", "F", "reversed", "strided"])
def test_depth_extremes_and_readonly_inputs_across_validation_boundaries(size, layout):
    values = np.array(
        [0.0, -0.0, np.finfo(np.float32).smallest_subnormal, np.finfo(np.float32).max],
        dtype=np.float32,
    )
    depth = np.resize(values, size).reshape(1, -1)
    if layout == "F":
        depth = np.asfortranarray(depth.T)
    elif layout == "reversed":
        depth = depth[:, ::-1]
    elif layout == "strided":
        depth = np.repeat(depth, 2, axis=1)[:, ::2]
    valid = depth > 0
    depth.setflags(write=False)
    valid.setflags(write=False)
    original_depth, original_valid = depth.copy(), valid.copy()
    with np.errstate(all="raise"):
        assert _valid_depth(depth)
        assert _matching_depth_validity(depth, valid)
    np.testing.assert_array_equal(depth, original_depth)
    np.testing.assert_array_equal(valid, original_valid)


@pytest.mark.parametrize("size", [8192, 4194305])
@pytest.mark.parametrize(
    "bits",
    [0x80000001, 0x7F800000, 0xFF800000, 0x7F800001, 0x7FC00000, 0xFF800001],
)
def test_float32_negative_subnormal_infinities_and_nan_payloads_are_rejected(
    size, bits
):
    depth = np.ones(size, dtype=np.float32)
    # Set the bits directly so signaling NaNs reach the validator unchanged.
    depth.view(np.uint32)[-1] = bits
    depth.setflags(write=False)
    with np.errstate(all="raise"):
        assert not _valid_depth(depth)


@pytest.mark.parametrize("layout", ["C", "F", "reversed", "mixed", "strided"])
@pytest.mark.parametrize("size", [64, 513, 2049])
def test_mask_mismatch_preserves_readonly_depth_and_mask_at_last_pixel(size, layout):
    depth = np.ones((1, 1, size, size), dtype=np.float32)
    depth[..., ::3, ::5] = 0
    if layout == "F":
        depth = np.asfortranarray(depth)
    elif layout == "reversed":
        depth = depth[..., ::-1, ::-1]
    elif layout == "strided":
        depth = np.repeat(depth, 2, axis=-1)[..., ::2]
    valid = depth > 0
    if layout == "mixed":
        valid = np.asfortranarray(valid)
    valid[-1, -1, -1, -1] = not valid[-1, -1, -1, -1]
    depth.setflags(write=False)
    valid.setflags(write=False)
    previous_depth, previous_valid = depth.copy(), valid.copy()
    assert not _matching_depth_validity(depth, valid)
    np.testing.assert_array_equal(depth, previous_depth)
    np.testing.assert_array_equal(valid, previous_valid)


@pytest.mark.parametrize("value", [0.0, 1.0, -1.0, np.inf, np.nan])
def test_scalar_depth_helpers_keep_numpy_predicate_semantics(value):
    depth = np.array(value, dtype=np.float32)
    valid = np.array(value > 0)
    assert _valid_depth(depth) == bool(np.isfinite(depth) and depth >= 0)
    assert _matching_depth_validity(depth, valid)
    assert not _matching_depth_validity(depth, ~valid)


@pytest.mark.parametrize("dtype", [np.float64, ">f4"])
def test_non_camera_depth_dtype_and_numeric_validity_keep_numpy_semantics(dtype):
    depth = np.array([0.0, -0.0, 1.0, 2.0], dtype=dtype)
    valid = np.array([0.0, 0.0, 1.0, 1.0])
    assert _valid_depth(depth)
    assert _matching_depth_validity(depth, valid)
    assert not _matching_depth_validity(depth, valid.reshape(2, 2))
    valid[-1] = np.nan
    assert not _matching_depth_validity(depth, valid)


@pytest.mark.parametrize("fault", ["nonfinite", "negative", "validity", "both"])
def test_small_invalid_batch_preserves_existing_images_timestamps_and_calibration(
    fault,
):
    config = Config(num_envs=3, image_size=64, channels=("depth",))
    depth = np.ones((2, 1, 64, 64), dtype=np.float32)
    state = StateSnapshot(
        np.zeros((3, 2)), np.zeros((3, 2)), np.arange(3.0), np.arange(3)
    )
    ids = np.array([2, 0])
    batch = RenderBatch(
        {"depth": depth, "depth_valid": depth > 0},
        state.time[ids],
        state.version[ids],
        {"camera": {"width": 64}},
    )
    pipeline = SensorPipeline(
        config, SimpleNamespace(sync=lambda _: None, render=lambda _: batch)
    )
    update = SceneUpdate(state, {})
    pipeline.sample(ids, update)
    previous = pipeline.observe(state.time)
    if fault in ("nonfinite", "both"):
        depth[-1, -1, -1, -1] = np.nan
    if fault == "negative":
        depth[-1, -1, -1, -1] = -1
    if fault in ("validity", "both"):
        batch.images["depth_valid"][-1, -1, -1, -1] = False
    batch.calibration["camera"]["width"] = 1
    message = "validity" if fault == "validity" else "finite and nonnegative"
    with pytest.raises(ValueError, match=message):
        pipeline.sample(ids, update)
    for name, values in previous.items():
        np.testing.assert_array_equal(pipeline.observe(state.time)[name], values)
    assert pipeline.calibration == {"camera": {"width": 64}}
