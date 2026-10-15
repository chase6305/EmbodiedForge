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
