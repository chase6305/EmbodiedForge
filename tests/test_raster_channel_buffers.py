"""Exact pixels and owned outputs across selective raster channel buffers."""

import itertools

import numpy as np
import pytest

from embodiedforge.backends.render import RasterRenderer
from embodiedforge.core import Config, SceneSpec, SceneUpdate, StateSnapshot


def render_case(size, dtype, layout, channels, empty):
    count = 4
    coord = (np.arange(size) + 0.5) * 3 / size - 1.5
    position = np.array([[0, 0], [5, 5], [-1.5, 1.5], [0.1, -0.1]], dtype=dtype)
    position[3, 0] = np.nextafter(dtype(coord[size // 2]) + dtype(0.35), dtype(np.inf))
    target = np.array([[0, 0], [-5, -5], [1.5, -1.5], [0.1, -0.1]], dtype=dtype)
    if layout == "fortran":
        position, target = np.asfortranarray(position), np.asfortranarray(target)
    elif layout == "strided":
        position = np.repeat(position, 2, axis=0)[::2]
        target = np.repeat(target, 2, axis=0)[::2]
    position.flags.writeable = target.flags.writeable = False
    before = position.tobytes(), target.tobytes()
    state = StateSnapshot(
        position,
        np.zeros_like(position),
        np.arange(count, dtype=float),
        np.arange(count),
    )
    renderer = RasterRenderer()
    renderer.build(
        SceneSpec(radius=0.35),
        Config(num_envs=count, image_size=size, channels=channels),
    )
    renderer.sync(SceneUpdate(state, dict(agent=position, target=target)))
    ids = np.array([] if empty else [3, 0, 3, 2, 1], dtype=np.int64)
    ids.flags.writeable = False
    x, y = np.meshgrid(coord, -coord)
    labels = np.zeros((len(ids), size, size), np.int32)
    for entity, centers, radius in ((2, target, 0.1), (1, position, 0.35)):
        selected = centers[ids]
        mask = (x[None] - selected[:, 0, None, None]) ** 2 + (
            y[None] - selected[:, 1, None, None]
        ) ** 2 <= radius**2
        labels[mask] = entity
    labels = labels[:, None]
    palette = np.array([[20, 24, 32], [70, 155, 255], [65, 210, 120]], np.uint8)
    reference = dict(
        rgb=palette[labels],
        depth=np.where(labels != 0, np.float32(2), np.float32(0)),
        depth_valid=labels != 0,
        instance_id=labels,
        semantic_id=labels,
    )
    result = renderer.render(ids)
    assert list(result.images) == list(channels) + (
        ["depth_valid"] if "depth" in channels else []
    )
    for key, value in result.images.items():
        assert value.dtype == reference[key].dtype
        assert value.flags.owndata and value.flags.c_contiguous
        np.testing.assert_array_equal(value, reference[key])
        for other_key, other_value in result.images.items():
            if key != other_key:
                assert not np.shares_memory(value, other_value)
    np.testing.assert_array_equal(result.time, state.time[ids])
    np.testing.assert_array_equal(result.version, state.version[ids])
    for value in result.images.values():
        value[:] = 0
    again = renderer.render(ids)
    for key, value in again.images.items():
        np.testing.assert_array_equal(value, reference[key])
    assert before == (position.tobytes(), target.tobytes())
    renderer.close()


@pytest.mark.parametrize("size", [1, 64, 257])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize("channels", [("depth",), ("rgb",)])
@pytest.mark.parametrize("empty", [False, True])
def test_selective_channels_match_dense_geometry(size, dtype, layout, channels, empty):
    render_case(size, dtype, layout, channels, empty)


CHANNELS = ("rgb", "depth", "instance_id", "semantic_id")
COMBINATIONS = [
    combination
    for count in range(1, 5)
    for combination in itertools.combinations(CHANNELS, count)
]


@pytest.mark.parametrize("channels", COMBINATIONS)
@pytest.mark.parametrize("reverse", [False, True])
def test_requested_channel_order_and_layout_stay_compatible(channels, reverse):
    render_case(
        129, np.float64, "strided", channels[::-1] if reverse else channels, False
    )
