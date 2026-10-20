"""Pixel-exact geometry checks for separable orthographic disk rendering."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv


@pytest.mark.parametrize("size", [1, 7, 64, 127])
@pytest.mark.parametrize("radius", [0.06, 0.1, 0.35])
def test_raster_matches_dense_reference_including_overlap_and_edge(size, radius):
    from embodiedforge.backends.render import RasterRenderer
    from embodiedforge.core import SceneSpec

    count = 24
    config = Config(
        num_envs=count,
        image_size=size,
        channels=("rgb", "depth", "instance_id", "semantic_id"),
    )
    rng = np.random.default_rng(3)
    with VectorEnv(Config(num_envs=count)) as env:
        renderer = RasterRenderer()
        renderer.build(SceneSpec(radius=radius), config)
        update = env.task.scene_update(env.physics.snapshot())
        update.entities["agent"][:] = rng.uniform(-2, 2, (count, 2))
        update.entities["target"][:] = rng.uniform(-2, 2, (count, 2))
        update.entities["agent"][0] = update.entities["target"][0] = 0
        # Put disk edges exactly on selected pixel centers and image boundaries.
        coord = (np.arange(size) + 0.5) * 3 / size - 1.5
        update.entities["agent"][1] = (coord[0] + radius, -coord[0])
        update.entities["target"][2] = (1.5, -1.5)
        renderer.sync(update)
        ids = np.array([2, 0, 1, 23, 5])
        actual = renderer.render(ids)
        x, y = np.meshgrid(coord, -coord)
        expected = np.zeros((len(ids), 1, size, size), dtype=np.int32)
        for label, entity, disk_radius in ((2, "target", 0.1), (1, "agent", radius)):
            for row, position in enumerate(update.entities[entity][ids]):
                expected[row, 0][
                    (x - position[0]) ** 2 + (y - position[1]) ** 2 <= disk_radius**2
                ] = label
        np.testing.assert_array_equal(actual.images["instance_id"], expected)
        np.testing.assert_array_equal(actual.images["semantic_id"], expected)
        np.testing.assert_array_equal(actual.images["depth"], (expected != 0) * 2)
        np.testing.assert_array_equal(actual.images["depth_valid"], expected != 0)
        palette = np.array([[20, 24, 32], [70, 155, 255], [65, 210, 120]], np.uint8)
        np.testing.assert_array_equal(actual.images["rgb"], palette[expected])
        actual.images["instance_id"][:] = 99
        np.testing.assert_array_equal(
            renderer.render(ids).images["instance_id"], expected
        )
        assert renderer.render(np.array([], dtype=np.int64)).images["rgb"].shape == (
            0,
            1,
            size,
            size,
            3,
        )


@pytest.mark.parametrize("size,count", [(32, 128), (64, 16), (256, 1), (257, 7)])
@pytest.mark.parametrize("radius", [1e-12, 0.06, 0.1, 0.35, 1.5, 4.0])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.longdouble])
def test_bounded_disk_matches_full_frame_at_tangencies(size, count, radius, dtype):
    from embodiedforge.backends.render import RasterRenderer
    from embodiedforge.core import SceneSpec

    renderer = RasterRenderer()
    renderer.build(SceneSpec(radius=radius), Config(image_size=size))
    coord = (np.arange(size) + 0.5) * 3 / size - 1.5
    rng = np.random.default_rng(13)
    positions = rng.uniform(-2 - radius, 2 + radius, (count, 2)).astype(dtype)
    for axis in range(2):
        for sign in (-1, 1):
            # A circle tangent at a pixel center, and its two neighboring
            # representable centers, exercise both rectangle and disk bounds.
            center = dtype(coord[size // 3]) * (1 if axis == 0 else -1)
            center += sign * dtype(radius)
            positions[0, 1 - axis] = coord[size // 2] * (-1 if axis == 0 else 1)
            for delta in (-np.inf, center, np.inf):
                positions[0, axis] = np.nextafter(center, dtype(delta))
                expected = (
                    (coord[None, None, :] - positions[:, 0, None, None]) ** 2
                    + (-coord[None, :, None] - positions[:, 1, None, None]) ** 2
                ) <= radius**2
                actual = np.full((count, size, size), 2, dtype=np.int32)
                renderer._draw_disk(actual, positions, radius, 1)
                np.testing.assert_array_equal(actual, np.where(expected, 1, 2))


@pytest.mark.parametrize("size,count", [(64, 1), (540, 1), (256, 128)])
@pytest.mark.parametrize("radius", [0.06, 0.35, 4.0])
def test_rgb_only_cast_keeps_owned_exact_pixels(size, count, radius):
    from embodiedforge.backends.render import RasterRenderer
    from embodiedforge.core import SceneSpec, SceneUpdate, StateSnapshot

    rng = np.random.default_rng(53)
    positions = rng.uniform(-2, 2, (count, 2))
    targets = rng.uniform(-2, 2, (count, 2))
    state = StateSnapshot(
        positions, np.zeros_like(positions), np.zeros(count), np.zeros(count, np.int64)
    )
    update = SceneUpdate(state, {"agent": positions, "target": targets})
    renderer = RasterRenderer()
    renderer.build(
        SceneSpec(radius=radius),
        Config(num_envs=count, image_size=size, channels=("rgb",)),
    )
    renderer.sync(update)
    ids = np.arange(count - 1, -1, -1)
    actual = renderer.render(ids).images["rgb"]
    coord = (np.arange(size) + 0.5) * 3 / size - 1.5
    labels = np.zeros((count, 1, size, size), np.int32)
    for entity, centers, disk_radius in ((2, targets, 0.1), (1, positions, radius)):
        mask = (
            (coord[None, None, :] - centers[ids, 0, None, None]) ** 2
            + (-coord[None, :, None] - centers[ids, 1, None, None]) ** 2
        ) <= disk_radius**2
        labels[:, 0][mask] = entity
    palette = np.array([[20, 24, 32], [70, 155, 255], [65, 210, 120]], np.uint8)
    expected = palette[labels]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8 and actual.flags.owndata
    actual[:] = 0
    np.testing.assert_array_equal(renderer.render(ids).images["rgb"], expected)


@pytest.mark.parametrize("size", [1, 64, 257])
@pytest.mark.parametrize(
    "channels",
    [
        ("depth",),
        ("rgb", "depth"),
        ("depth", "rgb"),
        ("rgb", "depth", "instance_id", "semantic_id"),
    ],
)
@pytest.mark.parametrize("empty", [False, True])
def test_depth_and_validity_own_independent_exact_frame_arrays(size, channels, empty):
    from embodiedforge.backends.render import RasterRenderer
    from embodiedforge.core import SceneSpec, SceneUpdate, StateSnapshot

    positions = np.array([[0.0, 0.0], [5.0, 5.0], [-1.5, 1.5]])
    targets = np.array([[0.0, 0.0], [-5.0, -5.0], [1.5, -1.5]])
    state = StateSnapshot(
        positions, np.zeros_like(positions), np.zeros(3), np.zeros(3, np.int64)
    )
    renderer = RasterRenderer()
    renderer.build(
        SceneSpec(radius=0.35),
        Config(num_envs=3, image_size=size, channels=channels),
    )
    renderer.sync(SceneUpdate(state, {"agent": positions, "target": targets}))
    ids = np.array([] if empty else [2, 0, 2, 1], dtype=np.int64)
    coord = (np.arange(size) + 0.5) * 3 / size - 1.5
    x, y = np.meshgrid(coord, -coord)
    expected_valid = np.zeros((len(ids), 1, size, size), bool)
    for row, index in enumerate(ids):
        for center, radius in ((positions[index], 0.35), (targets[index], 0.1)):
            expected_valid[row, 0] |= (x - center[0]) ** 2 + (
                y - center[1]
            ) ** 2 <= radius**2
    expected_depth = np.where(expected_valid, 2.0, 0.0).astype(np.float32)
    images = renderer.render(ids).images
    assert images.keys() == set(channels) | {"depth_valid"}
    depth, valid = images["depth"], images["depth_valid"]
    assert depth.dtype == np.float32 and valid.dtype == bool
    assert depth.flags.owndata and valid.flags.owndata
    assert depth.flags.c_contiguous and valid.flags.c_contiguous
    for name, value in images.items():
        for other, array in images.items():
            if name != other:
                assert not np.shares_memory(value, array)
    np.testing.assert_array_equal(depth, expected_depth)
    np.testing.assert_array_equal(valid, expected_valid)
    depth[:] = -1
    np.testing.assert_array_equal(valid, expected_valid)
    valid[:] = ~valid
    again = renderer.render(ids).images
    np.testing.assert_array_equal(again["depth"], expected_depth)
    np.testing.assert_array_equal(again["depth_valid"], expected_valid)
    renderer.close()
