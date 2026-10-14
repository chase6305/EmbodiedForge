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
