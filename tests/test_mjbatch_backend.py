"""Runs under unittest in the isolated mjbatch env and under the core pytest suite."""

import importlib.util
import unittest

import numpy as np

from embodiedforge import Config, VectorEnv
from embodiedforge.backends.mjbatch import MjbatchPhysics
from embodiedforge.backends.physics import MujocoPhysics
from embodiedforge.core import SceneSpec


@unittest.skipUnless(
    importlib.util.find_spec("mjbatch"), "optional mjbatch unavailable"
)
class MjbatchContractTests(unittest.TestCase):
    def test_matches_sequential_mujoco_with_masks_resets_and_variable_dt(self):
        threaded, reference = MjbatchPhysics(num_threads=2), MujocoPhysics()
        rng = np.random.default_rng(123)
        try:
            for physics in (threaded, reference):
                physics.build(SceneSpec(mass=1.7), Config(num_envs=5))
                physics.reset(np.arange(5), np.zeros((5, 2)))
            for step in range(30):
                active = rng.random(5) < 0.7
                control = rng.uniform(-1, 1, (5, 2))
                if step % 7 == 0:
                    ids = np.array([4, 1, 3])
                    positions = rng.normal(size=(3, 2))
                    for physics in (threaded, reference):
                        physics.reset(ids, positions)
                dt, substeps = (0.02, 4) if step % 2 else (0.03, 3)
                before = threaded.snapshot()
                for physics in (threaded, reference):
                    physics.apply_control(control)
                    physics.step(dt, substeps, active)
                actual, expected = threaded.snapshot(), reference.snapshot()
                for field in ("position", "velocity", "time", "version"):
                    np.testing.assert_allclose(
                        getattr(actual, field), getattr(expected, field), atol=1e-12
                    )
                    np.testing.assert_array_equal(
                        getattr(actual, field)[~active], getattr(before, field)[~active]
                    )
                owned = actual.position.copy()
                threaded.position[:] += 1
                np.testing.assert_array_equal(actual.position, owned)
                threaded.position[:] -= 1
        finally:
            threaded.close()
            reference.close()

    def test_terminal_rows_freeze_and_reset_does_not_change_other_rows(self):
        with VectorEnv(Config(physics="mjbatch", num_envs=4, max_steps=1)) as env:
            result = env.step(np.zeros((4, 2)))
            self.assertTrue(result.truncated.all())
            frozen = env.observe()
            result = env.step(np.ones((4, 2)))
            for key in frozen:
                np.testing.assert_array_equal(result.observation[key], frozen[key])
            env.reset([3, 1])
            current = env.observe()
            for key in frozen:
                np.testing.assert_array_equal(current[key][[0, 2]], frozen[key][[0, 2]])
            self.assertEqual(current["step_id"].tolist(), [1, 0, 1, 0])
            env.reset([])


if __name__ == "__main__":
    unittest.main()
