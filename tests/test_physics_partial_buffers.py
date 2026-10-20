"""Partial Euler updates preserve shared buffers and nonstandard array layouts."""

import numpy as np
import pytest

from embodiedforge.backends.physics import NumpyPhysics
from embodiedforge.core import Config, SceneSpec


def backend(layout, alias):
    physics = NumpyPhysics()
    physics.build(SceneSpec(mass=1.3), Config(num_envs=17))
    rng = np.random.default_rng(731)
    for name in ("position", "velocity", "control"):
        value = rng.normal(size=(17, 2))
        if layout == "fortran":
            value = np.asfortranarray(value)
        elif layout == "strided":
            value = np.repeat(value, 2, axis=0)[::2]
        setattr(physics, name, value)
    if alias is not None:
        left, right = alias.split("/")
        setattr(physics, left, getattr(physics, right))
    return physics


@pytest.mark.parametrize("layout", ["contiguous", "fortran", "strided"])
@pytest.mark.parametrize(
    "alias", [None, "velocity/position", "control/position", "control/velocity"]
)
def test_partial_euler_preserves_buffer_relationships(layout, alias):
    actual, reference = backend(layout, alias), backend(layout, alias)
    rng = np.random.default_rng(732)
    for step in range(24):
        active = rng.random(17) < 0.5
        # Strided and read-only masks are borrowed, never mutated.
        active = np.repeat(active, 2)[::2]
        active.flags.writeable = False
        mask_before = active.tobytes()
        dt, substeps = 0.01 + 0.005 * (step % 3), 1 + step % 9
        h = dt / substeps
        before = actual.snapshot()
        for _ in range(substeps):
            reference.velocity[active] += (
                reference.control[active] * h / reference.scene.mass
            )
            reference.position[active] += reference.velocity[active] * h
        reference.time[active] += dt
        reference.version[active] += 1
        actual.step(dt, substeps, active)
        for name in ("position", "velocity", "time", "version"):
            np.testing.assert_array_equal(
                getattr(actual, name), getattr(reference, name)
            )
            np.testing.assert_array_equal(
                getattr(actual, name)[~active], getattr(before, name)[~active]
            )
        assert active.tobytes() == mask_before


def test_partial_euler_preserves_completed_substeps_on_floating_error():
    physics = NumpyPhysics()
    physics.build(SceneSpec(), Config(num_envs=3))
    limit = np.finfo(np.float64).max
    physics.position[:] = -limit * 0.4
    physics.velocity[:] = limit * 0.3
    physics.control[:] = limit * 0.4
    before = physics.snapshot()
    active = np.array([True, False, True])
    expected_velocity = before.velocity[active] + physics.control[active]
    expected_position = before.position[active] + expected_velocity
    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            physics.step(4, 4, active)
    np.testing.assert_array_equal(physics.velocity[active], expected_velocity)
    np.testing.assert_array_equal(physics.position[active], expected_position)
    for name, value in vars(before).items():
        np.testing.assert_array_equal(getattr(physics, name)[~active], value[~active])
    np.testing.assert_array_equal(physics.time, before.time)
    np.testing.assert_array_equal(physics.version, before.version)
