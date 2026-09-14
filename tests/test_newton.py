"""Compatibility tests against the user's exact Newton source archive release."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.backends import execution_plan
from embodiedforge.backends.newton import NEWTON_VERSION, NewtonPhysics


@pytest.fixture
def sdk(monkeypatch, tmp_path):
    newton = pytest.importorskip("newton")
    # Kernel compilation writes only into the test's temporary directory.
    monkeypatch.setenv("WARP_CACHE_PATH", str(tmp_path / "warp-cache"))
    assert newton.__version__ == NEWTON_VERSION
    return newton


def test_newton_plan_without_sdk_initialization():
    plan = execution_plan(Config(physics="newton", task="hold"))
    assert plan["physics"] == "newton"
    assert plan["physics_version"] == "1.6.0rc1"
    assert plan["physics_source"].endswith("/refs/tags/v1.6.0rc1.zip")
    assert plan["proprio_dim"] == 4


@pytest.mark.parametrize("task", ["reach", "hold"])
@pytest.mark.parametrize("render", ["null", "raster"])
def test_tagged_newton_matches_reference_dynamics(sdk, task, render):
    options = dict(
        task=task,
        render=render,
        num_envs=3,
        channels=("rgb", "depth") if render == "raster" else (),
    )
    with (
        VectorEnv(Config(physics="numpy", **options)) as reference,
        VectorEnv(Config(physics="newton", **options)) as actual,
    ):
        assert actual.physics.model.world_count == 3
        for _ in range(20):
            action = np.array([[0.3, -0.2], [0.2, 0.1], [-0.3, 0.4]])
            expected = reference.step(action)
            result = actual.step(action)
            np.testing.assert_allclose(
                result.observation["proprio"],
                expected.observation["proprio"],
                atol=2e-6,
            )
            np.testing.assert_allclose(result.reward, expected.reward, atol=2e-6)
            if render == "raster":
                np.testing.assert_array_equal(
                    result.observation["rgb"], expected.observation["rgb"]
                )


def test_newton_partial_reset_and_nonzero_velocity_freeze(sdk):
    with VectorEnv(Config(physics="newton", num_envs=2, max_steps=3)) as env:
        env.step(np.full((2, 2), 0.5))
        env.reset([0])
        for _ in range(2):
            result = env.step(np.full((2, 2), 0.5))
        assert result.truncated.tolist() == [False, True]
        frozen = env.physics.snapshot()
        assert (frozen.velocity[1] != 0).any()
        env.step(np.full((2, 2), -0.5))
        current = env.physics.snapshot()
        np.testing.assert_array_equal(frozen.position[1], current.position[1])
        np.testing.assert_array_equal(frozen.velocity[1], current.velocity[1])
        assert frozen.time[1] == current.time[1]
        env.reset([0])
        np.testing.assert_array_equal(
            current.velocity[1], env.physics.snapshot().velocity[1]
        )


def test_newton_rejects_other_versions(sdk, monkeypatch):
    monkeypatch.setattr(sdk, "__version__", "1.5.0")
    with pytest.raises(RuntimeError, match="requires 1.6.0rc1"):
        VectorEnv(Config(physics="newton"))
    NewtonPhysics().close()  # cleanup must be safe even before build.
