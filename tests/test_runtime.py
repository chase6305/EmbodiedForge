import subprocess
import sys

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.backends import execution_plan
from embodiedforge.backends.physics import NumpyPhysics
from embodiedforge.core import SceneSpec


def test_core_import_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import embodiedforge, sys; "
            "assert not {'torch', 'mujoco', 'warp', 'newton'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_capabilities_fail_before_construction():
    with pytest.raises(ValueError, match="missing render channels"):
        execution_plan(Config(channels=("rgb",)))
    with pytest.raises(ValueError, match="Unknown physics"):
        execution_plan(Config(physics="missing"))
    with pytest.raises(ValueError, match="divisible"):
        Config(physics_hz=201)


def test_reference_force_and_owned_state():
    physics = NumpyPhysics()
    physics.build(SceneSpec(), Config(num_envs=2))
    physics.reset(np.array([0, 1]), np.zeros((2, 2)))
    before = physics.snapshot()
    physics.apply_control(np.ones((2, 2)))
    physics.step(0.1, 2, np.array([True, False]))
    after = physics.snapshot()
    np.testing.assert_allclose(after.position[0], [0.0075, 0.0075])
    np.testing.assert_allclose(after.velocity[0], [0.1, 0.1])
    np.testing.assert_array_equal(after.position[1], [0, 0])
    np.testing.assert_array_equal(before.position, np.zeros((2, 2)))


@pytest.mark.parametrize("render", ["null", "raster"])
def test_partial_reset_isolated_and_deterministic(render):
    config = Config(
        num_envs=3, render=render, channels=("rgb",) if render == "raster" else ()
    )
    with VectorEnv(config) as a, VectorEnv(config) as b:
        a.step(np.ones((3, 2)) * 0.2)
        b.step(np.ones((3, 2)) * 0.2)
        before = a.observe()
        a.reset([1])
        after = a.observe()
        for key in before:
            np.testing.assert_array_equal(before[key][[0, 2]], after[key][[0, 2]])
        b.reset([2])
        b.reset([1])
        np.testing.assert_array_equal(
            a.observe()["proprio"][1], b.observe()["proprio"][1]
        )


@pytest.mark.parametrize("physics", ["numpy", "mujoco"])
def test_done_freezes_and_final_frame_is_fresh(physics):
    if physics == "mujoco":
        pytest.importorskip("mujoco")
    config = Config(
        physics=physics, num_envs=2, max_steps=1, render="raster", channels=("rgb",)
    )
    with VectorEnv(config) as env:
        result = env.step(np.zeros((2, 2)))
        assert result.truncated.all()
        np.testing.assert_allclose(result.observation["frame_age"], 0)
        frozen = env.observe()
        result = env.step(np.ones((2, 2)))
        assert not result.info["active"].any()
        assert not result.reward.any()
        for key, value in frozen.items():
            np.testing.assert_array_equal(result.observation[key], value)
        env.reset([0])
        assert env.observe()["step_id"].tolist() == [0, 1]


def test_render_time_depth_and_mapping():
    config = Config(
        num_envs=1,
        render="raster",
        image_size=100,
        channels=("rgb", "depth", "instance_id", "semantic_id"),
    )
    with VectorEnv(config) as env:
        env.physics.reset(np.array([0]), np.array([[0.0, 0.0]]))
        env.task.target[:] = [[0.6, 0.6]]
        env.sensors.sample(np.array([0]), env.task.scene_update(env.physics.snapshot()))
        obs = env.observe()
        assert obs["rgb"].shape == (1, 1, 100, 100, 3)
        assert obs["instance_id"][0, 0, 50, 50] == 1
        assert obs["instance_id"][0, 0, 30, 70] == 2
        np.testing.assert_array_equal(obs["instance_id"], obs["semantic_id"])
        np.testing.assert_array_equal(obs["depth"] > 0, obs["depth_valid"])
        np.testing.assert_allclose(obs["depth"][obs["depth_valid"]], 2)
        result = env.step(np.ones((1, 2)))
        np.testing.assert_allclose(result.observation["frame_age"], 0.02)
        result = env.step(np.ones((1, 2)))
        np.testing.assert_allclose(result.observation["frame_age"], 0)


def test_validation_and_close():
    env = VectorEnv(Config(num_envs=2))
    for actions in (np.zeros((1, 2)), np.full((2, 2), np.nan), np.ones((2, 2)) * 2):
        with pytest.raises(ValueError):
            env.step(actions)
    for ids in ([0, 0], [-1], [2], [0.1]):
        with pytest.raises(ValueError):
            env.reset(ids)
    env.close()
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.observe()


def test_mujoco_agrees_on_simple_dynamics_and_render():
    pytest.importorskip("mujoco")
    config = dict(num_envs=3, render="raster", channels=("rgb",))
    with (
        VectorEnv(Config(**config)) as reference,
        VectorEnv(Config(physics="mujoco", **config)) as actual,
    ):
        for _ in range(10):
            actions = np.full((3, 2), 0.3)
            a, b = reference.step(actions), actual.step(actions)
            np.testing.assert_allclose(
                a.observation["proprio"], b.observation["proprio"], atol=1e-6
            )
            np.testing.assert_array_equal(a.observation["rgb"], b.observation["rgb"])
        frozen = actual.observe()["proprio"][2].copy()
        actual.reset([0])
        np.testing.assert_array_equal(frozen, actual.observe()["proprio"][2])


def test_failed_renderer_initialization_closes_both_backends(monkeypatch):
    from embodiedforge.backends import PHYSICS, RENDER, Registration
    from embodiedforge.core import Capabilities

    closed = []

    class Physics(NumpyPhysics):
        def close(self):
            closed.append("physics")

    class Renderer:
        def build(self, scene, config):
            raise RuntimeError("renderer initialization failed")

        def close(self):
            closed.append("renderer")

    monkeypatch.setitem(PHYSICS, "test", Registration(Physics, Physics.capabilities))
    monkeypatch.setitem(RENDER, "test", Registration(Renderer, Capabilities()))
    with pytest.raises(RuntimeError, match="initialization failed"):
        VectorEnv(Config(physics="test", render="test"))
    assert closed == ["renderer", "physics"]
