"""Cross-module contracts: substitution, ownership, rejection and work budgets."""

from dataclasses import replace

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.backends import PHYSICS, RENDER, Registration, execution_plan
from embodiedforge.backends.physics import NumpyPhysics
from embodiedforge.backends.render import RasterRenderer
from embodiedforge.core import Capabilities
from embodiedforge.tasks import ReachTask


@pytest.mark.parametrize("task,features", [("reach", 6), ("hold", 4)])
@pytest.mark.parametrize("physics", ["numpy", "mujoco"])
@pytest.mark.parametrize("render", ["null", "raster"])
def test_task_backend_substitution(task, features, physics, render):
    if physics == "mujoco":
        pytest.importorskip("mujoco")
    config = Config(
        task=task,
        physics=physics,
        render=render,
        num_envs=2,
        channels=("rgb",) if render == "raster" else (),
    )
    assert execution_plan(config)["proprio_dim"] == features
    with VectorEnv(config) as env:
        obs = env.observe()
        assert obs["proprio"].shape == (2, features)
        result = env.step(env.task.expert_action(obs))
        untouched = result.observation["proprio"][1].copy()
        reset = env.reset([0])
        assert reset["env_id"].tolist() == [0]
        np.testing.assert_array_equal(env.observe()["proprio"][1], untouched)


def test_physics_snapshot_budget_and_no_render_for_state_training(monkeypatch):
    class CountingPhysics(NumpyPhysics):
        count = 0

        def snapshot(self):
            self.count += 1
            return super().snapshot()

    class NeverRender(RasterRenderer):
        def sync(self, update):
            pytest.fail("state-only training must not sync the renderer")

    monkeypatch.setitem(
        PHYSICS, "counting", Registration(CountingPhysics, CountingPhysics.capabilities)
    )
    monkeypatch.setitem(RENDER, "never", Registration(NeverRender, Capabilities()))
    with VectorEnv(Config(physics="counting", render="never", num_envs=1)) as env:
        before = env.physics.count
        env.step(np.zeros((1, 2)))
        assert env.physics.count - before <= 2


@pytest.mark.parametrize("fault", ["stale", "dtype", "missing", "depth_valid"])
def test_bad_sensor_batch_does_not_partially_update_cache(monkeypatch, fault):
    class BrokenRenderer(RasterRenderer):
        broken = False

        def render(self, ids):
            batch = super().render(ids)
            if self.broken:
                if fault == "stale":
                    batch.version -= 1
                elif fault == "dtype":
                    batch.images["rgb"] = batch.images["rgb"].astype(np.float32)
                elif fault == "missing":
                    del batch.images["rgb"]
                else:
                    batch.images["depth_valid"][:] = False
            return batch

    monkeypatch.setitem(
        RENDER, "broken", Registration(BrokenRenderer, RasterRenderer.capabilities)
    )
    with VectorEnv(
        Config(render="broken", num_envs=1, channels=("rgb", "depth"))
    ) as env:
        previous = env.sensors.observe(np.zeros(1))
        env.renderer.broken = True
        with pytest.raises(ValueError):
            env.sensors.sample(
                np.array([0]), env.task.scene_update(env.physics.snapshot())
            )
        for name, value in previous.items():
            np.testing.assert_array_equal(env.sensors.observe(np.zeros(1))[name], value)


def test_task_features_must_not_overwrite_runtime_metadata():
    class InvalidTask(ReachTask):
        def observe(self, state):
            return {**super().observe(state), "time": np.zeros(len(state.time))}

    with pytest.raises(ValueError, match="overwrite"):
        VectorEnv(task=InvalidTask())


def test_task_defines_action_bounds_and_feature_ownership():
    class StrongActuatorTask(ReachTask):
        spec = replace(ReachTask.spec, action_low=-2.0, action_high=2.0)

        def observe(self, state):
            # An adapter may return its own borrowed cache; runtime copies it.
            self.features = super().observe(state)["proprio"]
            return {"proprio": self.features}

    with VectorEnv(Config(num_envs=1), task=StrongActuatorTask()) as env:
        result = env.step(np.full((1, 2), 1.5))
        expected = result.observation["proprio"].copy()
        env.task.features[:] = 99
        np.testing.assert_array_equal(result.observation["proprio"], expected)
        with pytest.raises(ValueError, match="actions must lie"):
            env.step(np.full((1, 2), 2.1))
