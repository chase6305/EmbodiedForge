"""Yaw tracking retains the original world-frame projection without unused axes."""

import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("count", [1, 17])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_h1_yaw_metrics_keep_full_projection_values(
    monkeypatch, count, training, dtype
):
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    from embodiedforge.locomotion import h1_native

    model_path = Path(
        os.environ.get(
            "EF_H1_MODEL",
            "/home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml",
        )
    )
    if not model_path.exists():
        pytest.skip("Set EF_H1_MODEL to run real H1 asset integration checks")
    env = h1_native.H1(
        h1_native.build_model(model_path), count, threads=1, seed=818, training=training
    )
    rng = np.random.default_rng(818)
    env.robot.reset(qpos=env.robot.qpos, qvel=rng.normal(0, 0.05, env.robot.qvel.shape))
    einsum = np.einsum
    projected_shapes = []

    def track(*args, **kwargs):
        result = einsum(*args, **kwargs)
        projected_shapes.append(result.shape)
        return result

    monkeypatch.setattr(h1_native.np, "einsum", track)
    for _ in range(3):
        actions = rng.normal(0, 0.02, (count, 19)).astype(dtype)
        before = actions.copy()
        _, _, _, _, terms = env.step(actions)
        expected = einsum("nij,nj->ni", env.rotation, env.gyro)[:, 2]
        np.testing.assert_array_equal(env.measured_velocity[:, 2], expected)
        np.testing.assert_array_equal(
            terms["track_yaw"], np.exp(-((expected - env.commands[:, 2]) ** 2) / 0.25)
        )
        np.testing.assert_array_equal(actions, before)
    assert projected_shapes == [(count,)] * 3
