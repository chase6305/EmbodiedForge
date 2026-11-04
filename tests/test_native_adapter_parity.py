"""Real native physics/observations must match an unwrapped environment exactly."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from embodiedforge.envs.locomotion import Go1Environment, H1Environment


@pytest.mark.parametrize("kind", ["go1", "h1"])
def test_adapter_matches_native_actions_resets_and_final_observations(kind):
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    if kind == "go1":
        pytest.importorskip("mujoco_menagerie")
        from embodiedforge.locomotion.go1 import Go1

        reference = Go1(3, seed=17, num_threads=1, episode_steps=3)
        native = Go1(3, seed=17, num_threads=1, episode_steps=3)
        env = Go1Environment(native)
        actions = np.random.default_rng(19).normal(0, 0.1, (8, 3, 12))

        def state(owner):
            return owner.qpos, owner.qvel, owner.ctrl

        rng = native.rng
    else:
        from embodiedforge.locomotion.h1_native import H1, build_model

        path = Path(
            os.environ.get(
                "EF_H1_MODEL",
                "/home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml",
            )
        )
        if not path.is_file():
            pytest.skip("Set EF_H1_MODEL to check real native H1 adapter parity")
        reference = H1(build_model(path), 3, seed=17, threads=1, episode_steps=3)
        native = H1(build_model(path), 3, seed=17, threads=1, episode_steps=3)
        env = H1Environment(native)
        actions = np.random.default_rng(19).normal(0, 0.1, (8, 3, 19))

        def state(owner):
            return owner.robot.qpos, owner.robot.qvel, owner.robot.ctrl

        rng = native.robot.rng
    np.testing.assert_array_equal(reference.obs(), env.observe()["proprio"])
    rng_before = json.dumps(rng.bit_generator.state)
    env.observe()
    env.observe()
    assert json.dumps(rng.bit_generator.state) == rng_before
    retained = []
    for action in actions:
        direct = reference.step(action)
        if kind == "go1":
            reward, done, terminated, terms = direct
            truncated = done & ~terminated
            observation = reference.obs()
        else:
            observation, reward, terminated, truncated, terms = direct
        result = env.step(action)
        retained.append((result, result.observation["proprio"].copy()))
        for left, right in zip(state(reference), state(native), strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(result.observation["proprio"], observation)
        np.testing.assert_array_equal(result.reward, reward)
        np.testing.assert_array_equal(result.terminated, terminated)
        np.testing.assert_array_equal(result.truncated, truncated)
        for name, value in terms.items():
            np.testing.assert_array_equal(result.info[f"reward/{name}"], value)
        ids = np.flatnonzero(terminated | truncated)[::-1].copy()
        if ids.size:
            direct = reference.reset(ids)
            if direct is None:
                direct = reference.obs()
            reset = env.reset(ids)
            np.testing.assert_array_equal(reset["proprio"], direct[ids])
    for result, original in retained:
        np.testing.assert_array_equal(result.observation["proprio"], original)


@pytest.mark.parametrize("kind", ["go1", "h1"])
def test_real_native_tasks_record_complete_episodes_with_the_shared_reader(
    kind, tmp_path
):
    pytest.importorskip("mujoco")
    pytest.importorskip("mjbatch")
    from embodiedforge.data import (
        EpisodeRecorder,
        RecordingMetadata,
        action_windows,
        read_episodes,
    )
    from embodiedforge.envs.evaluation import evaluate_policy
    from embodiedforge.runtime.implementation import component

    if kind == "go1":
        pytest.importorskip("mujoco_menagerie")
        from embodiedforge.locomotion.go1 import Go1

        native = Go1(2, seed=13, num_threads=1, episode_steps=3)
        env = Go1Environment(native)
    else:
        from embodiedforge.locomotion.h1_native import H1, build_model

        path = Path(
            os.environ.get(
                "EF_H1_MODEL",
                "/home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml",
            )
        )
        if not path.is_file():
            pytest.skip("Set EF_H1_MODEL for real native recording checks")
        native = H1(build_model(path), 2, seed=13, threads=1, episode_steps=3)
        env = H1Environment(native)
    metadata = RecordingMetadata(
        family=kind,
        max_steps=3,
        seed=13,
        instruction="Follow the velocity command in the observation",
        scene=f"native-{kind}:v1",
        scene_spec={"kind": kind},
        config={"num_envs": 2, "seed": 13, "episode_steps": 3, "threads": 1},
        source={"environment": component(native), "asset_identity": "not_recorded"},
    )
    output = tmp_path / "episodes"
    with EpisodeRecorder(
        output, env, metadata=metadata, write_workers=1, episode_format="compact"
    ) as writer:
        summary = evaluate_policy(
            env,
            lambda obs: np.zeros((2, env.spec["action_shape"][0])),
            steps=5,
            writer=writer,
        )
    episodes = list(read_episodes(output))
    assert len(episodes) == summary.completed_episodes
    assert len(episodes) >= 2
    assert all(
        (episode["terminated"] | episode["truncated"])[-1] for episode in episodes
    )
    assert list(action_windows(output, history=2, horizon=1))
