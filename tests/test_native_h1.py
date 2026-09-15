"""Native articulation contracts and H1 training without IsaacLab."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("mjbatch")


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string("""<mujoco><option timestep=".005"/>
      <worldbody><body pos="0 0 1"><joint name="left" axis="0 1 0"/>
      <geom type="capsule" size=".05 .2" mass="1" pos="0 0 -.2"/>
      <body pos="0 0 -.4"><joint name="right" axis="0 1 0"/>
      <geom type="capsule" size=".05 .2" mass="1" pos="0 0 -.2"/></body>
      </body></worldbody><actuator>
      <position name="right_drive" joint="right" kp="20" kv="1" forcerange="-2 2"/>
      <position name="left_drive" joint="left" kp="30" kv="2" forcerange="-3 3"/>
      </actuator></mujoco>""")


def test_named_groups_and_commands_do_not_assume_joint_order(model):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, 3, threads=1)
    assert robot.names == ("right", "left")
    assert robot.group("left").tolist() == [1]
    robot.set_joint_position_targets(
        [[0.2], [0.4]], env_ids=[2, 0], joint_ids=robot.group("left")
    )
    np.testing.assert_allclose(robot.ctrl, [[0, 0.4], [0, 0], [0, 0.2]])
    with pytest.raises(ValueError, match="matches nothing"):
        robot.group("missing")
    with pytest.raises(ValueError):
        robot.set_joint_position_targets([[9], [9]], env_ids=[1, 1], joint_ids=[0])
    assert robot.ctrl[1, 0] == 0


def test_torques_and_state_match_direct_mujoco_and_snapshots_are_owned(model):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, 2, threads=1)
    robot.set_joint_position_targets([[1, -1], [0, 0]])
    data = mujoco.MjData(model)
    data.ctrl[:] = (1, -1)
    for _ in range(4):
        mujoco.mj_step(model, data)
    force, torque = data.actuator_force.copy(), data.qfrc_actuator[robot.vadr].copy()
    robot.step(4)
    np.testing.assert_allclose(robot.qpos[0], data.qpos, atol=1e-12)
    np.testing.assert_allclose(robot.actuator_force[0], force, atol=1e-12)
    np.testing.assert_allclose(robot.joint_force[0], torque, atol=1e-12)
    assert (abs(force) <= [2, 3]).all()
    snapshot = robot.snapshot()
    robot.reset([0])
    assert np.any(snapshot["joint_torque"][0])
    assert not robot.joint_force[0].any()


def test_selective_randomization_recomputes_constants_and_reset_is_isolated(model):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, 3, threads=1, seed=4)
    robot.set_joint_position_targets(np.ones((3, 2)))
    robot.step(2)
    untouched = robot.qpos[[0, 2]].copy()
    robot.randomize([1], body_ids=[1], mass_scale=(2, 2), friction=(0.7, 0.7))
    masses = robot.batch.expand("body_mass")
    np.testing.assert_allclose(masses[[0, 2]], np.tile(model.body_mass, (2, 1)))
    assert masses[1, 1] == 2 * model.body_mass[1]
    assert robot.batch.expand("body_subtreemass")[1, 1] == 3
    robot.reset([1])
    np.testing.assert_array_equal(robot.qpos[[0, 2]], untouched)
    # A second draw scales nominal parameters, never compounds the first draw.
    robot.randomize([1], body_ids=[1], mass_scale=(2, 2), friction=(0.7, 0.7))
    assert masses[1, 1] == 2 * model.body_mass[1]
    before, rng = robot.qpos.copy(), json.dumps(robot.rng.bit_generator.state)
    with pytest.raises(ValueError):
        robot.randomize([1], mass_scale=(2, -1))
    np.testing.assert_array_equal(robot.qpos, before)
    assert json.dumps(robot.rng.bit_generator.state) == rng


def test_motor_actuators_are_not_misinterpreted_as_position_targets(model):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    model.actuator_biastype[0] = mujoco.mjtBias.mjBIAS_NONE
    with pytest.raises(ValueError, match="Position control"):
        ArticulationBatch(model, 1)


@pytest.fixture
def h1_path():
    import os

    path = Path(
        os.environ.get(
            "EF_H1_MODEL",
            "/home/ubuntu/workspace/3rdparty/mink/examples/unitree_h1/h1.xml",
        )
    )
    if not path.exists():
        pytest.skip("Set EF_H1_MODEL to run real H1 asset integration checks")
    return path


def test_h1_standing_pose_preserves_original_mjcf_kinematics(h1_path):
    from embodiedforge.locomotion.h1_native import H1, build_model

    original = mujoco.MjModel.from_xml_path(str(h1_path))
    model = build_model(h1_path)
    np.testing.assert_array_equal(model.qpos0, original.qpos0)
    env = H1(model, 2, training=False, threads=1)
    reference = mujoco.MjData(original)
    reference.qpos[:] = env.robot.qpos[0]
    mujoco.mj_forward(original, reference)
    np.testing.assert_allclose(env.batch.bind("xpos")[0], reference.xpos, atol=1e-12)
    assert env.obs().shape == (2, 69)
    env.set_commands([[0.5, 0, 0]], [1])
    env.reset([1])
    np.testing.assert_array_equal(env.commands[1], [0.5, 0, 0])
    with pytest.raises(ValueError):
        env.set_commands([[2, 0, 0]], [1])


def test_rollout_bootstraps_pre_reset_timeout_observation():
    torch = pytest.importorskip("torch")
    from embodiedforge.locomotion.h1_ppo import collect

    class Policy:
        def distribution(self, obs):
            return torch.distributions.Normal(torch.zeros((len(obs), 19)), 1)

        def value(self, obs):
            return obs[:, 0]

    class Env:
        def step(self, actions):
            return (
                np.full((1, 69), 10, np.float32),
                np.ones(1),
                np.array([False]),
                np.array([True]),
                {},
            )

        def reset(self, ids):
            return np.zeros((1, 69), np.float32)

    rollout, observation = collect(Policy(), Env(), np.zeros((1, 69), np.float32), 2)
    np.testing.assert_array_equal(rollout["next_value"], [[10], [10]])
    np.testing.assert_array_equal(rollout["value"], [[0], [0]])
    assert not observation.any()


def test_native_training_resume_evaluate_under_import_blocker(h1_path, tmp_path):
    pytest.importorskip("torch")
    script = """import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.startswith(("isaaclab", "isaacsim", "omni", "rsl_rl")):
   raise RuntimeError("Forbidden runtime import: " + fullname)
sys.meta_path.insert(0, Block())
from embodiedforge.native_h1 import main
main(sys.argv[1:])
"""
    run, resumed, evaluation = [tmp_path / name for name in ("train", "resume", "eval")]
    commands = [
        [
            "train",
            "--model",
            str(h1_path),
            "--updates",
            "2",
            "--horizon",
            "4",
            "--output",
            str(run),
        ],
        [
            "train",
            "--resume",
            str(run),
            "--updates",
            "1",
            "--horizon",
            "4",
            "--output",
            str(resumed),
        ],
        [
            "evaluate",
            "--run",
            str(resumed),
            "--steps",
            "5",
            "--record-motion",
            "--output",
            str(evaluation),
        ],
    ]
    for args in commands:
        subprocess.run(
            [sys.executable, "-c", script, *args, "--num-envs", "2", "--threads", "1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    report = json.loads((resumed / "run.json").read_text())
    for path in (run, resumed):
        runtime = json.loads((path / "training-runtime.json").read_text())
        assert runtime["core_vector_env"] is False
        assert runtime["environment"]["module"] == "embodiedforge.locomotion.h1_native"
        assert runtime["learner"]["module"] == "embodiedforge.locomotion.h1_ppo"
        assert runtime["physics_adapter"]["module"].startswith("mjbatch")
    assert report["updates"] == 3 and report["initial_updates"] == 2
    assert report["status"] == "complete"
    assert (evaluation / "motion.npz").is_file()
    from embodiedforge.robot_replay import RobotMotion

    replay = RobotMotion(
        evaluation / "motion.npz",
        mujoco.MjModel.from_binary_path(str(resumed / "model.mjb")),
    )
    assert replay.max_position_error < 1e-5
    from embodiedforge.native_h1 import load_run

    with (resumed / "model.mjb").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_run(resumed)
