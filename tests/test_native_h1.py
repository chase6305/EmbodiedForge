"""Native articulation contracts and H1 training without IsaacLab."""

import json
import signal
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


@pytest.mark.parametrize("backend", ["h1", "core"])
@pytest.mark.parametrize(
    "reset_step,reset_count", [(None, 0), (0, 1), (1, 2), (3, 2), (1, 3)]
)
def test_rollout_reuses_values_without_crossing_reset_or_policy_update(
    backend, reset_step, reset_count
):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from embodiedforge.locomotion.h1_ppo import collect
    from embodiedforge.training import _collect_rollout

    class Policy:
        calls = 0
        rows = 0
        offset = 0

        def distribution(self, obs):
            return torch.distributions.Normal(torch.zeros((len(obs), 19)), 1)

        def value(self, obs):
            self.calls += 1
            self.rows += len(obs)
            return obs[:, 0] + self.offset

        def to_action(self, raw):
            return raw

    class Env:
        step_index = 0

        def step(self, actions):
            ended = (self.step_index == reset_step) & (np.arange(3) < reset_count)
            self.step_index += 1
            result = SimpleNamespace(
                observation={
                    "proprio": np.full((3, 69), 10 * self.step_index, np.float32)
                },
                reward=np.ones(3, np.float32),
                terminated=ended & (np.arange(3) % 2 == 1),
                truncated=ended & (np.arange(3) % 2 == 0),
                info={"success": np.zeros(3, bool)},
            )
            if backend == "core":
                return result
            return (
                result.observation["proprio"],
                result.reward,
                result.terminated,
                result.truncated,
                result.info,
            )

        def reset(self, ids):
            if backend == "core":
                return {"proprio": np.zeros((len(ids), 69), np.float32)}
            return np.zeros((3, 69), np.float32)

    def collect_batch(observation, horizon):
        if backend == "core":
            batch, obs, *_ = _collect_rollout(
                env, policy, {"proprio": observation}, horizon
            )
            return batch, obs["proprio"]
        return collect(policy, env, observation, horizon)

    env, policy = Env(), Policy()
    rollout, observation = collect_batch(np.zeros((3, 69), np.float32), 4)
    expected = np.repeat(np.arange(4, dtype=np.float32)[:, None] * 10, 3, axis=1)
    if reset_step is not None and reset_step < 3:
        expected[reset_step + 1, :reset_count] = 0
    np.testing.assert_array_equal(rollout["obs"][:, :, 0], expected)
    np.testing.assert_array_equal(rollout["value"], expected)
    np.testing.assert_array_equal(
        rollout["next_value"], np.repeat(np.arange(1, 5)[:, None] * 10, 3, axis=1)
    )
    assert policy.calls == 5 + int(reset_step is not None and reset_step < 3)
    assert policy.rows == 15 + reset_count * int(
        reset_step is not None and reset_step < 3
    )
    saved = {key: value.copy() for key, value in rollout.items()}
    policy.offset = 100
    expected = observation[:, 0].copy() + policy.offset
    following, _ = collect_batch(observation, 1)
    np.testing.assert_array_equal(following["value"][0], expected)
    observation.fill(-999)
    for key, value in rollout.items():
        np.testing.assert_array_equal(value, saved[key], strict=True)


@pytest.mark.parametrize("damage", ["negative_second_moment", "missing_parameter"])
def test_resume_rejects_damaged_adam_before_creating_run(tmp_path, monkeypatch, damage):
    from types import SimpleNamespace

    torch = pytest.importorskip("torch")
    from embodiedforge import native_h1

    policy = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(policy.parameters())
    policy(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    state = optimizer.state_dict()
    key = next(iter(state["state"]))
    if damage == "negative_second_moment":
        state["state"][key]["exp_avg_sq"].fill_(-1)
    else:
        del state["state"][key]
    original_handler = signal.getsignal(signal.SIGTERM)

    def load_run(directory):
        assert callable(signal.getsignal(signal.SIGTERM))
        assert signal.getsignal(signal.SIGTERM) is not original_handler
        return None, policy, {"updates": 1, "optimizer": state}, {}

    monkeypatch.setattr(native_h1, "load_run", load_run)
    output = tmp_path / "resumed"
    args = SimpleNamespace(
        threads=1, seed=0, resume=tmp_path / "input", learning_rate=0.001, output=output
    )
    with pytest.raises(ValueError, match="H1 optimizer"):
        native_h1.train(args)
    assert not output.exists()
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_saved_h1_artifacts_load_after_interruption_and_source_replacement(
    h1_path, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from embodiedforge import native_h1
    from embodiedforge.locomotion.h1_native import VERSION, build_model
    from embodiedforge.locomotion.h1_ppo import Policy

    model, policy = build_model(h1_path), Policy()
    model_path, weights = tmp_path / "model.mjb", tmp_path / "checkpoint.pt"
    mujoco.mj_saveModel(model, str(model_path))
    model_hash = native_h1.digest(model_path)
    torch.save(
        {
            "task": VERSION,
            "updates": 1,
            "model_sha256": model_hash,
            "joint_names": [
                model.joint(int(j)).name for j in model.actuator_trnid[:, 0]
            ],
            "policy": policy.state_dict(),
        },
        weights,
    )
    metadata = dict(
        task=VERSION,
        status="complete",
        updates=1,
        model_sha256=model_hash,
        checkpoint_sha256=native_h1.digest(weights),
    )
    for status in ("running", "failed", "interrupted", "complete"):
        metadata["status"] = status
        native_h1.write_json(tmp_path / "run.json", metadata)
        if status == "running":
            with pytest.raises(ValueError, match="stopped native H1 run"):
                native_h1.load_run(tmp_path)
        else:
            _, _, restored, _ = native_h1.load_run(tmp_path)
            assert restored["updates"] == 1
    native_h1.write_json(
        tmp_path / "run.json",
        {**metadata, "status": "interrupted", "checkpoint_sha256": None},
    )
    with pytest.raises(ValueError, match="saved checkpoint"):
        native_h1.load_run(tmp_path)
    metadata["status"] = "interrupted"
    native_h1.write_json(tmp_path / "run.json", metadata)
    original_load = torch.load

    def replace_sources(*args, **kwargs):
        model_path.write_bytes(b"replaced before MuJoCo load")
        weights.write_bytes(b"replaced before Torch load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", replace_sources)
    loaded_model, loaded_policy, _, loaded_metadata = native_h1.load_run(tmp_path)
    assert loaded_metadata == metadata
    np.testing.assert_array_equal(loaded_model.body_mass, model.body_mass)
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(
            loaded_policy.state_dict()[key], value, rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "failure", [RuntimeError("optimizer failed"), KeyboardInterrupt(), None]
)
def test_training_final_record_failure_preserves_primary_error(
    h1_path, tmp_path, monkeypatch, caplog, failure
):
    pytest.importorskip("torch")
    from embodiedforge import native_h1
    from embodiedforge.locomotion import h1_ppo

    write_json = native_h1.write_json
    disk_error = OSError("disk full")
    final_records = []

    def save_record(path, value):
        if path.name == "run.json" and value["status"] != "running":
            final_records.append(dict(value))
            raise disk_error
        write_json(path, value)

    if failure is not None:

        def fail_optimizer(*args, **kwargs):
            raise failure

        monkeypatch.setattr(h1_ppo, "optimize", fail_optimizer)
    monkeypatch.setattr(native_h1, "write_json", save_record)
    output = tmp_path / "train"
    expected = disk_error if failure is None else failure
    with pytest.raises(type(expected)) as caught:
        native_h1.main(
            [
                "train",
                "--model",
                str(h1_path),
                "--output",
                str(output),
                "--updates",
                "1",
                "--horizon",
                "1",
                "--num-envs",
                "2",
                "--threads",
                "1",
            ]
        )
    assert caught.value is expected
    assert len(final_records) == 1
    if failure is None:
        assert final_records[0]["status"] == "complete"
        assert (output / "checkpoint.pt").is_file()
    else:
        status = "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
        assert final_records[0]["status"] == status
        assert "disk full" in caplog.text
        assert str(output / "run.json") in caplog.text
    saved = json.loads((output / "run.json").read_text())
    assert saved["status"] == "running"
    assert saved["learning_rate"] == 3e-4


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
            "--headless",
            "--model",
            str(h1_path),
            "--learning-rate",
            "0.0002",
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
            "--headless",
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
    assert report["headless"] is True
    assert report["learning_rate"] == 2e-4
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
