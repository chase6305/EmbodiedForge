"""Native articulation contracts and H1 training without IsaacLab."""

import json
import os
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


@pytest.mark.parametrize("count", [1, 4, 17])
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
def test_joint_snapshots_keep_actuator_order_owned_contiguous_storage(
    model, count, layout
):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, count, threads=1)
    position = np.arange(count * 2, dtype=np.float64).reshape(count, 2) / 100
    velocity = position + 0.125
    robot.reset(qpos=position, qvel=velocity)
    if layout == "F":
        robot.qpos, robot.qvel = map(np.asfortranarray, (robot.qpos, robot.qvel))
    elif layout == "reverse":
        robot.qpos, robot.qvel = robot.qpos[:, ::-1], robot.qvel[:, ::-1]
    snapshot = robot.snapshot()
    assert snapshot["joint_names"] == ("right", "left")
    np.testing.assert_array_equal(snapshot["joint_position"], robot.qpos[:, [1, 0]])
    np.testing.assert_array_equal(snapshot["joint_velocity"], robot.qvel[:, [1, 0]])
    arrays = {
        name: value for name, value in snapshot.items() if isinstance(value, np.ndarray)
    }
    for value in arrays.values():
        assert (
            value.flags.owndata and value.flags.writeable and value.flags.c_contiguous
        )
    for first_name, first in arrays.items():
        for second_name, second in arrays.items():
            if first_name != second_name:
                assert not np.shares_memory(first, second)
    original_position = robot.qpos.copy()
    snapshot["joint_position"].fill(7)
    np.testing.assert_array_equal(robot.qpos, original_position)
    np.testing.assert_array_equal(snapshot["qpos"], original_position)
    frozen = {name: value.copy() for name, value in arrays.items()}
    robot.qpos.fill(99)
    robot.qvel.fill(99)
    for name, value in frozen.items():
        np.testing.assert_array_equal(snapshot[name], value)


@pytest.mark.parametrize("ids", [None, [3, 1], [2], []])
@pytest.mark.parametrize("custom_position", [False, True])
@pytest.mark.parametrize("custom_velocity", [False, True])
def test_default_and_custom_reset_rows_preserve_order_and_unselected_state(
    model, ids, custom_position, custom_velocity
):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, 4, threads=1)
    robot.reset(
        qpos=np.arange(8, dtype=np.float64).reshape(4, 2) / 100,
        qvel=np.arange(8, dtype=np.float64).reshape(4, 2) / 100 + 0.5,
    )
    robot.actuator_force.fill(2)
    robot.joint_force.fill(3)
    chosen = np.arange(4) if ids is None else np.asarray(ids, dtype=np.int64)
    untouched = np.setdiff1d(np.arange(4), chosen)
    before = {
        name: getattr(robot, name).copy()
        for name in ("qpos", "qvel", "ctrl", "actuator_force", "joint_force")
    }
    qpos = (
        np.arange(len(chosen) * 2, dtype=np.float64).reshape(-1, 2) / 100 + 0.25
        if custom_position
        else None
    )
    qvel = (
        np.arange(len(chosen) * 2, dtype=np.float64).reshape(-1, 2) / 100 - 0.25
        if custom_velocity
        else None
    )
    expected_position = (
        qpos.copy() if qpos is not None else np.tile(model.qpos0, (len(chosen), 1))
    )
    expected_velocity = (
        qvel.copy() if qvel is not None else np.zeros((len(chosen), model.nv))
    )
    robot.reset(ids, qpos=qpos, qvel=qvel)
    np.testing.assert_array_equal(robot.qpos[chosen], expected_position)
    np.testing.assert_array_equal(robot.qvel[chosen], expected_velocity)
    np.testing.assert_array_equal(robot.ctrl[chosen], expected_position[:, [1, 0]])
    for name, original in before.items():
        np.testing.assert_array_equal(
            getattr(robot, name)[untouched], original[untouched]
        )
    np.testing.assert_array_equal(
        robot.actuator_force[chosen], np.zeros((len(chosen), 2))
    )
    np.testing.assert_array_equal(robot.joint_force[chosen], np.zeros((len(chosen), 2)))
    if qpos is not None:
        qpos.fill(99)
        np.testing.assert_array_equal(robot.qpos[chosen], expected_position)
    if qvel is not None:
        qvel.fill(99)
        np.testing.assert_array_equal(robot.qvel[chosen], expected_velocity)


def test_reset_copies_borrowed_state_before_backend_reset(model):
    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, 4, threads=1)
    robot.reset(qpos=np.arange(8).reshape(4, 2) / 100, qvel=np.ones((4, 2)))
    position, velocity = robot.qpos[1::2], robot.qvel[1::2]
    expected_position, expected_velocity = position.copy(), velocity.copy()
    robot.reset([3, 1], qpos=position, qvel=velocity)
    np.testing.assert_array_equal(robot.qpos[[3, 1]], expected_position)
    np.testing.assert_array_equal(robot.qvel[[3, 1]], expected_velocity)


def test_default_reset_snapshots_template_before_backend_can_change_it(model):
    from types import SimpleNamespace

    from embodiedforge.locomotion.articulation import ArticulationBatch

    robot = ArticulationBatch(model, 4, threads=1)
    expected = model.qpos0.copy()
    backend = robot.batch

    def reset(ids):
        backend.reset(ids)
        model.qpos0[:] = expected + 0.25

    robot.batch = SimpleNamespace(reset=reset, forward=backend.forward)
    robot.reset([3, 1])
    np.testing.assert_array_equal(robot.qpos[[3, 1]], np.tile(expected, (2, 1)))
    np.testing.assert_array_equal(robot.ctrl[[3, 1]], np.tile(expected[[1, 0]], (2, 1)))


def test_empty_reset_keeps_quaternion_validation_and_no_mutation_semantics():
    from embodiedforge.locomotion.articulation import ArticulationBatch

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body pos="0 0 1"><freejoint/>
      <geom size=".1" mass="1"/><body><joint name="joint"/><geom size=".1" mass="1"/></body>
      </body></worldbody><actuator><position joint="joint" kp="10"/></actuator></mujoco>""")
    robot = ArticulationBatch(model, 2, threads=1)
    before = robot.snapshot()
    model.qpos0[3:7] = 0
    robot.reset([])
    with pytest.raises(ValueError, match="quaternions"):
        robot.reset([0])
    with pytest.raises(ValueError, match="qvel"):
        robot.reset([], qvel=[])
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(robot.snapshot()[name], value)


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


@pytest.mark.parametrize("count", [2, 17, 64])
def test_h1_acceleration_reward_keeps_contiguous_reduction_rounding(h1_path, count):
    from embodiedforge.locomotion.h1_native import CTRL_DT, H1, build_model

    env = H1(build_model(h1_path), count, threads=1, training=False, seed=412)
    rng = np.random.default_rng(412 + count)
    env.robot.reset(
        qpos=env.robot.qpos,
        qvel=rng.normal(0, 0.02, env.robot.qvel.shape),
    )
    previous = env.robot.qvel[:, env.robot.vadr].copy(order="C")
    _, _, _, _, terms = env.step(rng.normal(0, 0.01, (count, 19)))
    expected = -1.25e-7 * np.sum(
        ((env.robot.qvel[:, env.robot.vadr] - previous) / CTRL_DT) ** 2, axis=1
    )
    np.testing.assert_array_equal(terms["acceleration"], expected)


@pytest.mark.parametrize("count", [1, 17])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided", "broadcast"])
def test_h1_action_clipping_preserves_caller_storage(h1_path, count, dtype, layout):
    from embodiedforge.locomotion.h1_native import ACTION_LIMIT, H1, build_model

    env = H1(build_model(h1_path), count, threads=1, training=False, seed=414)
    actions = np.linspace(
        -2 * ACTION_LIMIT, 2 * ACTION_LIMIT, count * 19, dtype=dtype
    ).reshape(count, 19)
    if layout == "F":
        actions = np.asfortranarray(actions)
    elif layout == "strided":
        padded = np.zeros((count, 38), dtype=dtype)
        padded[:, ::2] = actions
        actions = padded[:, ::2]
    elif layout == "broadcast":
        actions = np.broadcast_to(actions[0], (count, 19))
    original = actions.copy()
    expected = np.clip(original.astype(np.float64), -ACTION_LIMIT, ACTION_LIMIT)
    env.step(actions)
    np.testing.assert_array_equal(actions, original)
    np.testing.assert_array_equal(env.action, expected)


@pytest.mark.parametrize("outcome", ["complete", "rejected", "failed", "interrupted"])
def test_native_evaluation_records_lifecycle_and_retains_primary_failures(
    h1_path, tmp_path, monkeypatch, outcome
):
    from types import SimpleNamespace

    pytest.importorskip("torch")
    from embodiedforge import native_h1
    from embodiedforge.locomotion.h1_config import VERSION
    from embodiedforge.locomotion.h1_native import build_model
    from embodiedforge.locomotion.h1_ppo import Policy
    from embodiedforge.runtime.records import read_run

    model, policy = build_model(h1_path), Policy()
    metadata = {
        "task": VERSION,
        "checkpoint_sha256": "a" * 64,
        "model_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        native_h1, "load_run", lambda run: (model, policy, {}, metadata)
    )
    failure = (
        KeyboardInterrupt()
        if outcome == "interrupted"
        else RuntimeError("evaluation failed")
    )

    def evaluate(args, *unused):
        if outcome in ("failed", "interrupted"):
            raise failure
        native_h1.write_json(
            args.output / "evaluation.json", {"accepted": outcome == "complete"}
        )
        return 2 if outcome == "rejected" else 0

    monkeypatch.setattr(native_h1, "_evaluate_native", evaluate)
    args = SimpleNamespace(
        threads=1,
        run=tmp_path / "source",
        num_envs=2,
        seed=0,
        randomized_reset=False,
        steps=2,
        velocity=[0.2, 0, 0],
        output=tmp_path / "evaluation",
    )
    if outcome in ("failed", "interrupted"):
        with pytest.raises(type(failure)) as caught:
            native_h1.evaluate(args)
        assert caught.value is failure
    else:
        assert native_h1.evaluate(args) == (2 if outcome == "rejected" else 0)
    summary = read_run(args.output)
    assert summary["workflow"] == "h1-native"
    assert summary["operation"] == "evaluate"
    assert summary["status"] == outcome
    saved = json.loads((args.output / "run.json").read_text())
    if outcome in ("complete", "rejected"):
        assert saved["result"]["sha256"] == native_h1.digest(
            args.output / "evaluation.json"
        )
    else:
        assert "result" not in saved


@pytest.mark.parametrize("randomized_reset", [False, True])
def test_h1_standing_pose_preserves_original_mjcf_kinematics(h1_path, randomized_reset):
    from embodiedforge.locomotion.h1_native import H1, build_model

    original = mujoco.MjModel.from_xml_path(str(h1_path))
    model = build_model(h1_path)
    np.testing.assert_array_equal(model.qpos0, original.qpos0)
    env = H1(model, 2, training=False, randomized_reset=randomized_reset, threads=1)
    reference = mujoco.MjData(original)
    reference.qpos[:] = env.robot.qpos[0]
    mujoco.mj_forward(original, reference)
    np.testing.assert_allclose(env.batch.bind("xpos")[0], reference.xpos, atol=1e-12)
    assert env.obs().shape == (2, 69)
    # Reset randomization must not enable observation noise or couple rows.
    np.testing.assert_array_equal(env.obs(), env.obs())
    if randomized_reset:
        assert not np.array_equal(env.robot.qpos[0], env.robot.qpos[1])
        same = H1(model, 2, training=False, randomized_reset=True, threads=1)
        np.testing.assert_array_equal(env.robot.qpos, same.robot.qpos)
        np.testing.assert_array_equal(env.robot.qvel, same.robot.qvel)
        np.testing.assert_array_equal(
            env.batch.expand("body_mass"), same.batch.expand("body_mass")
        )
    else:
        np.testing.assert_array_equal(env.robot.qpos[0], env.robot.qpos[1])
    untouched = env.robot.qpos[0].copy()
    env.set_commands([[0.5, 0, 0]], [1])
    env.reset([1])
    np.testing.assert_array_equal(env.robot.qpos[0], untouched)
    np.testing.assert_array_equal(env.commands[1], [0.5, 0, 0])
    with pytest.raises(ValueError):
        env.set_commands([[2, 0, 0]], [1])


def test_h1_observation_noise_preserves_scales_and_rng_stream(h1_path):
    from copy import deepcopy

    from embodiedforge.locomotion.h1_native import H1, build_model

    env = H1(build_model(h1_path), 3, seed=7, training=True, threads=1)
    rng = np.random.default_rng()
    rng.bit_generator.state = deepcopy(env.rng.bit_generator.state)
    env.training = False
    # Build the clean values in the original float64 precision before the
    # public float32 conversion; otherwise tiny quantization differences arise.
    clean = np.concatenate(
        (
            env.velocity,
            env.gyro,
            -env.rotation[:, 2, :],
            env.commands,
            env.robot.qpos[:, env.robot.qadr] - env.default,
            env.robot.qvel[:, env.robot.vadr],
            env.action,
        ),
        axis=1,
    )
    scale = np.array(
        [0.1] * 3
        + [0.2] * 3
        + [0.05] * 3
        + [0] * 3
        + [0.01] * 19
        + [1.5] * 19
        + [0] * 19
    )
    expected = (clean + rng.uniform(-1, 1, clean.shape) * scale).astype(np.float32)
    env.training = True
    np.testing.assert_array_equal(env.obs(), expected)
    assert env.rng.bit_generator.state == rng.bit_generator.state
    actions = np.random.default_rng(32)
    for _ in range(20):
        previous = env.robot.qvel[:, env.robot.vadr].copy()
        _, _, _, _, terms = env.step(actions.normal(0, 0.2, (3, 19)))
        expected = -1.25e-7 * np.sum(
            ((env.robot.qvel[:, env.robot.vadr] - previous) / 0.02) ** 2, 1
        )
        np.testing.assert_array_equal(terms["acceleration"], expected)


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
    if failure is None:
        assert len(final_records) == 2
        assert final_records[0]["status"] == "complete"
        assert final_records[1]["status"] == "failed"
        assert (output / final_records[0]["checkpoint"]).is_file()
    else:
        assert len(final_records) == 1
        status = "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
        assert final_records[0]["status"] == status
        assert "disk full" in caplog.text
        assert str(output / "run.json") in caplog.text
    saved = json.loads((output / "run.json").read_text())
    assert saved["status"] == "running"
    assert saved["learning_rate"] == 3e-4


@pytest.mark.parametrize("interrupt", [False, True])
def test_model_publication_failure_leaves_an_inspectable_startup_record(
    h1_path, tmp_path, monkeypatch, interrupt
):
    pytest.importorskip("torch")
    import mujoco

    from embodiedforge import native_h1
    from embodiedforge.runtime.records import read_run

    failure = KeyboardInterrupt() if interrupt else OSError("model write failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(mujoco, "mj_saveModel", fail)
    output = tmp_path / "train"
    with pytest.raises(type(failure)) as caught:
        native_h1.main(
            [
                "train",
                "--model",
                str(h1_path),
                "--output",
                str(output),
                "--updates",
                "1",
                "--num-envs",
                "2",
                "--threads",
                "1",
            ]
        )
    assert caught.value is failure
    assert read_run(output)["status"] == ("interrupted" if interrupt else "failed")
    record = json.loads((output / "run.json").read_text())
    assert record["model_sha256"] is None
    assert "checkpoint" not in record


@pytest.mark.parametrize(
    "cleanup_error", [None, PermissionError, KeyboardInterrupt, SystemExit]
)
@pytest.mark.parametrize(
    "boundary,error",
    [
        ("write", OSError("checkpoint write failed")),
        ("write", KeyboardInterrupt()),
        ("checkpoint", OSError("checkpoint publication failed")),
        ("checkpoint", KeyboardInterrupt()),
        ("run.json", OSError("manifest publication failed")),
    ],
)
def test_checkpoint_publication_keeps_committed_state_recoverable(
    h1_path, tmp_path, monkeypatch, boundary, error, cleanup_error, caplog
):
    torch = pytest.importorskip("torch")
    from embodiedforge import native_h1
    from embodiedforge.locomotion.h1_native import VERSION, build_model
    from embodiedforge.locomotion.h1_ppo import Policy

    model, policy = build_model(h1_path), Policy()
    mujoco.mj_saveModel(model, str(tmp_path / "model.mjb"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    sum(p.square().sum() for p in policy.parameters()).backward()
    optimizer.step()
    metadata = {
        "task": VERSION,
        "status": "running",
        "updates": 0,
        "model_sha256": native_h1.digest(tmp_path / "model.mjb"),
    }

    def save(update):
        native_h1.save_checkpoint(
            tmp_path,
            {
                "task": VERSION,
                "updates": update,
                "model_sha256": metadata["model_sha256"],
                "joint_names": [
                    model.joint(int(j)).name for j in model.actuator_trnid[:, 0]
                ],
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            optimizer,
            metadata,
        )

    save(50)
    save(100)
    committed = dict(metadata)
    replace = os.replace

    def fail_after_replace(path, target):
        result = replace(path, target)
        name = Path(target).name
        if (boundary == "checkpoint" and name == "checkpoint-000000150.pt") or (
            boundary == "run.json" and name == "run.json"
        ):
            raise error
        return result

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_after_replace)
        if boundary == "write":

            def partial_save(checkpoint, path):
                Path(path).write_bytes(b"partial H1 checkpoint")
                raise error

            patch.setattr(torch, "save", partial_save)
        if cleanup_error:

            def fail_cleanup(path, **kwargs):
                raise cleanup_error("H1 temporary cleanup interrupted")

            patch.setattr(os, "unlink", fail_cleanup)
        with pytest.raises(type(error)) as caught:
            save(150)
    assert caught.value is error
    if cleanup_error:
        assert "H1 temporary cleanup interrupted" in caplog.text
    assert metadata == committed
    # Model the trainer's error finalization, then load through the real consumer.
    native_h1.write_json(tmp_path / "run.json", {**metadata, "status": "failed"})
    _, _, restored, _ = native_h1.load_run(tmp_path)
    assert restored["updates"] == 100
    assert (
        native_h1.digest(tmp_path / committed["checkpoint"])
        == committed["checkpoint_sha256"]
    )
    partials = list(tmp_path.glob("*.tmp"))
    if boundary == "write" and cleanup_error:
        assert partials == [tmp_path / "checkpoint-000000150.pt.tmp"]
        assert partials[0].read_bytes() == b"partial H1 checkpoint"
    else:
        assert not partials
    save(200)
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*.pt")) == [
        "checkpoint-000000100.pt",
        "checkpoint-000000200.pt",
    ]
    native_h1.write_json(
        tmp_path / "run.json",
        {**metadata, "status": "complete", "checkpoint": "../outside.pt"},
    )
    with pytest.raises(ValueError, match="checkpoint filename"):
        native_h1.load_run(tmp_path)


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
    run, resumed, evaluation, randomized = [
        tmp_path / name for name in ("train", "resume", "eval", "randomized")
    ]
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
        [
            "evaluate",
            "--run",
            str(resumed),
            "--randomized-reset",
            "--record-motion",
            "--record-env",
            "1",
            "--seed",
            "42",
            "--steps",
            "1000",
            "--output",
            str(randomized),
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
    original = json.loads((run / "run.json").read_text())
    assert original["resume_checkpoint_sha256"] is None
    assert report["resume_checkpoint_sha256"] == original["checkpoint_sha256"]
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
    nominal = json.loads((evaluation / "evaluation.json").read_text())
    varied = json.loads((randomized / "evaluation.json").read_text())
    assert nominal["identical_initial_states"] is True
    assert varied["identical_initial_states"] is False
    assert "no observation noise" in varied["reset_protocol"]
    assert len(set(varied["observed_seconds"])) == 2
    assert all(0 < seconds < 20 for seconds in varied["observed_seconds"])
    assert varied["survival_fraction"] == 0
    assert nominal["survived"] == [True, True]
    assert varied["survived"] == [False, False]
    from embodiedforge.robot_replay import RobotMotion

    for path, result in ((evaluation, nominal), (randomized, varied)):
        assert result["survival_fraction"] == np.mean(result["survived"])
        assert len(result["termination_reasons"]) == result["num_envs"]
        clipping = np.array(result["action_clip_fraction_per_env"])
        assert clipping.shape == (2,)
        assert np.isfinite(clipping).all() and ((clipping >= 0) & (clipping <= 1)).all()
        assert result["action_clip_fraction"] == pytest.approx(
            np.average(clipping, weights=result["observed_seconds"])
        )
        for alive, reasons in zip(
            result["survived"], result["termination_reasons"], strict=True
        ):
            assert alive == (not reasons)
            assert set(reasons) <= {"torso_contact", "base_height", "base_tilt"}
            assert len(reasons) == len(set(reasons))
        for metric in ("planar_rmse", "yaw_rmse"):
            per_env = np.array(result[metric + "_per_env"])
            assert per_env.shape == (2,)
            assert np.isfinite(per_env).all() and (per_env >= 0).all()
            assert result[metric] == pytest.approx(
                np.sqrt(np.average(per_env**2, weights=result["observed_seconds"]))
            )
        replay = RobotMotion(
            path / "motion.npz",
            mujoco.MjModel.from_binary_path(str(resumed / "model.mjb")),
        )
        assert replay.max_position_error < 1e-5
        recorded_env = 1 if path == randomized else 0
        assert replay.metadata["environment_index"] == recorded_env
        assert replay.metadata["num_envs"] == result["num_envs"]
        assert replay.metadata["seed"] == result["seed"]
        assert replay.metadata["randomized_reset"] == (path == randomized)
        assert replay.metadata["checkpoint_sha256"] == report["checkpoint_sha256"]
        assert replay.metadata["model_sha256"] == report["model_sha256"]
        assert replay.time[-1] == result["observed_seconds"][recorded_env]
        with np.load(path / "motion.npz", allow_pickle=False) as motion:
            root_height = motion["positions"][-1, 0, 2]
            reasons = result["termination_reasons"][recorded_env]
            assert ("base_height" in reasons) == (root_height < 0.45)
    from embodiedforge.native_h1 import load_run

    with (resumed / "model.mjb").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_run(resumed)
