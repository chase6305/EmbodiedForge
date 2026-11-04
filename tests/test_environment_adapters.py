"""Evaluate native tasks through the same pre-reset transition contract."""

import json

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.envs.evaluation import evaluate_policy
from embodiedforge.envs.locomotion import Go1Environment, H1Environment


class NativeFixture:
    """Native-shaped tasks with observation sampling and staggered episode ends."""

    def __init__(self, kind):
        self.kind = kind
        self.steps = np.zeros(2, dtype=np.int64)
        self.samples = 0
        self.calls = 0
        self.action_dim, self.obs_dim = (12, 50) if kind == "go1" else (19, 69)
        self.last_action = None

    def obs(self):
        self.samples += 1
        return np.broadcast_to(self.steps[:, None], (2, self.obs_dim)).astype(
            np.float32
        )

    def reset(self, ids):
        self.steps[ids] = 0
        if self.kind == "h1":
            return self.obs()

    def step(self, actions):
        self.calls += 1
        self.last_action = actions.copy()
        self.steps += 1
        reward = self.steps.astype(np.float32)
        fell = (self.steps >= 2) & (np.arange(2) == 0)
        timeout = (self.steps >= 3) & ~fell
        terms = {"tracking": reward}
        if self.kind == "go1":
            return reward, fell | timeout, fell, terms
        return self.obs(), reward, fell, timeout, terms


@pytest.fixture(params=["go1", "h1"])
def adapted(request):
    native = NativeFixture(request.param)
    adapter = (
        Go1Environment(native) if request.param == "go1" else H1Environment(native)
    )
    return native, adapter


def test_observe_is_owned_and_does_not_resample_native_noise(adapted):
    native, env = adapted
    first = env.observe()
    samples = native.samples
    second = env.observe()
    assert native.samples == samples
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
        assert not np.shares_memory(first[name], second[name])
    first["proprio"][:] = 999
    assert not env.observe()["proprio"].any()
    spec = env.spec
    spec["action_shape"][0] = 999
    assert env.spec["action_shape"] == [native.action_dim]


@pytest.mark.parametrize("ids", [None, [], [1, 0], [1]])
def test_public_observations_own_all_fields_and_leave_caller_ids_untouched(
    adapted, ids
):
    native, env = adapted
    supplied = None if ids is None else np.asarray(ids, dtype=np.int64)
    original_ids = None if supplied is None else supplied.copy()
    result = env.reset(supplied)
    expected = env.observe()
    for name, value in result.items():
        assert (
            value.flags.owndata and value.flags.writeable and value.flags.c_contiguous
        )
        for other_name, other in result.items():
            if name != other_name:
                assert not np.shares_memory(value, other)
        value.fill(99)
    if supplied is not None:
        np.testing.assert_array_equal(supplied, original_ids)
    for name, value in env.observe().items():
        np.testing.assert_array_equal(value, expected[name])
    assert native.calls == 0


@pytest.mark.parametrize("kind", ["go1", "h1"])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_cache_and_results_detach_reused_native_features(kind, layout):
    native = NativeFixture(kind)
    values = np.zeros((2, native.obs_dim), dtype=np.float32)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = np.zeros((2, native.obs_dim * 2), dtype=np.float32)[:, ::2]

    def observation():
        native.samples += 1
        values[:] = native.steps[:, None]
        return values

    native.obs = observation
    env = Go1Environment(native) if kind == "go1" else H1Environment(native)
    values.fill(77)
    assert not env.observe()["proprio"].any()
    actions = np.zeros((2, native.action_dim), dtype=np.float32)
    env.step(actions)
    terminal = env.step(actions)
    saved = {key: value.copy() for key, value in terminal.observation.items()}
    values.fill(77)
    np.testing.assert_array_equal(env.observe()["proprio"], saved["proprio"])
    selected = env.reset([0])
    expected = env.observe()
    assert not selected["proprio"].any()
    assert (expected["proprio"][1] == 2).all()
    values.fill(77)
    selected["proprio"].fill(99)
    for name, value in env.observe().items():
        np.testing.assert_array_equal(value, expected[name])
        np.testing.assert_array_equal(terminal.observation[name], saved[name])


def test_partial_reset_validates_unselected_native_features_before_publication(
    adapted, monkeypatch
):
    native, env = adapted
    before = env.observe()["proprio"]

    def invalid_observation():
        values = np.zeros((2, native.obs_dim), dtype=np.float32)
        values[1, 0] = np.nan
        return values

    monkeypatch.setattr(native, "obs", invalid_observation)
    with pytest.raises(ValueError, match="Native observation") as error:
        env.reset([0])
    assert env.failure is error.value
    np.testing.assert_array_equal(env._proprio, before)
    with pytest.raises(RuntimeError, match="adapter failed"):
        env.observe()


def test_partial_reset_preserves_row_order_other_rows_and_terminal_results(adapted):
    native, env = adapted
    action = np.zeros((2, native.action_dim), dtype=np.float32)
    env.step(action)
    result = env.step(action)
    terminal = {key: value.copy() for key, value in result.observation.items()}
    assert result.terminated.tolist() == [True, False]
    with pytest.raises(RuntimeError, match="Reset all completed"):
        env.step(action)
    assert native.calls == 2
    selected = env.reset([0])
    assert selected["env_id"].tolist() == [0]
    assert selected["step_id"].tolist() == [0]
    observed = env.observe()
    assert observed["step_id"].tolist() == [0, 2]
    for key, value in terminal.items():
        np.testing.assert_array_equal(result.observation[key], value)
        np.testing.assert_array_equal(observed[key][1], value[1])
    result = env.step(action)
    assert result.truncated.tolist() == [False, True]
    selected = env.reset([1, 0])
    assert selected["env_id"].tolist() == [1, 0]
    assert selected["episode_id"].tolist() == [1, 2]


@pytest.mark.parametrize("ids", [[0, 0], [-1], [2], [0.5], [[0]], [True]])
def test_invalid_reset_is_rejected_before_native_mutation(adapted, ids):
    native, env = adapted
    before, samples = native.steps.copy(), native.samples
    with pytest.raises(ValueError, match="env_ids"):
        env.reset(ids)
    np.testing.assert_array_equal(native.steps, before)
    assert native.samples == samples
    assert env.failure is None


def test_empty_reset_does_not_sample_or_change_identity(adapted):
    native, env = adapted
    samples = native.samples
    selected = env.reset([])
    assert native.samples == samples
    assert all(len(value) == 0 for value in selected.values())
    assert selected["proprio"].shape == (0, native.obs_dim)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, "bad", complex(1, 2), True])
def test_invalid_actions_do_not_advance_the_native_task(adapted, invalid):
    native, env = adapted
    with pytest.raises(ValueError, match="Actions"):
        env.step(np.full((2, native.action_dim), invalid))
    assert native.calls == 0
    assert env.failure is None


def test_action_semantics_preserve_go1_unbounded_output_and_h1_clipping(adapted):
    native, env = adapted
    result = env.step(np.full((2, native.action_dim), 8.0))
    expected = 8.0 if native.kind == "go1" else 5.0
    assert (native.last_action == expected).all()
    assert (result.info["executed_action"] == expected).all()
    assert env.spec["action_range"] == (None if native.kind == "go1" else [-5, 5])


def test_native_error_poisons_adapter_and_preserves_cause(adapted, monkeypatch):
    native, env = adapted
    failure = RuntimeError("native step failed")

    def fail(actions):
        raise failure

    monkeypatch.setattr(native, "step", fail)
    with pytest.raises(RuntimeError, match="native step failed"):
        env.step(np.zeros((2, native.action_dim)))
    for operation in (env.observe, env.reset):
        with pytest.raises(RuntimeError, match="adapter failed") as error:
            operation()
        assert error.value.__cause__ is failure


@pytest.mark.parametrize("reward", [1e300, -1e300])
def test_reward_cast_cannot_publish_an_infinite_transition(
    adapted, monkeypatch, reward
):
    native, env = adapted
    transition = env._transition

    def extreme(actions):
        observation, _, terminated, truncated, terms = transition(actions)
        return observation, np.full(2, reward), terminated, truncated, terms

    monkeypatch.setattr(env, "_transition", extreme)
    with pytest.raises(ValueError, match="representable as float32"):
        env.step(np.zeros((2, native.action_dim)))
    assert env.failure is not None
    with pytest.raises(RuntimeError, match="adapter failed"):
        env.observe()


def test_conflicting_native_terminal_flags_poison_the_adapter(adapted, monkeypatch):
    native, env = adapted
    transition = env._transition

    def conflicting(actions):
        observation, reward, _, _, terms = transition(actions)
        return observation, reward, np.ones(2, bool), np.ones(2, bool), terms

    monkeypatch.setattr(env, "_transition", conflicting)
    with pytest.raises(ValueError, match="terminated and truncated"):
        env.step(np.zeros((2, native.action_dim)))
    assert env.failure is not None


def test_shared_evaluator_records_terminal_transitions_before_reset(adapted):
    native, env = adapted
    retained, resets = [], []

    class Sink:
        def append(self, observation, result):
            retained.append((observation, result))

    summary = evaluate_policy(
        env,
        lambda observation: np.zeros((2, native.action_dim)),
        steps=6,
        writer=Sink(),
        on_reset=lambda ids: resets.append(ids.tolist()),
    )
    assert summary.transitions == 12
    assert summary.completed_episodes == 5
    assert summary.terminated_episodes == 3
    assert summary.truncated_episodes == 2
    assert summary.mean_episode_return == pytest.approx(21 / 5)
    assert summary.mean_step_reward == pytest.approx(21 / 12)
    assert summary.success_rate is None
    assert resets == [[0, 1], [0], [1], [0]]
    assert retained[1][1].observation["step_id"].tolist() == [2, 2]
    assert retained[2][0]["step_id"].tolist() == [0, 2]
    assert retained[-1][1].observation["step_id"].tolist() == [2, 3]


def test_same_evaluator_accepts_core_vector_env_and_existing_episode_recorder(tmp_path):
    from embodiedforge.data import EpisodeRecorder, read_episodes

    with VectorEnv(Config(num_envs=2, max_steps=2)) as env:
        with EpisodeRecorder(tmp_path / "episodes", env) as writer:
            summary = evaluate_policy(
                env, lambda obs: np.zeros((2, 2)), steps=5, writer=writer
            )
    assert summary.transitions == 10
    assert summary.completed_episodes == 4
    assert summary.truncated_episodes == 4
    assert len(list(read_episodes(tmp_path / "episodes"))) == 4
    manifest = json.loads((tmp_path / "episodes/run.json").read_text())
    assert manifest["run_contract"]["workflow"] == "core-data"


@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("write_workers", [0, 1])
def test_native_evaluation_records_readable_episodes_and_action_windows(
    adapted, tmp_path, episode_format, write_workers
):
    from embodiedforge.data import (
        EpisodeRecorder,
        RecordingMetadata,
        action_windows,
        read_episodes,
    )
    from embodiedforge.runtime.records import read_run

    native, env = adapted
    metadata = RecordingMetadata(
        family=f"fixture-{native.kind}",
        max_steps=3,
        seed=7,
        instruction="Follow the recorded velocity command",
        scene="fixture-locomotion:v1",
        scene_spec={"kind": "fixture"},
        config={"backend": "fixture", "episode_steps": 3},
        source={"test_fixture": True},
    )
    root = tmp_path / "episodes"
    with EpisodeRecorder(
        root,
        env,
        metadata=metadata,
        episode_format=episode_format,
        write_workers=write_workers,
    ) as writer:
        metadata.config["backend"] = "changed after construction"
        summary = evaluate_policy(
            env, lambda obs: np.zeros((2, native.action_dim)), steps=6, writer=writer
        )
    episodes = list(read_episodes(root))
    assert len(episodes) == summary.completed_episodes == 5
    assert sorted(len(episode["action"]) for episode in episodes) == [2, 2, 2, 3, 3]
    for episode in episodes:
        assert episode["action"].shape[1:] == (native.action_dim,)
        assert episode["obs/proprio"].shape[1:] == (native.obs_dim,)
        assert not (episode["terminated"] | episode["truncated"])[:-1].any()
        assert (episode["terminated"] | episode["truncated"])[-1]
    windows = list(action_windows(root, history=2, horizon=1))
    assert len(windows) == 7
    assert all(sample["instruction"] == metadata.instruction for sample in windows)
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["config"]["backend"] == "fixture"
    assert manifest["source"] == {"test_fixture": True}
    status = read_run(root)
    assert status["workflow"] == "episode-data"
    assert status["status"] == "complete"
    assert status["recorded_status"] == "closed"


def test_native_recording_requires_explicit_provenance_before_creating_output(
    adapted, tmp_path
):
    from embodiedforge.data import EpisodeRecorder

    _, env = adapted
    with pytest.raises(ValueError, match="explicit RecordingMetadata"):
        EpisodeRecorder(tmp_path / "episodes", env)
    assert not (tmp_path / "episodes").exists()


def test_non_core_dataset_cannot_be_silently_reconstructed_as_a_core_act_task(
    adapted, tmp_path, monkeypatch
):
    from embodiedforge import act
    from embodiedforge.data import EpisodeRecorder, RecordingMetadata

    native, env = adapted
    metadata = RecordingMetadata(
        family=native.kind,
        max_steps=3,
        seed=0,
        instruction="Follow velocity",
        scene="test:v1",
        scene_spec={},
        config={},
    )
    output = tmp_path / "episodes"
    with EpisodeRecorder(output, env, metadata=metadata) as writer:
        evaluate_policy(
            env, lambda obs: np.zeros((2, native.action_dim)), steps=3, writer=writer
        )

    def no_sdk():
        pytest.fail("unsupported dataset must be rejected before importing LeRobot")

    monkeypatch.setattr(act, "_require_lerobot", no_sdk)
    with pytest.raises(ValueError, match="core environment recording"):
        act.export_dataset(output, tmp_path / "lerobot", "local/native")
    assert not (tmp_path / "lerobot").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "reward_shape",
        "reward_nan",
        "reward_bool",
        "done_dtype",
        "conflicting_done",
        "success_shape",
        "success_none",
    ],
)
def test_shared_evaluator_rejects_malformed_results_before_calling_a_sink(
    monkeypatch, damage
):
    with VectorEnv(Config(num_envs=2)) as env:
        step = env.step
        written = []

        def malformed(actions):
            result = step(actions)
            if damage == "reward_shape":
                result.reward = result.reward[:1]
            elif damage == "reward_nan":
                result.reward[0] = np.nan
            elif damage == "reward_bool":
                result.reward = np.ones(2, dtype=bool)
            elif damage == "done_dtype":
                result.terminated = result.terminated.astype(np.int32)
            elif damage == "conflicting_done":
                result.terminated[:] = result.truncated[:] = True
            elif damage == "success_shape":
                result.info["success"] = np.ones(1, dtype=bool)
            else:
                result.info["success"] = None
            return result

        class Sink:
            def append(self, observation, result):
                written.append(result)

        monkeypatch.setattr(env, "step", malformed)
        with pytest.raises(ValueError, match="Evaluation"):
            evaluate_policy(env, lambda obs: np.zeros((2, 2)), steps=2, writer=Sink())
        assert not written


def test_shared_evaluator_does_not_retry_a_failed_storage_side_effect(adapted):
    native, env = adapted
    calls, resets = [], []
    failure = OSError("storage failed")

    class Sink:
        def append(self, observation, result):
            calls.append(result)
            raise failure

    with pytest.raises(OSError) as caught:
        evaluate_policy(
            env,
            lambda obs: np.zeros((2, native.action_dim)),
            steps=3,
            writer=Sink(),
            on_reset=lambda ids: resets.append(ids.tolist()),
        )
    assert caught.value is failure
    assert len(calls) == native.calls == 1
    assert resets == [[0, 1]]
