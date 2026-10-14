"""Exercise chunk execution, partial resets and recording through one runner."""

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes
from embodiedforge.rollout import DemonstrationPolicy, run_episode_batch, run_rollout
from embodiedforge.tasks import ReachTask


class StaggeredTask(ReachTask):
    """End row zero every two steps and row one every three steps."""

    def evaluate(self, state, actions, previous):
        done = state.time >= np.array([0.04, 0.06])
        return np.ones(2, dtype=np.float32), done, done.copy()


class ChunkPolicy:
    def __init__(self):
        self.calls = []
        self.resets = []
        self.needs_reset = {0, 1}

    def reset(self, ids):
        self.resets.append(ids.tolist())
        self.needs_reset.difference_update(ids.tolist())

    def act(self, observation):
        ids = observation["env_id"].tolist()
        assert not self.needs_reset.intersection(ids)
        self.calls.append(ids)
        return np.broadcast_to(
            np.arange(1, 5, dtype=np.float32)[None, :, None] / 10,
            (len(ids), 4, 2),
        )


class RetainingWriter:
    """Retain results to detect mutation by the runner after append returns."""

    def __init__(self, policy):
        self.results = []
        self.policy = policy

    def append(self, observation, result):
        self.results.append(result)
        self.policy.needs_reset.update(
            np.flatnonzero(result.terminated | result.truncated).tolist()
        )


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("channels", [(), ("rgb", "depth", "instance_id")])
def test_full_reset_keeps_retained_terminal_frames_and_owned_policy_inputs(
    num_envs, channels
):
    class TimeoutTask(ReachTask):
        def evaluate(self, state, actions, previous):
            return (
                np.ones(num_envs, dtype=np.float32),
                np.zeros(num_envs, dtype=bool),
                np.zeros(num_envs, dtype=bool),
            )

    class Writer:
        def __init__(self):
            self.inputs, self.results, self.copies = [], [], []

        def append(self, observation, result):
            self.inputs.append(observation)
            self.results.append(result)
            self.copies.append({k: v.copy() for k, v in result.observation.items()})

    class ZeroPolicy:
        def act(self, observation):
            return np.zeros((len(observation["env_id"]), 1, 2), dtype=np.float32)

    writer = Writer()
    resets = []
    with VectorEnv(
        Config(num_envs=num_envs, max_steps=1, render="raster", channels=channels),
        task=TimeoutTask(),
    ) as env:
        summary = run_rollout(
            env, ZeroPolicy(), steps=4, writer=writer, on_reset=resets.append
        )
        assert summary.completed_episodes == 4 * num_envs
        assert len(resets) == 4
        for ids in resets:
            np.testing.assert_array_equal(ids, np.arange(num_envs))
        for index, (observation, result, expected) in enumerate(
            zip(writer.inputs, writer.results, writer.copies, strict=True)
        ):
            np.testing.assert_array_equal(observation["episode_id"], index + 1)
            np.testing.assert_array_equal(observation["step_id"], 0)
            np.testing.assert_array_equal(result.observation["step_id"], 1)
            for key, value in observation.items():
                assert value.flags.owndata
                assert not np.shares_memory(value, result.observation[key])
                np.testing.assert_array_equal(result.observation[key], expected[key])
        # Consumers can mutate a retained input without changing other episodes
        # or the terminal observations held by the writer.
        for value in writer.inputs[0].values():
            value[...] = 0
        for result, expected in zip(writer.results, writer.copies, strict=True):
            for key in expected:
                np.testing.assert_array_equal(result.observation[key], expected[key])


@pytest.mark.parametrize(
    "layout", ["dtype", "broadcast", "extra_reset_key", "key_order"]
)
def test_full_reset_preserves_replacement_behavior_for_dynamic_extra_fields(layout):
    class DynamicTask(ReachTask):
        def observe(self, state):
            observation = super().observe(state)
            resetting = not state.time.any()
            shape = (3, 1 if resetting and layout == "broadcast" else 2)
            dtype = np.float32 if resetting and layout == "dtype" else np.float64
            observation["extra"] = np.full(shape, 9.9, dtype=dtype)
            if resetting and layout == "extra_reset_key":
                observation["reset_only"] = np.ones((3, 1))
            if layout == "key_order":
                names = ("first", "second") if resetting else ("second", "first")
                observation.update({name: np.ones((3, 1)) for name in names})
            return observation

    class Policy:
        def __init__(self):
            self.inputs = []

        def act(self, observation):
            self.inputs.append(observation)
            return np.zeros((3, 1, 2), dtype=np.float32)

    policy = Policy()
    with VectorEnv(Config(num_envs=3, max_steps=1), task=DynamicTask()) as env:
        run_rollout(env, policy, steps=3)
    assert policy.inputs[0]["extra"].dtype == (
        np.float32 if layout == "dtype" else np.float64
    )
    for observation in policy.inputs[1:]:
        # Replacement into the terminal layout preserves its dtype, broadcasts
        # compatible reset shapes and ignores additional reset-only fields.
        assert observation["extra"].dtype == np.float64
        assert observation["extra"].shape == (3, 2)
        expected = np.float64(np.float32(9.9)) if layout == "dtype" else 9.9
        np.testing.assert_array_equal(observation["extra"], expected)
        assert "reset_only" not in observation
        if layout == "key_order":
            assert tuple(observation)[-2:] == ("second", "first")


def test_full_reset_keeps_missing_extra_field_error_before_next_inference():
    class DynamicTask(ReachTask):
        def observe(self, state):
            observation = super().observe(state)
            if state.time.any():
                observation["terminal_only"] = np.ones((3, 1))
            return observation

    class Policy:
        calls = 0

        def act(self, observation):
            self.calls += 1
            return np.zeros((3, 1, 2), dtype=np.float32)

    policy = Policy()
    with VectorEnv(Config(num_envs=3, max_steps=1), task=DynamicTask()) as env:
        with pytest.raises(KeyError, match="terminal_only"):
            run_rollout(env, policy, steps=2)
        assert policy.calls == 1


def test_chunk_runner_replans_and_resets_only_finished_rows():
    policy = ChunkPolicy()
    writer = RetainingWriter(policy)
    with VectorEnv(Config(num_envs=2), task=StaggeredTask()) as env:
        summary = run_rollout(
            env,
            policy,
            steps=5,
            chunk_horizon=4,
            writer=writer,
            on_reset=policy.reset,
        )
        assert summary.completed_episodes == 3
        assert summary.success_rate == 1
        assert summary.mean_step_reward == 1
    assert policy.calls == [[0, 1], [0], [1], [0]]
    assert policy.resets == policy.calls
    actions = np.stack([r.info["executed_action"][:, 0] for r in writer.results])
    np.testing.assert_allclose(
        actions, [[0.1, 0.1], [0.2, 0.2], [0.1, 0.3], [0.2, 0.1], [0.1, 0.2]]
    )
    # Terminal frames stay terminal even after replacement rows are prepared.
    np.testing.assert_array_equal(writer.results[1].observation["step_id"], [2, 2])
    np.testing.assert_array_equal(writer.results[2].observation["step_id"], [1, 3])


def test_runner_records_complete_episodes_and_leaves_final_done_frozen(tmp_path):
    class ZeroPolicy:
        def act(self, observation):
            return np.zeros((len(observation["env_id"]), 2, 2), dtype=np.float32)

    with VectorEnv(Config(num_envs=2, max_steps=3)) as env:
        with EpisodeRecorder(tmp_path / "run", env) as writer:
            summary = run_rollout(
                env, ZeroPolicy(), steps=6, chunk_horizon=2, writer=writer
            )
        assert summary.completed_episodes == 4
        assert env.truncated.all()
        np.testing.assert_array_equal(env.observe()["step_id"], [3, 3])
    episodes = list(read_episodes(tmp_path / "run"))
    assert len(episodes) == 4
    for episode in episodes:
        np.testing.assert_array_equal(episode["obs/step_id"], [0, 1, 2])
        np.testing.assert_array_equal(episode["next/step_id"], [1, 2, 3])


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": 1, "chunk_horizon": 0}])
def test_invalid_budget_does_not_reset_environment(kwargs):
    with VectorEnv(Config(num_envs=1)) as env:
        before = env.observe()["episode_id"].copy()
        with pytest.raises(ValueError):
            run_rollout(env, DemonstrationPolicy(env.task.expert_action), **kwargs)
        np.testing.assert_array_equal(env.observe()["episode_id"], before)


def test_reset_callback_failure_prevents_inference_and_stepping():
    class NeverPolicy:
        def act(self, observation):
            pytest.fail("must not infer after reset callback failure")

    def fail(ids):
        raise RuntimeError("policy history reset failed")

    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(RuntimeError, match="history reset failed"):
            run_rollout(env, NeverPolicy(), steps=2, on_reset=fail)
        assert env.observe()["step_id"].tolist() == [0]


def test_episode_batch_freezes_early_finishes_and_excludes_them_from_inference():
    class OneStepPolicy:
        def __init__(self):
            self.calls = []

        def act(self, observation):
            self.calls.append(observation["env_id"].tolist())
            return np.zeros((len(observation["env_id"]), 1, 2), dtype=np.float32)

    initial_hashes = []
    for _ in range(2):
        policy = OneStepPolicy()
        resets = []
        with VectorEnv(Config(num_envs=2), task=StaggeredTask()) as env:
            batch = run_episode_batch(
                env,
                policy,
                on_reset=lambda ids, resets=resets: resets.append(ids.tolist()),
            )
            assert batch.lengths.tolist() == [2, 3]
            assert batch.returns.tolist() == [2.0, 3.0]
            assert batch.success.tolist() == [True, True]
            assert env.observe()["episode_id"].tolist() == [0, 0]
            assert env.observe()["step_id"].tolist() == [2, 3]
        assert policy.calls == [[0, 1], [0, 1], [1]]
        assert resets == [[0, 1]]
        initial_hashes.append(batch.initial_observation_sha256)
    assert initial_hashes[0] == initial_hashes[1]


def test_episode_batch_keeps_chunk_order_and_counts_timeouts():
    policy = ChunkPolicy()
    with VectorEnv(Config(num_envs=2, max_steps=3)) as env:
        batch = run_episode_batch(env, policy, chunk_horizon=4, on_reset=policy.reset)
        assert batch.lengths.tolist() == [3, 3]
        assert not batch.success.any()
        assert env.truncated.all()
    assert policy.calls == [[0, 1]]


def test_episode_batch_rejects_broadcasting_policy_rows():
    class WrongRows:
        def act(self, observation):
            return np.zeros((1, 1, 2), dtype=np.float32)

    with VectorEnv(Config(num_envs=2)) as env:
        with pytest.raises(ValueError, match="active_rows"):
            run_episode_batch(env, WrongRows())
