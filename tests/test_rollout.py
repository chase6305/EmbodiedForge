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
