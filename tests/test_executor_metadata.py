"""Episode/step tracking must preserve exact integer identity through uint64."""

import numpy as np
import pytest

from embodiedforge.policies import ActionExecutor


class PlannedActions:
    def __init__(self, count, horizon):
        self.calls = np.zeros(count, dtype=np.int64)
        self.horizon = horizon

    def act(self, observation):
        ids = observation["env_id"]
        self.calls[ids] += 1
        return (
            (self.calls[ids] % 5).astype(np.float32)[:, None, None] * 0.05
            + np.arange(self.horizon, dtype=np.float32)[None, :, None] * 0.1
            + ids.astype(np.float32)[:, None, None] * 0.001
        )


class IntegerReference:
    """Use Python integers and per-row queues, with no NumPy metadata comparisons."""

    def __init__(self, count):
        self.previous = [None] * count
        self.actions = [[] for _ in range(count)]

    def reset(self, ids):
        for row in ids:
            self.previous[row] = None
            self.actions[row] = []

    def act(self, observation, policy):
        needed = []
        for row in range(len(self.actions)):
            identity = (
                int(observation["episode_id"][row]),
                int(observation["step_id"][row]),
            )
            previous = self.previous[row]
            if previous is not None and (
                identity[0] != previous[0] or identity[1] != previous[1] + 1
            ):
                self.actions[row] = []
            self.previous[row] = identity
            if not self.actions[row]:
                needed.append(row)
        if needed:
            predicted = policy.act(
                {key: value[needed] for key, value in observation.items()}
            )
            for row, chunk in zip(needed, predicted, strict=True):
                self.actions[row] = list(chunk.copy())
        return np.stack([queue.pop(0) for queue in self.actions])


@pytest.mark.parametrize("dtype", ["u8", ">u8", "i8", ">i8", "u4"])
@pytest.mark.parametrize("count", [1, 3, 16])
@pytest.mark.parametrize("horizon", [1, 4])
def test_integer_tracking_matches_python_reference(dtype, count, horizon):
    maximum = int(np.iinfo(dtype).max)
    boundaries = sorted({0, min(2**53 - 1, maximum), min(2**63 - 2, maximum), maximum})
    episode = [boundaries[row % len(boundaries)] for row in range(count)]
    step = [max(0, boundaries[(row + 1) % len(boundaries)] - 2) for row in range(count)]
    actual = ActionExecutor(count, 1, horizon)
    reference = IntegerReference(count)
    policy, expected_policy = (
        PlannedActions(count, horizon),
        PlannedActions(count, horizon),
    )
    rng = np.random.default_rng(42)
    for control in range(80):
        if control:
            for row in range(count):
                mode = int(rng.integers(6))
                if mode == 0:  # Repeated observations invalidate chunks.
                    continue
                step[row] = (step[row] + (2 if mode == 1 else 1)) % (maximum + 1)
                if mode == 2:  # Episode changes matter even if steps continue.
                    episode[row] = (episode[row] + 1) % (maximum + 1)
                elif mode == 3:
                    episode[row] = boundaries[int(rng.integers(len(boundaries)))]
                    step[row] = 0
            if control % 9 == 0:
                selected = [control % count]
                actual.reset(selected)
                reference.reset(selected)
        observation = {
            "env_id": np.arange(count),
            "episode_id": np.asarray(episode, dtype=dtype),
            "step_id": np.asarray(step, dtype=dtype),
        }
        action = actual.act(observation, policy)
        np.testing.assert_array_equal(
            action, reference.act(observation, expected_policy)
        )
        np.testing.assert_array_equal(policy.calls, expected_policy.calls)
        assert action.flags.owndata
        action[:] = -1  # Returned actions cannot alter subsequent cached steps.


@pytest.mark.parametrize("dtype", ["<u8", ">u8"])
@pytest.mark.parametrize("case", ["large_episode", "adjacent_episode", "wrapped_step"])
def test_uint64_boundaries_invalidate_only_stale_chunks(dtype, case):
    executor = ActionExecutor(1, 1, 4)
    policy = PlannedActions(1, 4)
    if case == "large_episode":
        identities = [(2**63 + 1, step) for step in range(3)]
        expected_calls = [1, 1, 1]
    elif case == "adjacent_episode":
        identities = [(2**53, 0), (2**53 + 1, 1), (2**53 + 1, 2)]
        expected_calls = [1, 2, 2]
    else:
        identities = [(1, 2**64 - 1), (1, 0), (1, 1)]
        expected_calls = [1, 2, 2]
    for (episode, step), calls in zip(identities, expected_calls, strict=True):
        executor.act(
            {
                "env_id": np.array([0]),
                "episode_id": np.array([episode], dtype=dtype),
                "step_id": np.array([step], dtype=dtype),
            },
            policy,
        )
        assert policy.calls.tolist() == [calls]
