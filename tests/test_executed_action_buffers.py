"""Uniform activity shortcuts retain executed-action bits and ownership."""

import warnings

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.env import _executed_action


@pytest.mark.parametrize(
    "count,dim", [(0, 2), (1, 2), (1023, 2), (1024, 2), (8192, 2), (171, 12)]
)
@pytest.mark.parametrize("layout", ["C", "F", "reverse"])
@pytest.mark.parametrize("activity", ["all", "none", "mixed"])
@pytest.mark.parametrize("dtype", ["float64", "float32", ">f8"])
def test_executed_action_bits_layout_and_ownership(count, dim, layout, activity, dtype):
    rng = np.random.default_rng(192)
    action = rng.uniform(-1, 1, (count, dim)).astype(dtype)
    action[::3, 0] = -0.0
    if layout == "F":
        action = np.asfortranarray(action)
    elif layout == "reverse":
        action = action[::-1, ::-1]
    active = np.ones(count, bool) if activity == "all" else np.zeros(count, bool)
    if activity == "mixed":
        active[::3] = True
    before = action.tobytes(), active.tobytes()
    action.flags.writeable = active.flags.writeable = False
    expected = np.where(active[:, None], action, 0).astype(np.float32)
    output = _executed_action(action, active)
    assert type(output) is type(expected)
    assert output.dtype == expected.dtype == np.float32
    assert output.shape == expected.shape and output.strides == expected.strides
    assert output.tobytes() == expected.tobytes()
    assert output.flags.owndata
    assert not np.shares_memory(output, action)
    assert not np.shares_memory(output, active)
    output[:] = 99
    assert _executed_action(action, active).tobytes() == expected.tobytes()
    assert before == (action.tobytes(), active.tobytes())


@pytest.mark.parametrize("activity", ["all", "none", "mixed"])
@pytest.mark.parametrize("mode", ["ignore", "warn", "raise"])
@pytest.mark.parametrize("layout", ["C", "F"])
def test_executed_action_preserves_fp_diagnostics(activity, mode, layout):
    bits = np.array([0, 2**63, 1, 0x7FEFFFFFFFFFFFFF, 0x7FF0000000000081], np.uint64)
    action = np.resize(bits, (1024, 2)).view(np.float64)
    if layout == "F":
        action = np.asfortranarray(action)
    active = np.full(1024, activity == "all", bool)
    if activity == "mixed":
        active[::3] = True

    def evaluate(current):
        with warnings.catch_warnings(record=True) as records, np.errstate(all=mode):
            warnings.simplefilter("always")
            try:
                output = (
                    _executed_action(action, active)
                    if current
                    else np.where(active[:, None], action, 0).astype(np.float32)
                )
                result = ("passed", output.tobytes())
            except FloatingPointError as error:
                result = ("error", str(error))
            return result, [(w.category, str(w.message)) for w in records]

    assert evaluate(True) == evaluate(False)


def test_executed_actions_survive_later_steps_and_partial_reset():
    with VectorEnv(Config(num_envs=1024, task="hold", max_steps=100)) as env:
        action = np.full((1024, 2), -0.25)
        action.flags.writeable = False
        first = env.step(action)
        output = first.info["executed_action"]
        before = output.tobytes()
        env.terminated[:] = True
        frozen = env.step(action)
        assert (
            frozen.info["executed_action"].tobytes()
            == np.zeros((1024, 2), np.float32).tobytes()
        )
        assert output.tobytes() == before
        output[:] = 99
        env.reset([3, 1])
        following = env.step(action)
        expected = np.zeros((1024, 2), np.float32)
        expected[[3, 1]] = -0.25
        assert following.info["executed_action"].tobytes() == expected.tobytes()
        assert np.all(output == 99)


@pytest.mark.parametrize("subclass_field", ["actions", "active"])
def test_executed_action_retains_subclass_dispatch(subclass_field):
    class Borrowed(np.ndarray):
        @property
        def size(self):
            raise RuntimeError("Do not inspect subclass size")

        def all(self, *args, **kwargs):
            raise RuntimeError("Do not reduce subclass masks")

        def any(self, *args, **kwargs):
            raise RuntimeError("Do not reduce subclass masks")

    action = np.ones((1024, 2))
    active = np.ones(1024, bool)
    if subclass_field == "actions":
        action = action.view(Borrowed)
    else:
        active = active.view(Borrowed)
    expected = np.where(active[:, None], action, 0).astype(np.float32)
    assert _executed_action(action, active).tobytes() == expected.tobytes()
