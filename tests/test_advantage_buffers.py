"""Every GAE output element must be defined even when scratch storage is dirty."""

import numpy as np
import pytest

from embodiedforge.learning import advantages


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 1), (3, 5)])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_numpy_advantages_fully_overwrite_scratch_and_preserve_inputs(
    monkeypatch, shape, dtype, layout
):
    reward = np.zeros(shape, dtype=dtype)
    if layout == "F":
        reward = np.asfortranarray(reward)
    elif layout == "strided":
        reward = np.zeros((shape[0], shape[1] * 2), dtype=dtype)[:, ::2]
    value, following = reward.copy(), reward.copy()
    terminated, truncated = np.zeros(shape, bool), np.zeros(shape, bool)
    for array in (reward, value, following, terminated, truncated):
        array.setflags(write=False)
    allocator = np.empty_like
    allocations = []

    def dirty_scratch(template, *args, **kwargs):
        result = allocator(template, *args, **kwargs)
        result.fill(np.nan)
        allocations.append(result)
        return result

    monkeypatch.setattr(advantages.np, "empty_like", dirty_scratch)
    advantage, returns = advantages.generalized_advantage(
        reward, value, following, terminated, truncated, 0.99, 0.95
    )
    assert len(allocations) == 1
    for result in (advantage, returns):
        assert result.dtype == dtype and result.shape == shape
        np.testing.assert_array_equal(result, np.zeros(shape, dtype=dtype))
        for array in (reward, value, following):
            assert not np.shares_memory(result, array)
    assert not np.shares_memory(advantage, returns)


@pytest.mark.parametrize("shape", [(1, 1), (3, 5)])
@pytest.mark.parametrize("dtype_name", ["float16", "float32", "float64"])
@pytest.mark.parametrize("transposed", [False, True])
def test_go1_advantages_overwrite_scratch_and_keep_gradients(
    monkeypatch, shape, dtype_name, transposed
):
    torch = pytest.importorskip("torch")
    from embodiedforge.locomotion import go1_ppo

    dtype = getattr(torch, dtype_name)
    reward = (
        torch.zeros(tuple(reversed(shape)), dtype=dtype).T
        if transposed
        else torch.zeros(shape, dtype=dtype)
    ).requires_grad_()
    value = torch.zeros_like(reward, requires_grad=True)
    following = torch.zeros(shape[1], dtype=dtype, requires_grad=True)
    batch = {
        "rew": reward,
        "val": value,
        "alive": torch.zeros(shape, dtype=torch.bool),
        "last_val": following,
    }
    allocator = torch.empty_like
    allocations = []

    def dirty_scratch(template, *args, **kwargs):
        result = allocator(template, *args, **kwargs)
        result.fill_(float("nan"))
        allocations.append(result)
        return result

    monkeypatch.setattr(go1_ppo.torch, "empty_like", dirty_scratch)
    advantage, returns = go1_ppo.gae(batch)
    assert len(allocations) == 1
    for result in (advantage, returns):
        assert result.shape == shape and result.dtype == dtype
        assert torch.equal(result, torch.zeros(shape, dtype=dtype))
    (advantage.sum() + returns.sum()).backward()
    assert torch.equal(reward.grad, torch.full(shape, 2, dtype=dtype))
    assert torch.equal(value.grad, torch.full(shape, -1, dtype=dtype))
    assert torch.equal(following.grad, torch.zeros(shape[1], dtype=dtype))
