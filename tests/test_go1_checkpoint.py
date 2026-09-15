"""Invalid Go1 checkpoints must fail consistently before simulation starts."""

from types import SimpleNamespace

import pytest

from embodiedforge._go1_checkpoint import load_go1_checkpoint
from embodiedforge.recipes import sha256

torch = pytest.importorskip("torch")


@pytest.fixture
def saved(tmp_path):
    path = tmp_path / "model.pt"
    checkpoint = {
        "iteration": 0,
        "model_state_dict": {
            "mean": torch.zeros(50),
            "var": torch.ones(50),
            "count": torch.tensor(1e-4),
        },
    }
    return path, checkpoint, {"checkpoint_iteration": 0}


def test_legacy_defaults_and_zero_variance_remain_valid(saved):
    path, checkpoint, contract = saved
    checkpoint["model_state_dict"]["var"].zero_()
    torch.save(checkpoint, path)
    loaded = load_go1_checkpoint(path, sha256=sha256(path), contract=contract)
    assert loaded["iteration"] == 0
    assert torch.equal(loaded["model_state_dict"]["var"], torch.zeros(50))


def test_mismatched_hash_never_deserializes(saved, monkeypatch):
    path, checkpoint, contract = saved
    torch.save(checkpoint, path)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Deserialized a checkpoint with mismatched hash")

    monkeypatch.setattr(torch, "load", unexpected_load)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_go1_checkpoint(path, sha256="wrong", contract=contract)


@pytest.mark.parametrize("entry", ["resume", "evaluate", "live"])
@pytest.mark.parametrize(
    "damage,error",
    [
        ("negative_variance", "normalization statistics"),
        ("zero_count", "normalization statistics"),
        ("negative_count", "normalization statistics"),
        ("vector_count", "normalization statistics"),
        ("integer_count", "normalization statistics"),
        ("nan_mean", "Non-finite checkpoint tensor"),
        ("wrong_iteration", "iteration differs"),
        ("boolean_iteration", "iteration differs"),
        ("float_iteration", "iteration differs"),
        ("wrong_recipe", "recipe differs"),
        ("changed_profile", "reward_profile differs"),
    ],
)
def test_all_entrypoints_reject_invalid_state_before_simulation(
    saved, monkeypatch, entry, damage, error
):
    pytest.importorskip("mjbatch")
    pytest.importorskip("mujoco_menagerie")
    from embodiedforge import _go1_assets, _mjbatch_recipe
    from embodiedforge._go1_live_worker import Go1LiveRuntime

    path, checkpoint, contract = saved
    state = checkpoint["model_state_dict"]
    if damage == "negative_variance":
        state["var"][0] = -1
    elif damage == "zero_count":
        state["count"].zero_()
    elif damage == "negative_count":
        state["count"].fill_(-1)
    elif damage == "vector_count":
        state["count"] = torch.ones(2)
    elif damage == "integer_count":
        state["count"] = torch.tensor(1)
    elif damage == "nan_mean":
        state["mean"][0] = float("nan")
    elif damage == "wrong_iteration":
        checkpoint["iteration"] = 1
    elif damage == "boolean_iteration":
        checkpoint["iteration"] = False
    elif damage == "float_iteration":
        checkpoint["iteration"] = 0.0
    elif damage == "wrong_recipe":
        checkpoint["recipe"] = "another-task"
    else:
        checkpoint["reward_profile"] = "different"
    torch.save(checkpoint, path)
    digest = sha256(path)  # Valid hash must not mask invalid checkpoint contents.
    request = {
        "checkpoint": str(path),
        "input_checkpoint_sha256": digest,
        "input_checkpoint_iteration": 0,
        "input_reward_profile": "original",
        "input_command_profile": "original",
        "input_task_semantics": "upstream-v1",
        "input_learning_rate_override": None,
        "threads": 1,
        "seed": 0,
        "sha256": digest,
        "contract": contract,
    }

    def unexpected_simulation(*args, **kwargs):
        pytest.fail("Invalid checkpoint reached asset loading or simulation")

    monkeypatch.setattr(_go1_assets, "verified_go1_assets", unexpected_simulation)
    owner = SimpleNamespace(Go1=unexpected_simulation)
    with pytest.raises(ValueError, match=error):
        if entry == "resume":
            _mjbatch_recipe.go1_train(request, owner)
        elif entry == "evaluate":
            _mjbatch_recipe.go1_evaluate(request, owner, SimpleNamespace())
        else:
            Go1LiveRuntime(request)
