"""Reject damaged Adam state and preserve the last checkpoint on save failure."""

from copy import deepcopy
from pathlib import Path

import pytest

from embodiedforge._go1_checkpoint import restore_go1_optimizer, save_go1_checkpoint

torch = pytest.importorskip("torch")


def new_optimizer():
    parameters = [
        torch.nn.Parameter(torch.ones(2)),
        torch.nn.Parameter(torch.ones(2, 3)),
    ]
    return torch.optim.Adam(parameters, lr=0.001)


def step(optimizer):
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.full_like(parameter, 0.2)
    optimizer.step()


@pytest.fixture
def trained():
    optimizer = new_optimizer()
    step(optimizer)
    return optimizer


def test_restored_adam_matches_the_next_uninterrupted_update(trained):
    resumed = new_optimizer()
    for source, target in zip(
        trained.param_groups[0]["params"],
        resumed.param_groups[0]["params"],
        strict=True,
    ):
        target.data.copy_(source.data)
    restore_go1_optimizer(resumed, deepcopy(trained.state_dict()))
    step(trained)
    step(resumed)
    for source, target in zip(
        trained.param_groups[0]["params"],
        resumed.param_groups[0]["params"],
        strict=True,
    ):
        torch.testing.assert_close(source, target, atol=0, rtol=0)
        for name in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                trained.state[source][name], resumed.state[target][name], atol=0, rtol=0
            )


@pytest.mark.parametrize(
    "damage",
    [
        "negative_second_moment",
        "nonfinite_moment",
        "moment_shape",
        "moment_dtype",
        "negative_step",
        "fractional_step",
        "vector_step",
        "missing_buffer",
        "missing_parameter",
        "extra_parameter",
        "duplicate_id",
        "parameter_count",
        "nan_lr",
        "bad_betas",
        "negative_eps",
        "unsupported_mode",
    ],
)
def test_invalid_state_never_mutates_the_target_optimizer(trained, damage):
    state = deepcopy(trained.state_dict())
    group = state["param_groups"][0]
    slot = state["state"][group["params"][0]]
    if damage == "negative_second_moment":
        slot["exp_avg_sq"].fill_(-1)
    elif damage == "nonfinite_moment":
        slot["exp_avg"].fill_(float("inf"))
    elif damage == "moment_shape":
        slot["exp_avg"] = torch.zeros(3)
    elif damage == "moment_dtype":
        slot["exp_avg"] = slot["exp_avg"].double()
    elif damage == "negative_step":
        slot["step"].fill_(-1)
    elif damage == "fractional_step":
        slot["step"].fill_(1.5)
    elif damage == "vector_step":
        slot["step"] = torch.ones(2)
    elif damage == "missing_buffer":
        del slot["exp_avg"]
    elif damage == "missing_parameter":
        del state["state"][group["params"][0]]
    elif damage == "extra_parameter":
        state["state"][99] = deepcopy(slot)
    elif damage == "duplicate_id":
        group["params"][1] = group["params"][0]
    elif damage == "parameter_count":
        group["params"].pop()
    elif damage == "nan_lr":
        group["lr"] = float("nan")
    elif damage == "bad_betas":
        group["betas"] = (0.9, 1.0)
    elif damage == "negative_eps":
        group["eps"] = -1
    else:
        group["maximize"] = True
    target = new_optimizer()
    before = deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="Go1 optimizer"):
        restore_go1_optimizer(target, state)
    assert target.state_dict() == before


def test_invalid_moments_cannot_replace_last_checkpoint(trained, tmp_path):
    path = tmp_path / "model.pt"
    checkpoint = {
        "optimizer_state_dict": deepcopy(trained.state_dict()),
        "iteration": 1,
    }
    save_go1_checkpoint(path, checkpoint, optimizer=trained)
    before = path.read_bytes()
    first = next(iter(checkpoint["optimizer_state_dict"]["state"].values()))
    first["exp_avg_sq"].fill_(float("inf"))
    with pytest.raises(ValueError, match="Go1 optimizer exp_avg_sq"):
        save_go1_checkpoint(path, checkpoint, optimizer=trained)
    assert path.read_bytes() == before
    assert not path.with_suffix(".pt.tmp").exists()


def test_resume_rejects_negative_moment_before_creating_simulation(
    tmp_path, monkeypatch
):
    pytest.importorskip("mjbatch")
    pytest.importorskip("mujoco_menagerie")
    from embodiedforge._mjbatch_recipe import go1_train
    from embodiedforge.locomotion import go1, go1_ppo
    from embodiedforge.recipes import sha256

    policy = go1_ppo.ActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    step(optimizer)
    state = optimizer.state_dict()
    next(iter(state["state"].values()))["exp_avg_sq"].fill_(-1)
    checkpoint = tmp_path / "input.pt"
    torch.save(
        {
            "iteration": 0,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": state,
        },
        checkpoint,
    )
    request = {
        "checkpoint": str(checkpoint),
        "input_checkpoint_sha256": sha256(checkpoint),
        "input_checkpoint_iteration": 0,
        "start_iteration": 1,
        "threads": 1,
        "seed": 0,
        "num_envs": 4,
        "horizon": 4,
        "updates": 1,
        "go1_learning_rate": None,
        "input_learning_rate_override": None,
        "go1_command_profile": "original",
        "input_command_profile": "original",
        "go1_reward_profile": "original",
        "input_reward_profile": "original",
        "go1_semantics": "upstream-v1",
        "input_task_semantics": "upstream-v1",
    }

    def unexpected_simulation(*args, **kwargs):
        pytest.fail("Damaged optimizer state reached simulation creation")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(go1, "Go1", unexpected_simulation)
    with pytest.raises(ValueError, match="Go1 optimizer exp_avg_sq"):
        go1_train(request, go1)
    assert not (tmp_path / "training-runtime.json").exists()
    assert not (tmp_path / "resume-report.json").exists()


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_save_preserves_checkpoint_and_removes_partial_file(
    trained, tmp_path, monkeypatch, failure
):
    path = tmp_path / "model.pt"
    checkpoint = {"optimizer_state_dict": trained.state_dict(), "iteration": 1}
    save_go1_checkpoint(path, checkpoint, optimizer=trained)
    before = path.read_bytes()

    def failed_write(value, temporary):
        temporary.write_bytes(b"partial checkpoint")
        raise OSError("Injected write failure")

    def failed_replace(self, target):
        raise OSError("Injected replace failure")

    if failure == "write":
        monkeypatch.setattr(torch, "save", failed_write)
    else:
        monkeypatch.setattr(Path, "replace", failed_replace)
    with pytest.raises(OSError, match="Injected"):
        save_go1_checkpoint(path, checkpoint, optimizer=trained)
    assert path.read_bytes() == before
    assert not path.with_suffix(".pt.tmp").exists()
