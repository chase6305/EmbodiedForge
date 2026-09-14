"""Offline checks reject partial and non-finite training outputs."""

import argparse

import pytest

from embodiedforge._h1_worker import verify


def make_run(tmp_path, *, start=0, updates=3):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    final = start + updates - 1
    run = tmp_path / "logs/rsl_rl/h1_flat/example"
    (run / "params").mkdir(parents=True)
    for name in ("agent.yaml", "env.yaml"):
        (run / "params" / name).write_text("{}\n")
    state = {
        "iter": final,
        "actor_state_dict": {
            "mlp.0.weight": torch.zeros(128, 69),
            "mlp.6.weight": torch.zeros(19, 128),
        },
        "critic_state_dict": {
            "mlp.0.weight": torch.zeros(128, 69),
            "mlp.6.weight": torch.zeros(1, 128),
        },
        "optimizer_state_dict": {},
    }
    checkpoint = run / f"model_{final}.pt"
    torch.save(state, checkpoint)
    with SummaryWriter(str(run)) as writer:
        for step in range(start, start + updates):
            for tag in ("Loss/value", "Loss/surrogate", "Policy/mean_std"):
                writer.add_scalar(tag, 1.0, step)
    return (
        argparse.Namespace(run=tmp_path, start=start, updates=updates),
        checkpoint,
        state,
    )


@pytest.mark.parametrize("start", [0, 4])
def test_verify_new_and_resumed_update_ranges(tmp_path, start):
    args, _, _ = make_run(tmp_path, start=start)
    result = verify(args)
    assert result["checkpoint_iteration"] == start + 2
    assert result["completed_updates"] == 3
    assert result["scalar_count"] == 9
    assert result["all_scalars_finite"]


def test_existing_final_file_does_not_hide_missing_updates(tmp_path):
    torch = pytest.importorskip("torch")
    args, checkpoint, state = make_run(tmp_path)
    state["iter"] = 3
    torch.save(state, checkpoint.with_name("model_3.pt"))
    args.updates = 4
    with pytest.raises(ValueError, match="Incomplete PPO metric"):
        verify(args)


def test_nonfinite_model_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    args, checkpoint, state = make_run(tmp_path)
    state["actor_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="Non-finite checkpoint"):
        verify(args)


def test_nonfinite_metric_is_rejected(tmp_path):
    args, checkpoint, _ = make_run(tmp_path)
    from torch.utils.tensorboard import SummaryWriter

    with SummaryWriter(str(checkpoint.parent)) as writer:
        writer.add_scalar("Metrics/velocity", float("inf"), 2)
    with pytest.raises(ValueError, match="Non-finite training metric"):
        verify(args)


def test_wrong_policy_dimensions_are_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    args, checkpoint, state = make_run(tmp_path)
    state["actor_state_dict"]["mlp.6.weight"] = torch.zeros(12, 128)
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="H1 flat actor/critic"):
        verify(args)
