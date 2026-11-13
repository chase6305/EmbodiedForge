"""Offline checks reject partial and non-finite training outputs."""

import argparse
import json
import subprocess
import sys

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


def test_isolated_worker_verifies_artifacts_without_installed_package(tmp_path):
    from embodiedforge import _h1_worker

    args, _, _ = make_run(tmp_path)
    output = tmp_path / "verification.json"
    subprocess.run(
        [
            sys.executable,
            "-I",
            _h1_worker.__file__,
            "verify",
            "--run",
            str(args.run),
            "--start",
            str(args.start),
            "--updates",
            str(args.updates),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert json.loads(output.read_text()) == verify(args)
    assert not list(tmp_path.glob("verification.json*.tmp"))


def test_verification_hash_matches_loaded_checkpoint_during_replacement(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    args, checkpoint, state = make_run(tmp_path)
    before = verify(args)
    state["iter"] = 3
    replacement = tmp_path / "replacement.pt"
    torch.save(state, replacement)
    load = torch.load

    def load_then_publish(*args, **kwargs):
        result = load(*args, **kwargs)
        replacement.replace(checkpoint)
        return result

    monkeypatch.setattr(torch, "load", load_then_publish)
    assert verify(args) == before


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_checkpoint_metadata_is_rejected(tmp_path, invalid):
    torch = pytest.importorskip("torch")
    args, checkpoint, state = make_run(tmp_path)
    state["metadata"] = {"nested": [(invalid,)]}
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="Non-finite checkpoint scalar"):
        verify(args)


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
