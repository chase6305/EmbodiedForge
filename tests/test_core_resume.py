"""Saved updates survive interruption and resume actual policy/Adam state."""

import hashlib
import json
import signal
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodiedforge import (  # noqa: E402
    Config,
    VectorEnv,
    _core_run,
    training,
)
from embodiedforge._microduck_process import termination_signals  # noqa: E402
from embodiedforge.training import (  # noqa: E402
    PPOConfig,
    _collect_rollout,
    _optimize_policy,
    load_training_checkpoint,
    train_ppo,
)


@pytest.fixture
def trained(tmp_path):
    torch.set_num_threads(1)
    env_config = Config(num_envs=4, max_steps=8)
    ppo = PPOConfig(updates=2, rollout_steps=8, epochs=1, minibatch_size=16)
    root = tmp_path / "source"
    train_ppo(env_config, ppo, root, checkpoint_interval=1)
    return root, env_config, ppo


def assert_state_equal(expected, actual):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            assert_state_equal(expected[key], actual[key])
    elif isinstance(expected, (list, tuple)):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual, strict=True):
            assert_state_equal(a, b)
    else:
        assert expected == actual


def test_resume_matches_policy_adam_restart_and_keeps_input_immutable(
    trained, tmp_path
):
    source, env_config, ppo = trained
    saved = load_training_checkpoint(source)
    policy_before = deepcopy(saved.bundle.policy.state_dict())
    adam_before = deepcopy(saved.optimizer_state)
    source_bytes = (source / "checkpoint.pt").read_bytes()
    config = PPOConfig(
        updates=1,
        rollout_steps=8,
        epochs=1,
        minibatch_size=16,
        learning_rate=1e-4,
        seed=17,
    )
    expected_model = deepcopy(saved.bundle.policy)
    optimizer = torch.optim.Adam(expected_model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(deepcopy(saved.optimizer_state))
    optimizer.param_groups[0]["lr"] = config.learning_rate
    torch.manual_seed(config.seed)
    with VectorEnv(env_config) as env:
        rollout, _, _, _ = _collect_rollout(
            env, expected_model, env.observe(), config.rollout_steps
        )
        _optimize_policy(
            expected_model,
            optimizer,
            rollout,
            config,
            np.random.default_rng(config.seed),
        )
    for index in range(2):
        target = tmp_path / f"resumed-{index}"
        actual_model = train_ppo(env_config, config, target, resume=saved)
        actual = torch.load(target / "checkpoint.pt", weights_only=True)
        assert_state_equal(expected_model.state_dict(), actual_model.state_dict())
        assert_state_equal(optimizer.state_dict(), actual["optimizer"])
        assert actual["updates"] == 3
        record = json.loads((target / "run.json").read_text())
        assert record["status"] == "complete"
        assert record["initial_updates"] == 2
        assert record["completed_updates"] == record["saved_updates"] == 3
        assert record["resume"]["sha256"] == hashlib.sha256(source_bytes).hexdigest()
        metric = json.loads((target / "metrics.jsonl").read_text())
        assert metric["update"] == 3
        assert metric["transitions"] == 32
    assert_state_equal(policy_before, saved.bundle.policy.state_dict())
    assert_state_equal(adam_before, saved.optimizer_state)
    assert (source / "checkpoint.pt").read_bytes() == source_bytes


@pytest.mark.parametrize("error", [OSError, KeyboardInterrupt])
def test_partial_save_keeps_last_committed_update_for_resume(
    tmp_path, monkeypatch, error
):
    root = tmp_path / "interrupted"
    original = torch.save

    def save(value, path, *args, **kwargs):
        if value["updates"] == 2:
            path.write_bytes(b"partial serialization")
            raise error("save interrupted")
        return original(value, path, *args, **kwargs)

    config = PPOConfig(updates=3, rollout_steps=8, epochs=1, minibatch_size=16)
    torch.set_num_threads(1)
    monkeypatch.setattr(torch, "save", save)
    with pytest.raises(error, match="save interrupted"):
        train_ppo(Config(num_envs=4), config, root, checkpoint_interval=1)
    record = json.loads((root / "run.json").read_text())
    assert record["status"] == (
        "interrupted" if error is KeyboardInterrupt else "failed"
    )
    assert record["completed_updates"] == 2
    assert record["saved_updates"] == 1
    assert not list(root.rglob("*.tmp"))
    restored = load_training_checkpoint(root)
    assert restored.updates == 1
    monkeypatch.setattr(torch, "save", original)
    train_ppo(
        Config(num_envs=4),
        PPOConfig(updates=1, rollout_steps=8, epochs=1, minibatch_size=16),
        tmp_path / "recovered",
        resume=root,
    )
    assert load_training_checkpoint(tmp_path / "recovered").updates == 2


def test_manifest_publication_failure_does_not_select_uncommitted_file(
    tmp_path, monkeypatch
):
    original = _core_run.write_json

    def write(path, value):
        if value.get("saved_updates") == 2:
            raise OSError("manifest full")
        return original(path, value)

    monkeypatch.setattr(_core_run, "write_json", write)
    root = tmp_path / "failed"
    with pytest.raises(OSError, match="manifest full"):
        train_ppo(
            Config(num_envs=4),
            PPOConfig(updates=2, rollout_steps=8, epochs=1, minibatch_size=16),
            root,
            checkpoint_interval=1,
        )
    assert load_training_checkpoint(root).updates == 1
    assert (root / "checkpoints/checkpoint-000000002.pt").is_file()
    assert json.loads((root / "run.json").read_text())["status"] == "failed"


def test_periodic_snapshots_keep_two_committed_files(tmp_path):
    root = tmp_path / "run"
    train_ppo(
        Config(num_envs=4),
        PPOConfig(updates=5, rollout_steps=8, epochs=1, minibatch_size=16),
        root,
        checkpoint_interval=1,
    )
    assert sorted(p.name for p in (root / "checkpoints").glob("*.pt")) == [
        "checkpoint-000000004.pt",
        "checkpoint-000000005.pt",
    ]
    assert load_training_checkpoint(root).updates == 5
    assert (root / "checkpoint.pt").read_bytes() == (
        root / "checkpoints/checkpoint-000000005.pt"
    ).read_bytes()


def test_signal_context_works_without_sighup_and_restores_handlers(monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    monkeypatch.delattr(signal, "SIGHUP", raising=False)
    with termination_signals():
        assert signal.getsignal(signal.SIGTERM) is not previous
    assert signal.getsignal(signal.SIGTERM) is previous


def test_alias_copy_failure_still_has_a_verified_recovery_point(tmp_path, monkeypatch):
    original = _core_run.shutil.copyfile

    def copy(source, target):
        if source.name == "checkpoint-000000002.pt":
            target.write_bytes(b"partial alias")
            raise OSError("alias copy failed")
        return original(source, target)

    monkeypatch.setattr(_core_run.shutil, "copyfile", copy)
    root = tmp_path / "alias-failed"
    with pytest.raises(OSError, match="alias copy failed"):
        train_ppo(
            Config(num_envs=4),
            PPOConfig(updates=2, rollout_steps=8, epochs=1, minibatch_size=16),
            root,
            checkpoint_interval=1,
        )
    assert not list(root.rglob("*.tmp"))
    assert load_training_checkpoint(root).updates == 2
    assert torch.load(root / "checkpoint.pt", weights_only=True)["updates"] == 1
    assert json.loads((root / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "damage", ["optimizer", "update", "hash", "path", "spec", "spec_type"]
)
def test_invalid_resume_rejected_before_environment_or_output_creation(
    trained, tmp_path, monkeypatch, damage
):
    source, env_config, ppo = trained
    checkpoint = torch.load(source / "checkpoint.pt", weights_only=True)
    target = tmp_path / "invalid.pt"
    if damage == "optimizer":
        checkpoint["optimizer"]["state"].clear()
    elif damage == "update":
        checkpoint["updates"] = True
    elif damage == "spec":
        checkpoint["env_spec"].pop("control_dt")
    elif damage == "spec_type":
        checkpoint["env_spec"] = None
    elif damage == "hash":
        record = json.loads((source / "run.json").read_text())
        record["checkpoint_sha256"] = "wrong"
        (source / "run.json").write_text(json.dumps(record))
        target = source
    else:
        record = json.loads((source / "run.json").read_text())
        record["checkpoint"] = "../outside.pt"
        (source / "run.json").write_text(json.dumps(record))
        target = source
    if target != source:
        torch.save(checkpoint, target)
    monkeypatch.setattr(
        training,
        "VectorEnv",
        lambda _: pytest.fail("Environment created before validation"),
    )
    with pytest.raises(ValueError):
        train_ppo(env_config, ppo, tmp_path / "invalid-output", resume=target)
    assert not (tmp_path / "invalid-output").exists()


def test_resume_semantics_mismatch_does_not_create_run(trained, tmp_path):
    source, _, ppo = trained
    with pytest.raises(ValueError, match="task"):
        train_ppo(Config(task="hold"), ppo, tmp_path / "mismatch", resume=source)
    assert not (tmp_path / "mismatch").exists()


def test_previous_v2_checkpoint_is_resumable(trained, tmp_path):
    source, env_config, ppo = trained
    checkpoint = torch.load(source / "checkpoint.pt", weights_only=True)
    checkpoint.pop("updates")
    path = tmp_path / "old-v2.pt"
    torch.save(checkpoint, path)
    assert load_training_checkpoint(path).updates == ppo.updates
    train_ppo(
        env_config,
        PPOConfig(updates=1, rollout_steps=8, epochs=1, minibatch_size=16),
        tmp_path / "continued",
        resume=path,
    )
    assert load_training_checkpoint(tmp_path / "continued").updates == 3


def test_cli_uses_one_verified_input_and_inherits_training_configuration(
    trained, tmp_path, monkeypatch
):
    from embodiedforge.cli import main

    source, env_config, ppo = trained
    target = tmp_path / "cli-resume"
    calls = []
    original = training.read_resume

    def read(path):
        result = original(path)
        calls.append(result["sha256"])
        # A later replacement must not change either configuration or weights.
        (source / "checkpoint.pt").write_bytes(b"replaced after reading")
        return result

    monkeypatch.setattr(training, "read_resume", read)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embodiedforge",
            "train",
            "--resume",
            str(source),
            "--updates",
            "1",
            "--output",
            str(target),
        ],
    )
    main()
    assert len(calls) == 1
    saved = load_training_checkpoint(target)
    assert saved.updates == 3
    assert saved.bundle.env_config == env_config
    assert saved.ppo_config.rollout_steps == ppo.rollout_steps
    assert saved.ppo_config.minibatch_size == ppo.minibatch_size
    record = json.loads((target / "run.json").read_text())
    assert record["resume"]["sha256"] == calls[0]


def resume_record(root, update, payload):
    checkpoint = f"checkpoints/checkpoint-{update:09d}.pt"
    (root / checkpoint).parent.mkdir(exist_ok=True)
    (root / checkpoint).write_bytes(payload)
    _core_run.write_json(
        root / "run.json",
        {
            "schema_version": 1,
            "kind": "core-ppo",
            "status": "running",
            "saved_updates": update,
            "checkpoint": checkpoint,
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
        },
    )
    return root / checkpoint


def test_read_live_resume_retries_a_pruned_snapshot(tmp_path, monkeypatch):
    old = resume_record(tmp_path, 1, b"old snapshot")
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path.name)
        if path == old:
            # Two subsequent commits can prune the selected file between
            # reading run.json and opening it. Use the new committed reference.
            resume_record(tmp_path, 3, b"new snapshot")
            old.unlink()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    result = _core_run.read_resume(tmp_path)
    assert result["payload"] == b"new snapshot"
    assert result["saved_updates"] == 3
    assert result["sha256"] == hashlib.sha256(result["payload"]).hexdigest()
    assert reads == ["checkpoint-000000001.pt", "checkpoint-000000003.pt"]


def test_missing_resume_without_new_commit_is_not_retried(tmp_path, monkeypatch):
    path = resume_record(tmp_path, 1, b"snapshot")
    path.unlink()
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(FileNotFoundError):
        _core_run.read_resume(tmp_path)
    assert reads == [path]


def test_live_resume_retry_limit_is_bounded(tmp_path, monkeypatch):
    resume_record(tmp_path, 1, b"snapshot")
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path.name)
        update = int(path.stem.removeprefix("checkpoint-"))
        resume_record(tmp_path, update + 2, b"next snapshot")
        path.unlink()
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(FileNotFoundError):
        _core_run.read_resume(tmp_path)
    assert len(reads) == 3


def test_live_resume_does_not_retry_or_hide_a_hash_mismatch(tmp_path, monkeypatch):
    old = resume_record(tmp_path, 1, b"snapshot")
    original = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        payload = original(path)
        resume_record(tmp_path, 3, b"new snapshot")
        return payload + b"corrupt"

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ValueError, match="hash mismatch"):
        _core_run.read_resume(tmp_path)
    assert reads == [old]
