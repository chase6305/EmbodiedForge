"""Checkpoint publication must remain recoverable across interruption windows."""

import json
from pathlib import Path

import pytest

from embodiedforge import _go1_checkpoint, recipes
from embodiedforge._go1_recovery import publish_training_checkpoint, recovery_result


@pytest.fixture
def run(tmp_path, monkeypatch):
    request = {
        "task": "go1-joystick",
        "command": "train",
        "start_iteration": 0,
        "updates": 100,
        "num_envs": 4,
        "horizon": 4,
        "go1_semantics": "transition-v2",
        "go1_reward_profile": "original",
        "go1_command_profile": "original",
        "go1_learning_rate": None,
    }
    frozen = recipes.snapshot_implementation(
        Path(recipes.__file__).parent, tmp_path / "implementation/embodiedforge"
    )
    manifest = {
        "schema": 1,
        "workflow": "recipe_train",
        "recipe": "go1-joystick",
        "source": recipes.GO1_STANDALONE_SOURCE,
        "status": "interrupted",
        "request": request,
        "implementation": {"path": "implementation/embodiedforge", **frozen},
    }
    for name, data in (
        ("request", request),
        ("run", manifest),
        ("runtime", {"versions": {}}),
        ("assets", {}),
    ):
        recipes.write_json(tmp_path / f"{name}.json", data)

    # These tests cover file publication. Real Adam/policy validation is exercised
    # in test_go1_optimizer and the actual interrupted-process integration below.
    def save(path, checkpoint, **kwargs):
        path.write_bytes(str(checkpoint["iteration"]).encode())

    monkeypatch.setattr(_go1_checkpoint, "save_go1_checkpoint", save)
    monkeypatch.chdir(tmp_path)
    checkpoint = {
        "iteration": 24,
        "task_semantics": "transition-v2",
        "reward_profile": "original",
        "command_profile": "original",
        "learning_rate_override": None,
    }
    publish_training_checkpoint(checkpoint, optimizer=None, request=request)
    return tmp_path, manifest, checkpoint


@pytest.mark.parametrize("status", ["interrupted", "timed_out", "failed"])
def test_resume_uses_saved_progress_and_keeps_stopped_status(run, status):
    root, manifest, _ = run
    manifest["status"] = status
    recipes.write_json(root / "run.json", manifest)
    # Later metric rows or an incomplete canonical file cannot select the input.
    (root / "metrics.jsonl").write_text('{"iteration":99}\n')
    (root / "model.pt").write_bytes(b"incomplete canonical copy")
    path, recovered = recipes.checkpoint_input(
        root, "go1-joystick", allow_recovery=True
    )
    assert path.read_bytes() == b"24"
    assert recovered["status"] == status
    assert recovered["checkpoint_origin"] == "recovery.json"
    assert recovered["result"]["completed_updates"] == 25
    assert json.loads((root / "run.json").read_text()) == manifest
    with pytest.raises(ValueError, match="completed training run"):
        recipes.checkpoint_input(root, "go1-joystick")  # evaluation/live stay strict


def test_running_source_is_not_accepted_even_with_a_valid_checkpoint(run):
    root, manifest, _ = run
    manifest["status"] = "running"
    recipes.write_json(root / "run.json", manifest)
    with pytest.raises(ValueError, match="completed training run"):
        recipes.checkpoint_input(root, "go1-joystick", allow_recovery=True)


@pytest.mark.parametrize("failure", ["receipt", "canonical"])
def test_interruption_during_publication_keeps_a_committed_checkpoint(
    run, monkeypatch, failure
):
    root, manifest, checkpoint = run
    checkpoint["iteration"] = 49
    replace = Path.replace

    def interrupt(path, target):
        if path.name == (
            "recovery.json.tmp" if failure == "receipt" else "model.pt.tmp"
        ):
            raise KeyboardInterrupt
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        publish_training_checkpoint(
            checkpoint, optimizer=None, request=manifest["request"]
        )
    result = recovery_result(root, manifest)
    assert result["checkpoint_iteration"] == (24 if failure == "receipt" else 49)
    assert (root / result["checkpoint"]).read_bytes() == str(
        result["checkpoint_iteration"]
    ).encode()


def test_retention_keeps_two_recent_snapshots(run):
    root, manifest, checkpoint = run
    for iteration in (49, 74):
        checkpoint["iteration"] = iteration
        publish_training_checkpoint(
            checkpoint, optimizer=None, request=manifest["request"]
        )
    assert sorted(path.name for path in (root / "checkpoints").iterdir()) == [
        "model-000000049.pt",
        "model-000000074.pt",
    ]
    assert recovery_result(root, manifest)["checkpoint_iteration"] == 74
    assert (root / "model.pt").read_bytes() == b"74"


@pytest.mark.parametrize(
    "damage", ["weights", "request", "counter", "source", "symlink", "no_receipt"]
)
def test_recovery_rejects_unverifiable_inputs(run, damage):
    root, manifest, _ = run
    if damage == "weights":
        (root / "checkpoints/model-000000024.pt").write_bytes(b"wrong")
    elif damage == "request":
        (root / "request.json").write_text("{}")
    elif damage == "counter":
        path = root / "recovery.json"
        data = json.loads(path.read_text())
        data["result"]["completed_updates"] = 100
        recipes.write_json(path, data)
    elif damage == "source":
        (root / "implementation/embodiedforge/locomotion/go1.py").write_text("changed")
    elif damage == "symlink":
        path = root / "checkpoints/model-000000024.pt"
        path.rename(root / "outside.pt")
        path.symlink_to(root / "outside.pt")
    else:
        (root / "recovery.json").unlink()
    with pytest.raises(ValueError):
        recovery_result(root, manifest)


def test_real_interrupted_training_resumes_from_committed_iteration(
    tmp_path, monkeypatch, capsys
):
    import os
    import signal
    import subprocess
    import time

    pytest.importorskip("torch")
    pytest.importorskip("mjbatch")
    pytest.importorskip("mujoco_menagerie")
    source, resumed = tmp_path / "interrupted", tmp_path / "resumed"
    common = [
        "--task",
        "go1-joystick",
        "--standalone",
        "--num-envs",
        "4",
        "--horizon",
        "4",
        "--threads",
        "1",
        "--timeout",
        "60",
    ]

    def interrupt_after_save(command, output, env, timeout):
        with (output / "console.log").open("w") as log:
            process = subprocess.Popen(
                command,
                cwd=output,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 20
                while not (output / "recovery.json").exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        pytest.fail(
                            "Training stopped or timed out before saving a checkpoint"
                        )
                    time.sleep(0.02)
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        raise KeyboardInterrupt

    with monkeypatch.context() as isolated:
        isolated.setattr(recipes, "run_process", interrupt_after_save)
        with pytest.raises(KeyboardInterrupt):
            recipes.main(
                ["train", *common, "--updates", "100000", "--output", str(source)]
            )
    original = (source / "run.json").read_bytes()
    manifest = json.loads(original)
    assert manifest["status"] == "interrupted"
    receipt = json.loads((source / "recovery.json").read_text())["result"]
    assert receipt["completed_updates"] >= 25
    assert receipt["completed_updates"] < 100000
    capsys.readouterr()
    recipes.main(["status", "--run", str(source)])
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "interrupted"
    assert status["completed_updates"] is None
    assert (
        status["recoverable_checkpoint"]["checkpoint_iteration"]
        == receipt["checkpoint_iteration"]
    )
    recipes.main(
        [
            "train",
            *common,
            "--resume-run",
            str(source),
            "--updates",
            "1",
            "--output",
            str(resumed),
        ]
    )
    result = json.loads((resumed / "run.json").read_text())
    assert result["status"] == "complete"
    assert result["request"]["start_iteration"] == receipt["checkpoint_iteration"] + 1
    assert result["result"]["cumulative_updates"] == receipt["cumulative_updates"] + 1
    assert result["input_sha256"] == receipt["checkpoint_sha256"]
    report = json.loads((resumed / "resume-report.json").read_text())
    assert report["source_status"] == "interrupted"
    assert report["checkpoint_origin"] == "recovery.json"
    assert (source / "run.json").read_bytes() == original
