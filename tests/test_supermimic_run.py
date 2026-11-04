"""Private data copies and honest SDK process outcomes for the workspace runner."""

import hashlib
import json
import signal
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from embodiedforge import supermimic_run as runner
from embodiedforge.workflows import supermimic_inputs as inputs


def archive(path, payload=b"reference motion"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as stream:
        stream.writestr("joint_pos.npy", payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_private_inputs_survive_shared_cache_changes(tmp_path):
    source, target = tmp_path / "cache", tmp_path / "run/inputs"
    digest = archive(source / "walk.npz")
    snapshot = inputs.snapshot_motions(source, target, {"walk.npz": digest})
    assert (source / "walk.npz").stat().st_ino != (target / "walk.npz").stat().st_ino
    assert not (target / "walk.npz").stat().st_mode & stat.S_IWUSR
    archive(source / "walk.npz", b"changed shared cache")
    inputs.validate_motion_snapshot(target, snapshot["files"])
    assert snapshot["expected_manifest_verified"]


def test_play_snapshot_only_copies_the_selected_clip(tmp_path):
    source, target = tmp_path / "cache", tmp_path / "run"
    hashes = {name: archive(source / name) for name in ("walk.npz", "dance.npz")}
    record = inputs.snapshot_motions(source, target, hashes, names=["walk.npz"])
    assert set(record["files"]) == {"walk.npz"}
    assert not (target / "dance.npz").exists()
    with pytest.raises(ValueError, match="flat NPZ"):
        inputs.snapshot_motions(
            source, tmp_path / "outside", hashes, names=["../walk.npz"]
        )


def test_crc_corruption_is_rejected_before_simulation(tmp_path):
    path = tmp_path / "cache/sprint.npz"
    archive(path)
    content = path.read_bytes().replace(b"reference motion", b"Reference motion", 1)
    path.write_bytes(content)
    with pytest.raises(ValueError, match="sprint.npz.*CRC"):
        inputs.snapshot_motions(path.parent, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_changed_private_file_identifies_the_clip(tmp_path):
    source, target = tmp_path / "cache", tmp_path / "run"
    digest = archive(source / "walk.npz")
    inputs.snapshot_motions(source, target, {"walk.npz": digest})
    (target / "walk.npz").chmod(0o644)
    archive(target / "walk.npz", b"new")
    with pytest.raises(ValueError, match="Private motion changed: walk.npz; expected"):
        inputs.validate_motion_snapshot(target, {"walk.npz": digest})


def test_written_copy_is_verified_independently(tmp_path, monkeypatch):
    source = tmp_path / "cache"
    archive(source / "walk.npz")
    copy = inputs.copy_file_snapshot

    def corrupt(path, target):
        digest = copy(path, target)
        target.write_bytes(b"bad output storage")
        return digest

    monkeypatch.setattr(inputs, "copy_file_snapshot", corrupt)
    with pytest.raises(ValueError, match="Motion changed during copy: walk.npz"):
        inputs.snapshot_motions(source, tmp_path / "run")


def test_expected_hash_and_file_set_are_enforced(tmp_path):
    source = tmp_path / "cache"
    archive(source / "walk.npz")
    with pytest.raises(ValueError, match="Motion hash mismatch: walk.npz"):
        inputs.snapshot_motions(source, tmp_path / "run", {"walk.npz": "0" * 64})
    with pytest.raises(ValueError, match="file set differs"):
        inputs.snapshot_motions(source, tmp_path / "run", {"missing.npz": "0" * 64})


def test_linked_motion_and_snapshot_overwrite_are_rejected(tmp_path):
    source = tmp_path / "cache"
    archive(tmp_path / "real.npz")
    source.mkdir()
    (source / "linked.npz").symlink_to(tmp_path / "real.npz")
    with pytest.raises(ValueError, match="regular file"):
        inputs.snapshot_motions(source, tmp_path / "run")
    (source / "linked.npz").unlink()
    archive(source / "walk.npz")
    inputs.snapshot_motions(source, tmp_path / "run")
    with pytest.raises(FileExistsError):
        inputs.snapshot_motions(source, tmp_path / "run")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1"])
def test_invalid_training_duration_is_rejected(value):
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["train", "--hours=" + value])


def test_dry_run_does_not_launch_or_create_outputs(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("dry run attempted SDK execution")

    monkeypatch.setattr(runner, "launch", unexpected)
    output = tmp_path / "not-created"
    assert runner.main(["eval", "--output", str(output), "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["num_envs"] == 40
    assert plan["expected_checkpoint_sha256"]
    assert not output.exists()


@pytest.mark.parametrize("recipe", ["cuboi", "reference"])
@pytest.mark.parametrize("worker_exit", [7, -signal.SIGTERM])
def test_worker_failure_is_not_reported_as_success(
    tmp_path, monkeypatch, recipe, worker_exit
):
    root = tmp_path / "sdk"
    python = root / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    termination = f"exit {worker_exit}" if worker_exit > 0 else "kill -TERM $$"
    python.write_text(f"#!/bin/sh\necho deliberate-worker-failure\n{termination}\n")
    python.chmod(0o755)
    monkeypatch.setattr(
        runner, "source_identity", lambda *a: {"files": {}, "revision": "test"}
    )
    args = runner.parser().parse_args(
        [
            "train",
            "--fresh",
            "--root",
            str(root),
            "--output",
            str(tmp_path / "run"),
            "--motions",
            str(tmp_path / "motions"),
        ]
    )
    launch = runner.launch
    if recipe == "reference":
        from embodiedforge.workflows import supermimic_reference as reference

        args.task, args.recipe = "backflip", "reference"
        monkeypatch.setattr(reference, "source_identity", lambda *a: {})

        def launch(options):
            return reference.launch(options, {})

    expected_exit = worker_exit if worker_exit > 0 else 128 - worker_exit
    assert launch(runner.resolve_options(args)) == expected_exit
    result = json.loads((tmp_path / "run/launcher.json").read_text())
    assert result["status"] == "failed"
    assert result["worker_exit_observed"] and result["worker_returncode"] == worker_exit
    assert "deliberate-worker-failure" in (tmp_path / "run/command.log").read_text()

    from embodiedforge.runtime.records import read_run

    assert read_run(tmp_path / "run")["workflow"] == "supermimic"


def test_runner_import_has_no_simulation_or_array_dependencies():
    source = Path(__file__).resolve().parents[1] / "src"
    code = f"import sys; sys.path.insert(0, {str(source)!r}); import embodiedforge.supermimic_run; assert not {{'torch', 'mujoco', 'warp', 'numpy'}} & sys.modules.keys()"
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr
