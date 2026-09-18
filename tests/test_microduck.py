"""Process boundary, provenance and failure recovery for upstream training."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest
from microduck_report_fixture import evaluation_report

from embodiedforge import microduck


def test_worker_directory_cannot_shadow_stdlib(tmp_path):
    # The worker sits beside logging.py: script-directory imports must be disabled.
    script = tmp_path / "probe.py"
    script.write_text("import logging; print(logging.getLogger('probe').name)")
    (tmp_path / "logging.py").write_text("raise RuntimeError('shadowed stdlib')")
    command = microduck.training_command(
        Path(sys.executable), num_envs=64, iterations=5
    )
    result = subprocess.run(
        command[:2] + [str(script)], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "probe"


def test_child_does_not_inherit_python_environment(monkeypatch):
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        monkeypatch.setenv(name, "/some/other/environment")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env = microduck.child_environment()
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & env.keys()
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["WANDB_MODE"] == "disabled"


def test_source_changes_rejected(tmp_path):
    repo = tmp_path / "upstream"
    (repo / "src/mjlab_microduck").mkdir(parents=True)
    for file in ("pyproject.toml", "uv.lock", "src/mjlab_microduck/train_cli.py"):
        (repo / file).touch()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@test",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ValueError, match="Expected Microduck revision"):
        microduck.source_identity(repo)


@pytest.mark.parametrize("interrupt", [False, True])
def test_failed_run_preserves_status_and_checkpoints(
    tmp_path, monkeypatch, caplog, interrupt
):
    output = tmp_path / "run"
    args = Namespace(command="train", num_envs=64, iterations=5, output=output)

    def fail(command, *, cwd, env, log_path=None, echo=True):
        if "mjlab_microduck.train_cli" not in command:
            return
        checkpoint = cwd / "logs/rsl_rl/microduck/first/model_0.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        (checkpoint.parent / "model_backup.pt").touch()
        (checkpoint.parent / "model_900.pt").mkdir()
        (checkpoint.parent / "model_999.pt").symlink_to(Path(__file__).resolve())
        (checkpoint.parent / "model_1000.pt.partial.tmp").touch()
        if interrupt:
            raise KeyboardInterrupt
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(microduck, "run", fail)
    with pytest.raises(
        KeyboardInterrupt if interrupt else subprocess.CalledProcessError
    ):
        microduck.train_run(
            args, Path("/isolated/bin/python"), {"revision": "test"}, {}
        )
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == ("interrupted" if interrupt else "failed")
    assert len(manifest["checkpoints"]) == 1
    assert manifest["checkpoints"][0].endswith("model_0.pt")
    if not interrupt:
        assert "exit status 7" in manifest["error"]
    assert "finished_at" in manifest
    with pytest.raises(FileExistsError):
        microduck.train_run(args, Path("/isolated/bin/python"), {}, {})

    write_json = microduck.write_json

    def fail_final_write(path, value):
        if "finished_at" in value:
            raise OSError("disk full")
        write_json(path, value)

    monkeypatch.setattr(microduck, "write_json", fail_final_write)
    args.output = tmp_path / "unwritable-final-record"
    with pytest.raises(
        KeyboardInterrupt if interrupt else subprocess.CalledProcessError
    ):
        microduck.train_run(args, Path("/isolated/bin/python"), {}, {})
    assert "Could not finalize run record" in caplog.text
    assert "disk full" in caplog.text


def test_train_selects_latest_checkpoint_without_automatic_export(
    tmp_path, monkeypatch
):
    output = tmp_path / "output with spaces"
    calls = []

    def execute(command, *, cwd, env, log_path=None, echo=True):
        calls.append(command)
        if "mjlab_microduck.train_cli" in command:
            folder = cwd / "logs/rsl_rl/microduck/first"
            folder.mkdir(parents=True)
            for iteration in (2, 10):
                (folder / f"model_{iteration}.pt").touch()
            (folder / "model_backup.pt").touch()
            (folder / "model_900.pt").mkdir()
        elif "metrics" in command:
            assert command[4].endswith("model_10.pt")
            assert command[5] == "12"

    monkeypatch.setattr(microduck, "run", execute)
    microduck.train_run(
        Namespace(command="train", output=output, num_envs=128, iterations=12),
        Path("/isolated/bin/python"),
        {},
        {},
    )
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["checkpoint"].endswith("model_10.pt")
    assert "onnx" not in manifest
    assert len(calls) == 3
    train = calls[1]
    assert train[train.index("--env.scene.num-envs") + 1] == "128"
    assert train[train.index("--agent.max-iterations") + 1] == "12"
    assert train[train.index("--agent.logger") + 1] == "tensorboard"

    write_json = microduck.write_json

    def fail_final_write(path, value):
        if "finished_at" in value:
            raise OSError("disk full")
        write_json(path, value)

    monkeypatch.setattr(microduck, "write_json", fail_final_write)
    with pytest.raises(OSError, match="disk full"):
        microduck.train_run(
            Namespace(
                command="train", output=tmp_path / "unsaved", num_envs=128, iterations=12
            ),
            Path("/isolated/bin/python"),
            {},
            {},
        )


def test_ambiguous_training_checkpoints_preserve_failure_record(tmp_path, monkeypatch):
    output = tmp_path / "run"

    def execute(command, **kwargs):
        if "mjlab_microduck.train_cli" in command:
            for session in ("first", "second"):
                checkpoint = output / f"logs/rsl_rl/microduck/{session}/model_1.pt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()

    monkeypatch.setattr(microduck, "run", execute)
    with pytest.raises(ValueError, match="Ambiguous checkpoint directories"):
        microduck.train_run(
            Namespace(command="train", output=output, num_envs=64, iterations=5),
            Path("/isolated/python"),
            {},
            {},
        )
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "failed" and manifest["finished_at"]
    assert len(manifest["checkpoints"]) == 2
    assert "checkpoint" not in manifest


def test_missing_checkpoint_is_not_reported_as_success(tmp_path, monkeypatch):
    monkeypatch.setattr(microduck, "run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="without producing a checkpoint"):
        microduck.train_run(
            Namespace(
                command="train", num_envs=64, iterations=5, output=tmp_path / "empty"
            ),
            Path("/isolated/bin/python"),
            {},
            {},
        )


def test_train_requires_explicit_iteration_budget():
    with pytest.raises(SystemExit) as exc:
        microduck.main(["train", "--repo", "/upstream", "--output", "/output"])
    assert exc.value.code == 2


@pytest.mark.parametrize("bad_audit", [False, True])
def test_resume_snapshots_input_and_validates_offset(tmp_path, monkeypatch, bad_audit):
    source = tmp_path / "source [old].pt"
    source.write_bytes(b"original checkpoint")
    output = tmp_path / "resumed"
    calls = []

    def execute(command, *, cwd, env, log_path=None, echo=True):
        assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env
        assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
        calls.append(command)
        if "checkpoint" in command:
            snapshot = Path(command[-2])
            assert snapshot.read_bytes() == b"original checkpoint"
            source.write_bytes(b"subsequent external edit")
            Path(command[-1]).write_text(
                json.dumps(
                    {
                        "iteration": 42,
                        "common_step_counter": 1032,
                        "sha256": "snapshot-sha",
                        "learning_rate": 0.00003,
                    }
                )
            )
        if "resume-train" in command:
            assert Path(command[4]).read_bytes() == b"original checkpoint"
            assert command[command.index("--agent.max-iterations") + 1] == "2"
            (cwd / "resume.initialization.json").write_text(
                json.dumps(
                    {
                        "counter_at_first_reset": 0 if bad_audit else 1032,
                        "checkpoint_sha256": "snapshot-sha",
                        "num_envs": 64,
                    }
                )
            )
            model = cwd / "logs/rsl_rl/microduck/new/model_43.pt"
            model.parent.mkdir(parents=True)
            model.touch()

    monkeypatch.setattr(microduck, "run", execute)
    args = Namespace(
        command="train",
        output=output,
        resume=source,
        num_envs=64,
        iterations=2,
        source_run={"checkpoint_sha256": "snapshot-sha"},
    )
    if bad_audit:
        with pytest.raises(RuntimeError, match="initialization disagrees"):
            microduck.train_run(args, Path("/isolated/bin/python"), {}, {})
        manifest = json.loads((output / "run.json").read_text())
        assert manifest["status"] == "failed"
        assert not any("metrics" in command for command in calls)
        return
    microduck.train_run(args, Path("/isolated/bin/python"), {}, {})
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["source_run"] == args.source_run
    from embodiedforge._microduck_run import checkpoint_digest

    assert manifest["checkpoint_sha256"] == checkpoint_digest(
        Path(manifest["checkpoint"])
    )
    assert output / manifest["checkpoint_relative"] == Path(manifest["checkpoint"])
    assert manifest["resume"]["iteration"] == 42
    assert manifest["resume"]["learning_rate"] == 0.00003
    assert manifest["resume_initialization"]["counter_at_first_reset"] == 1032
    assert Path(manifest["resume"]["snapshot"]).read_bytes() == b"original checkpoint"
    assert len(manifest["checkpoints"]) == 1
    metrics = next(command for command in calls if "metrics" in command)
    assert metrics[-1] == "42"


def test_missing_resume_does_not_create_output(tmp_path):
    output = tmp_path / "unused"
    with pytest.raises(ValueError, match="Resume checkpoint not found"):
        microduck.train_run(
            Namespace(
                command="train",
                output=output,
                resume=tmp_path / "missing.pt",
                num_envs=64,
                iterations=2,
            ),
            Path("/isolated/bin/python"),
            {},
            {},
        )
    assert not output.exists()


def test_evaluation_failure_preserves_status(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "evaluation"
    args = Namespace(
        checkpoint=checkpoint,
        output=output,
        num_envs=2,
        steps=8,
        seed=0,
        velocity=None,
        no_pushes=False,
        onnx=None,
    )

    failure = {
        "error_type": "RuntimeError",
        "error": "Parity failed at step 3, environment 1: ONNX action mismatch",
    }

    def fail(command, **kwargs):
        if command[3] == "check":
            return
        (output / "evaluation.failure.json").write_text(json.dumps(failure))
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(microduck, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        microduck.evaluate_run(args, Path("/isolated/bin/python"), {}, {})
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failed_phase"] == "evaluate"
    assert manifest["evaluation_failure"] == failure
    assert "finished_at" in manifest
    with pytest.raises(FileExistsError):
        microduck.evaluate_run(args, Path("/isolated/bin/python"), {}, {})


@pytest.mark.parametrize("invalid", ["velocity", "onnx"])
def test_invalid_evaluation_input_does_not_create_run(tmp_path, invalid):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    args = Namespace(
        checkpoint=checkpoint,
        output=tmp_path / "evaluation",
        num_envs=2,
        steps=8,
        seed=0,
        velocity=[float("nan"), 0, 0] if invalid == "velocity" else None,
        no_pushes=False,
        onnx=tmp_path / "missing.onnx" if invalid == "onnx" else None,
    )
    with pytest.raises(ValueError):
        microduck.evaluate_run(args, Path("/isolated/bin/python"), {}, {})
    assert not args.output.exists()


def test_evaluation_options_reach_worker_and_manifest(tmp_path, monkeypatch):
    checkpoint, onnx = tmp_path / "model.pt", tmp_path / "policy with spaces.onnx"
    checkpoint.touch()
    onnx.touch()
    args = Namespace(
        checkpoint=checkpoint,
        output=tmp_path / "evaluation",
        num_envs=2,
        steps=8,
        seed=3,
        velocity=[-0.2, 0.1, -0.5],
        no_pushes=True,
        onnx=onnx,
    )

    def complete(command, **kwargs):
        if command[3] == "evaluate":
            options = json.loads(command[9])
            assert options == {
                "velocity": args.velocity,
                "no_pushes": True,
                "onnx": str(args.output / "inputs/policy.onnx"),
                "curriculum_step": None,
            }
            Path(command[8]).write_text(json.dumps(evaluation_report(command)))

    monkeypatch.setattr(microduck, "run", complete)
    microduck.evaluate_run(args, Path("/isolated/bin/python"), {}, {})
    manifest = json.loads((args.output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["evaluation_options"]["velocity"] == args.velocity


def test_viser_retains_upstream_checkpoint_browser(tmp_path, monkeypatch):
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").symlink_to(sys.executable)
    (environment / "embodiedforge-source.json").write_text("{}")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    monkeypatch.setattr(microduck, "source_identity", lambda repo: {})
    commands = []
    monkeypatch.setattr(
        microduck, "run", lambda command, **kwargs: commands.append(command)
    )
    args = [
        "play",
        "--repo",
        str(tmp_path),
        "--env-dir",
        str(environment),
        "--checkpoint",
        str(checkpoint),
        "--viewer",
        "viser",
    ]
    microduck.main(args)
    assert commands[0][1:4] == ["-I", "-m", "mjlab.scripts.play"]
    with pytest.raises(SystemExit) as exc:
        microduck.main(args + ["--steps", "60"])
    assert exc.value.code == 1
    assert len(commands) == 1
