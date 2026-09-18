"""Export failures must not leave an invalid deployment model or block retries."""

import json
import os
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from embodiedforge import _microduck_export as exporter
from embodiedforge import microduck
from embodiedforge._microduck_run import checkpoint_digest


@pytest.fixture
def export_args(tmp_path):
    checkpoint = tmp_path / "source weights.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    return Namespace(
        checkpoint=checkpoint, output=tmp_path / "output space/policy.onnx", quiet=True
    )


def fake_worker(args, fail=None):
    def execute(command, *, cwd, env, log_path, echo):
        assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env
        assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
        assert echo is False
        log_path.write_text("worker diagnostics\n")
        assert not args.output.exists()
        if "--onnx-file" in command:
            frozen = Path(command[command.index("--checkpoint-file") + 1])
            assert frozen != args.checkpoint
            assert frozen.read_bytes() == b"checkpoint bytes"
            args.checkpoint.write_bytes(b"external changes after snapshot")
            model = Path(command[command.index("--onnx-file") + 1])
            model.write_bytes(b"onnx bytes")
            if fail == "export":
                raise subprocess.CalledProcessError(7, command)
            if fail == "interrupt":
                raise KeyboardInterrupt
        else:
            if fail == "validate":
                raise RuntimeError("nonfinite inference")
            model = Path(command[-1])
            report = {
                "onnx": str(model),
                "actor_dim": 61,
                "action_dim": 14,
                "finite_inference": fail != "report",
            }
            model.with_suffix(".validation.json").write_text(json.dumps(report))

    return execute


def run_export(args, identity=None, env=None):
    exporter.export_run(
        args,
        Path("/isolated/python"),
        identity or {},
        env or {"DISPLAY": ":1", "WAYLAND_DISPLAY": "wayland-0", "MUJOCO_GL": "glfw"},
    )


def attempts(args):
    return sorted(args.output.parent.glob(".policy.onnx.export-*"))


def test_success_publishes_validated_model_and_provenance(export_args, monkeypatch):
    digest = checkpoint_digest(export_args.checkpoint)
    export_args.source_run = {"checkpoint_sha256": digest, "verification": "verified"}
    monkeypatch.setattr(microduck, "run", fake_worker(export_args))
    run_export(export_args)
    assert export_args.output.read_bytes() == b"onnx bytes"
    report = json.loads(export_args.output.with_suffix(".validation.json").read_text())
    assert report["onnx"] == str(export_args.output)
    assert report["onnx_sha256"] == checkpoint_digest(export_args.output)
    assert report["checkpoint_sha256"] == digest
    manifest = json.loads(Path(report["export_run"]).read_text())
    assert manifest["status"] == manifest["phase"] == "complete"
    assert manifest["source_run"] == export_args.source_run
    assert manifest["checkpoint_sha256"] == digest
    assert manifest["logs"] == ["00-export.log", "01-validate.log"]
    assert manifest["finished_at"] >= manifest["started_at"]


@pytest.mark.parametrize(
    "phase", ["export", "validate", "report", "interrupt", "finalize", "finalize-interrupt"]
)
def test_failure_retains_diagnostics_and_allows_same_output_retry(
    export_args, monkeypatch, phase
):
    monkeypatch.setattr(microduck, "run", fake_worker(export_args, phase))
    expected = {
        "export": subprocess.CalledProcessError,
        "validate": RuntimeError,
        "report": ValueError,
        "interrupt": KeyboardInterrupt,
        "finalize": OSError,
        "finalize-interrupt": KeyboardInterrupt,
    }[phase]
    write_json = microduck.write_json

    def write_record(path, value):
        if phase.startswith("finalize") and value.get("status") == "complete":
            assert export_args.output.exists()
            assert export_args.output.with_suffix(".validation.json").exists()
            raise expected("final record could not be saved")
        write_json(path, value)

    monkeypatch.setattr(microduck, "write_json", write_record)
    with pytest.raises(expected):
        run_export(export_args)
    assert not export_args.output.exists()
    assert not export_args.output.with_suffix(".validation.json").exists()
    first = attempts(export_args)[0]
    manifest = json.loads((first / "run.json").read_text())
    assert manifest["status"] == (
        "interrupted" if expected is KeyboardInterrupt else "failed"
    )
    assert manifest["failed_phase"] == (
        "publish"
        if phase.startswith("finalize")
        else "export"
        if phase in ("export", "interrupt")
        else "validate"
    )
    assert (first / "00-export.log").read_text() == "worker diagnostics\n"
    assert (first / "policy.onnx").exists()
    export_args.checkpoint.write_bytes(b"checkpoint bytes")
    monkeypatch.setattr(microduck, "write_json", write_json)
    monkeypatch.setattr(microduck, "run", fake_worker(export_args))
    run_export(export_args)
    assert len(attempts(export_args)) == 2
    assert export_args.output.exists()


@pytest.mark.parametrize(
    "target,symlink",
    [("model", False), ("report", False), ("model", True), ("report", True)],
)
def test_existing_outputs_including_dangling_symlinks_are_preserved(
    export_args, monkeypatch, target, symlink
):
    path = (
        export_args.output
        if target == "model"
        else export_args.output.with_suffix(".validation.json")
    )
    path.parent.mkdir(parents=True)
    if symlink:
        path.symlink_to(path.parent / "missing")
    else:
        path.write_bytes(b"existing")
    monkeypatch.setattr(
        microduck, "run", lambda *a, **kw: pytest.fail("must fail before SDK starts")
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_export(export_args)
    assert not attempts(export_args)
    assert path.is_symlink() if symlink else path.read_bytes() == b"existing"


def test_changed_run_checkpoint_is_rejected_before_export(export_args, monkeypatch):
    export_args.source_run = {"checkpoint_sha256": "previous bytes"}
    monkeypatch.setattr(microduck, "run", lambda *a, **kw: pytest.fail("invalid input"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        run_export(export_args)
    assert not export_args.output.exists()


def test_native_export_and_validation_use_frozen_code(export_args, monkeypatch):
    from embodiedforge._microduck_native import source_identity

    delegate = fake_worker(export_args)

    def execute(command, **kwargs):
        assert command[1] == "-I"
        worker = Path(command[2])
        assert worker.is_relative_to(attempts(export_args)[0] / "implementation")
        assert worker.is_file()
        delegate(command, **kwargs)

    monkeypatch.setattr(microduck, "run", execute)
    run_export(export_args, source_identity(), {"EF_MICRODUCK_NATIVE": "1"})


@pytest.mark.parametrize("collision", ["model", "report", "interrupt", "cleanup"])
def test_publication_races_and_interruptions_do_not_overwrite_or_leave_own_files(
    tmp_path, monkeypatch, caplog, collision
):
    model, report, output, validation = [
        tmp_path / name
        for name in ("staged.onnx", "staged.json", "final.onnx", "final.json")
    ]
    model.write_bytes(b"new model")
    report.write_bytes(b"new report")
    real_link = os.link

    def link(source, destination):
        if (
            destination == (output if collision == "model" else validation)
            and collision in ("model", "report")
        ):
            destination.write_bytes(b"other export")
        real_link(source, destination)
        if collision in ("interrupt", "cleanup") and destination == output:
            raise KeyboardInterrupt("export interrupted")

    unlink = Path.unlink

    def remove(path, *args, **kwargs):
        if collision == "cleanup" and path == output:
            raise PermissionError("removal denied")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(exporter.os, "link", link)
    monkeypatch.setattr(Path, "unlink", remove)
    with pytest.raises(
        KeyboardInterrupt if collision in ("interrupt", "cleanup") else FileExistsError
    ):
        exporter.publish_export(model, report, output, validation)
    if collision == "model":
        assert output.read_bytes() == b"other export" and not validation.exists()
    elif collision == "report":
        assert validation.read_bytes() == b"other export" and not output.exists()
    elif collision == "cleanup":
        assert output.read_bytes() == b"new model" and not validation.exists()
        assert str(output) in caplog.text and "removal denied" in caplog.text
    else:
        assert not output.exists() and not validation.exists()
