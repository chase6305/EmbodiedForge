"""Evaluation keeps identical model bytes across seeds and offline auditing."""

import json
from argparse import Namespace
from pathlib import Path

import pytest
from microduck_report_fixture import evaluation_report

from embodiedforge import _microduck_inputs as inputs
from embodiedforge import microduck


@pytest.fixture
def models(tmp_path):
    checkpoint, onnx = tmp_path / "source.pt", tmp_path / "source.onnx"
    checkpoint.write_bytes(b"original checkpoint")
    onnx.write_bytes(b"original ONNX")
    output = tmp_path / "evaluation"
    return checkpoint, onnx, output


@pytest.mark.parametrize("onnx_enabled", [False, True])
def test_single_evaluation_freezes_inputs_before_check(
    models, monkeypatch, onnx_enabled
):
    checkpoint, onnx, output = models
    args = Namespace(
        checkpoint=checkpoint,
        onnx=onnx if onnx_enabled else None,
        output=output,
        num_envs=2,
        steps=4,
        seed=0,
        velocity=None,
        no_pushes=False,
    )

    def run(command, **kwargs):
        if command[3] == "check":
            checkpoint.unlink()
            onnx.unlink()
            return
        frozen = Path(command[4])
        assert frozen.read_bytes() == b"original checkpoint"
        options = json.loads(command[9])
        if onnx_enabled:
            assert Path(options["onnx"]).read_bytes() == b"original ONNX"
        report = evaluation_report(command)
        Path(command[8]).write_text(json.dumps(report))

    monkeypatch.setattr(microduck, "run", run)
    microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["input_snapshot"]["checkpoint"]["source"] == str(checkpoint)
    assert ("onnx" in manifest["input_snapshot"]) is onnx_enabled


@pytest.mark.parametrize("role", ["checkpoint", "onnx"])
def test_change_during_copy_rejected(models, monkeypatch, role):
    checkpoint, onnx, output = models
    output.mkdir()
    original = inputs.shutil.copyfileobj

    def copy(reader, writer):
        if Path(reader.name) == (checkpoint if role == "checkpoint" else onnx):
            Path(reader.name).write_bytes(b"changed while copying")
        original(reader, writer)

    monkeypatch.setattr(inputs.shutil, "copyfileobj", copy)
    with pytest.raises(ValueError, match=f"{role} changed while taking snapshot"):
        inputs.snapshot_inputs(output, checkpoint, {"onnx": str(onnx)})


def test_run_hash_checked_before_gpu_work(models, monkeypatch):
    checkpoint, onnx, output = models
    args = Namespace(
        checkpoint=checkpoint,
        onnx=onnx,
        output=output,
        num_envs=2,
        steps=4,
        seed=0,
        velocity=None,
        no_pushes=False,
        source_run={"checkpoint_sha256": "different"},
    )
    monkeypatch.setattr(microduck, "run", lambda *a, **kw: pytest.fail("GPU started"))
    with pytest.raises(ValueError, match="checkpoint SHA256 mismatch"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "failed" and manifest["failed_phase"] == "inputs"


@pytest.mark.parametrize("role", ["checkpoint", "onnx"])
def test_report_must_match_both_snapshot_hashes(models, role):
    checkpoint, onnx, output = models
    output.mkdir()
    snapshot = inputs.snapshot_inputs(output, checkpoint, {"onnx": str(onnx)})
    report = {
        "checkpoint_metadata": {"sha256": snapshot["checkpoint"]["sha256"]},
        "onnx_parity": {"sha256": snapshot["onnx"]["sha256"]},
    }
    report["checkpoint_metadata" if role == "checkpoint" else "onnx_parity"][
        "sha256"
    ] = "changed"
    with pytest.raises(ValueError, match="differs from input snapshot"):
        inputs.verify_report_inputs(report, snapshot)
