"""Portable inference export must be complete, verified, and independent of source paths."""

import json
import shutil
import struct
import subprocess
import sys
from argparse import Namespace

import pytest

from embodiedforge._act_bundle import read_bundle, write_bundle
from embodiedforge.act import bundle, evaluate, main


@pytest.fixture
def source(tmp_path):
    run = tmp_path / "run"
    model = run / "checkpoints/000001/pretrained_model"
    model.mkdir(parents=True)
    header = json.dumps(
        {"value": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    for name in ("model.safetensors", "stats.safetensors"):
        (model / name).write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)
        )
    (model / "config.json").write_text('{"type": "act"}')
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (model / name).write_text(
            json.dumps({"steps": [{"state_file": "stats.safetensors"}]})
        )
    (model / "train_config.json").write_text('{"dataset": {"root": "/missing/data"}}')
    (model / "unused.txt").write_text("not needed for inference")
    (run / "checkpoints/last").symlink_to("000001", target_is_directory=True)
    (run / "act-run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "dataset": {"source": "/missing/recording"},
            }
        )
    )
    return run, model


def test_portable_export_copies_only_required_files(source, tmp_path):
    run, model = source
    output = tmp_path / "portable"
    result = bundle(Namespace(run=run, output=output))
    moved = tmp_path / "moved"
    output.rename(moved)
    run.rename(tmp_path / "source-unavailable")
    metadata, checkpoint, digest = read_bundle(moved)
    assert metadata["checkpoint_fingerprint"] == result["checkpoint_fingerprint"]
    assert len(digest) == 64
    assert not model.exists()
    assert set(p.name for p in checkpoint.iterdir()) == {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "stats.safetensors",
    }
    assert all(not p.is_symlink() for p in checkpoint.iterdir())
    assert not (moved / "act-run.json").exists()


@pytest.mark.parametrize(
    "name", ["model.safetensors", "stats.safetensors", "config.json"]
)
def test_changed_bundle_is_rejected_before_sdk_loading(
    source, tmp_path, monkeypatch, name
):
    run, _ = source
    output = tmp_path / "portable"
    bundle(Namespace(run=run, output=output))
    path = output / "pretrained_model" / name
    path.write_bytes(path.read_bytes() + b" ")
    from embodiedforge import act

    def unexpected(*args, **kwargs):
        pytest.fail("Corrupt bundles must fail before loading the model")

    monkeypatch.setattr(act, "ACTAdapter", unexpected)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        evaluate(Namespace(bundle=output))


@pytest.mark.parametrize("failure", ["exception", "damaged_copy"])
def test_failed_copy_is_not_published(source, tmp_path, monkeypatch, failure):
    _, model = source
    output = tmp_path / "portable"
    original = shutil.copyfile

    def copy(source, destination):
        if failure == "exception":
            raise OSError("simulated copy failure")
        original(source, destination)
        if source.name == "model.safetensors":
            destination.write_bytes(destination.read_bytes() + b"x")

    monkeypatch.setattr(shutil, "copyfile", copy)
    with pytest.raises((OSError, ValueError)):
        write_bundle(model, {}, {}, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".portable-*"))


def test_export_does_not_overwrite_outputs_or_dangling_links(source, tmp_path):
    run, _ = source
    output = tmp_path / "portable"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("user data")
    with pytest.raises(FileExistsError):
        bundle(Namespace(run=run, output=output))
    assert marker.read_text() == "user data"
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        bundle(Namespace(run=run, output=link))
    assert link.is_symlink()


def test_bundle_cli_runs_in_core_environment(source, tmp_path):
    output = tmp_path / "portable"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodiedforge",
            "act",
            "bundle",
            "--run",
            str(source[0]),
            "--checkpoint",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["output"] == str(output)
    assert read_bundle(output)[0]["source"]["selection"]["step"] == 1


def test_evaluation_requires_one_source_and_rejects_bundle_step(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["evaluate", "--run", str(tmp_path), "--bundle", str(tmp_path)])
    assert error.value.code == 2
    with pytest.raises(ValueError, match="--checkpoint applies only"):
        evaluate(Namespace(bundle=tmp_path, checkpoint="1"))
