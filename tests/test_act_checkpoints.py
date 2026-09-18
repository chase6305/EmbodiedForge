"""Checkpoint file inspection works in the core environment without an ML SDK."""

import json
import struct

import pytest

from embodiedforge._act_checkpoints import (
    checkpoint_inventory,
    inference_fingerprint,
    inspect_checkpoint,
    select_checkpoint,
)


def tensor_file(path):
    header = json.dumps(
        {"value": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


def make_checkpoint(run, step):
    root = run / "checkpoints" / f"{step:06d}"
    model, state = root / "pretrained_model", root / "training_state"
    model.mkdir(parents=True)
    state.mkdir()
    (model / "config.json").write_text(json.dumps({"type": "act"}))
    (model / "train_config.json").write_text(json.dumps({"policy": {"type": "act"}}))
    tensor_file(model / "model.safetensors")
    tensor_file(model / "stats.safetensors")
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        (model / name).write_text(
            json.dumps({"steps": [{"state_file": "stats.safetensors"}]})
        )
    (state / "training_step.json").write_text(json.dumps({"step": step}))
    (state / "optimizer_param_groups.json").write_text("[]")
    for name in ("optimizer_state.safetensors", "rng_state.safetensors"):
        tensor_file(state / name)
    return root


@pytest.fixture
def checkpoints(tmp_path):
    first = make_checkpoint(tmp_path, 1)
    second = make_checkpoint(tmp_path, 2)
    (tmp_path / "checkpoints/last").symlink_to(second.name, target_is_directory=True)
    return tmp_path, first, second


def test_truncated_latest_requires_explicit_fallback(checkpoints):
    run, first, second = checkpoints
    weights = second / "pretrained_model/model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-1])
    entries = checkpoint_inventory(run)
    assert [entry["step"] for entry in entries] == [2, 1]
    assert not entries[0]["inference_ready"]
    assert "truncated" in " ".join(entries[0]["inference_errors"])
    for selector in ("last", "2"):
        with pytest.raises(ValueError, match="No usable"):
            select_checkpoint(run, selector)
    path, selection = select_checkpoint(run, "latest")
    assert path == first
    assert selection["skipped"][0]["step"] == 2
    assert (run / "checkpoints/last").resolve() == second


def test_inference_and_resume_have_distinct_requirements(checkpoints):
    run, first, second = checkpoints
    (second / "training_state/rng_state.safetensors").write_bytes(b"unfinished")
    entry = inspect_checkpoint(second)
    assert entry["inference_ready"] and not entry["resume_ready"]
    assert select_checkpoint(run, "latest")[0] == second
    assert select_checkpoint(run, "latest", for_resume=True)[0] == first


def test_processor_state_is_required(checkpoints):
    _, _, second = checkpoints
    (second / "pretrained_model/stats.safetensors").unlink()
    entry = inspect_checkpoint(second)
    assert not entry["inference_ready"]
    assert "stats.safetensors" in " ".join(entry["inference_errors"])


def test_step_mismatch_is_not_used_for_resume(checkpoints):
    run, first, second = checkpoints
    (second / "training_state/training_step.json").write_text('{"step": 100}')
    assert not inspect_checkpoint(second)["resume_ready"]
    assert select_checkpoint(run, "latest", for_resume=True)[0] == first


def test_malformed_metadata_is_reported(checkpoints):
    _, _, second = checkpoints
    (second / "pretrained_model/config.json").write_text('{"type":')
    (second / "pretrained_model/train_config.json").write_text('{"policy": []}')
    entry = inspect_checkpoint(second)
    assert entry["inference_errors"] and entry["resume_errors"]


@pytest.mark.parametrize("selector", ["../outside", "-1", "unknown"])
def test_invalid_selection_is_rejected(checkpoints, selector):
    with pytest.raises(ValueError, match="checkpoint must be"):
        select_checkpoint(checkpoints[0], selector)


def test_missing_last_can_be_recovered_explicitly(checkpoints):
    run, _, second = checkpoints
    (run / "checkpoints/last").unlink()
    with pytest.raises(FileNotFoundError):
        select_checkpoint(run)
    assert select_checkpoint(run, "latest")[0] == second


def test_step_selects_saved_checkpoint_without_changing_last(checkpoints):
    run, first, second = checkpoints
    assert select_checkpoint(run, "1")[0] == first
    assert (run / "checkpoints/last").resolve() == second


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "stats.safetensors",
    ],
)
def test_inference_identity_covers_more_than_weights(checkpoints, name):
    model = checkpoints[1] / "pretrained_model"
    before = inference_fingerprint(model)
    path = model / name
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        data["changed"] = True
        path.write_text(json.dumps(data))
    else:
        data = path.read_bytes()
        path.write_bytes(data[:-4] + struct.pack("<f", 2.0))
    after = inference_fingerprint(model)
    assert before["sha256"] != after["sha256"]

    def weights(report):
        return next(
            item for item in report["files"] if item["path"] == "model.safetensors"
        )

    assert weights(before) == weights(after)


def test_inference_identity_ignores_location_and_training_state(checkpoints, tmp_path):
    import shutil

    original = checkpoints[1] / "pretrained_model"
    copied = tmp_path / "relocated-model"
    shutil.copytree(original, copied)
    (copied / "train_config.json").write_text("{}")
    (copied / "README.md").write_text("notes")
    assert inference_fingerprint(copied) == inference_fingerprint(original)


@pytest.mark.parametrize("filename", ["../outside.safetensors", ".."])
def test_inference_identity_rejects_invalid_state_reference(checkpoints, filename):
    model = checkpoints[1] / "pretrained_model"
    (model / "policy_preprocessor.json").write_text(
        json.dumps({"steps": [{"state_file": filename}]})
    )
    with pytest.raises(ValueError, match="invalid state_file"):
        inference_fingerprint(model)
