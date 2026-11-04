"""Common status inspection preserves legacy formats and never loads a learner."""

import hashlib
import json
import os
import subprocess
import sys

import pytest

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.records import STATUSES, main, read_run, run_identity


@pytest.mark.parametrize("status", sorted(STATUSES))
def test_declared_run_status_is_a_snapshot_without_liveness_or_artifact_claims(
    tmp_path, status
):
    path = tmp_path / "run.json"
    write_json(
        path,
        {"run_contract": run_identity("example", "train", "robot"), "status": status},
    )
    before = path.read_bytes()
    result = read_run(tmp_path)
    assert result["status"] == result["recorded_status"] == status
    assert result["terminal"] == (status not in {"preparing", "running"})
    assert result["identity_source"] == "declared"
    assert result["manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert (
        result["process_liveness"] == result["artifact_verification"] == "not_checked"
    )
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "manifest,identity",
    [
        (
            {"kind": "core-ppo", "schema_version": 1, "env": {"task": "reach"}},
            ("core", "train", "reach"),
        ),
        (
            {"workflow": "recipe_evaluate", "schema": 1, "recipe": "go1-joystick"},
            ("recipes", "evaluate", "go1-joystick"),
        ),
        (
            {"workflow": "h1_train", "schema": 1, "task": "Isaac-Velocity-Flat-H1-v0"},
            ("h1-isaaclab", "train", "Isaac-Velocity-Flat-H1-v0"),
        ),
        (
            {
                "workflow": "evaluate_seeds",
                "schema": 1,
                "task": "Mjlab-Velocity-Flat-MicroDuck",
            },
            ("microduck", "evaluate_seeds", "Mjlab-Velocity-Flat-MicroDuck"),
        ),
        (
            {"task": "h1-mjbatch-flat-v1", "updates": 1},
            ("h1-native", "train", "h1-mjbatch-flat-v1"),
        ),
        ({"backend": "gmr"}, ("gmr", "retarget", None)),
        ({"backend": "sonic"}, ("sonic", "evaluate", None)),
        (
            {
                "schema_version": 1,
                "config": {},
                "env_spec": {},
                "completed_episodes": 0,
                "run_id": "abc",
                "task": "hold",
            },
            ("core-data", "record", "hold"),
        ),
    ],
)
def test_legacy_manifest_identity_is_read_without_rewriting(
    tmp_path, manifest, identity
):
    write_json(tmp_path / "run.json", {**manifest, "status": "failed"})
    result = read_run(tmp_path)
    assert tuple(result[key] for key in ("workflow", "operation", "task")) == identity
    assert result["identity_source"] == "legacy"


@pytest.mark.parametrize("resumed", [False, True])
def test_act_uses_its_own_authoritative_manifest(tmp_path, resumed):
    write_json(
        tmp_path / "act-run.json",
        {
            "schema_version": 1,
            "status": "complete",
            "resumed_from": "source" if resumed else None,
        },
    )
    result = read_run(tmp_path)
    assert result["workflow"] == "act"
    assert result["operation"] == ("resume" if resumed else "train")
    assert result["manifest"] == "act-run.json"


@pytest.mark.parametrize(
    "sdk_request,workflow,task",
    [
        (
            {"config": "maniskill_sac_mlp", "revision": "fixed"},
            "rlinf",
            "maniskill_sac_mlp",
        ),
        (
            {
                "task": "G1-Inspire-HOI-v0",
                "weave_revision": "fixed",
                "isaaclab_revision": "fixed",
            },
            "weave",
            "G1-Inspire-HOI-v0",
        ),
    ],
)
def test_legacy_sdk_identity_comes_from_matching_request(
    tmp_path, sdk_request, workflow, task
):
    write_json(tmp_path / "run.json", {"command": "evaluate", "status": "complete"})
    write_json(tmp_path / "request.json", {**sdk_request, "command": "evaluate"})
    result = read_run(tmp_path)
    assert result["workflow"] == workflow
    assert result["task"] == task
    write_json(tmp_path / "request.json", {**sdk_request, "command": "train"})
    with pytest.raises(ValueError, match="Unrecognized legacy"):
        read_run(tmp_path)


@pytest.mark.parametrize(
    "recorded,normalized", [("recording", "running"), ("closed", "complete")]
)
def test_dataset_status_keeps_original_alongside_common_status(
    tmp_path, recorded, normalized
):
    write_json(
        tmp_path / "run.json",
        {
            "run_contract": run_identity("core-data", "record", "reach"),
            "status": recorded,
        },
    )
    result = read_run(tmp_path)
    assert result["status"] == normalized
    assert result["recorded_status"] == recorded


@pytest.mark.parametrize("schema", [True, 0, 2, "1", None])
def test_invalid_declared_schema_never_falls_back_to_legacy(tmp_path, schema):
    write_json(
        tmp_path / "run.json",
        {
            "kind": "core-ppo",
            "schema_version": 1,
            "status": "complete",
            "run_contract": {
                "schema_version": schema,
                "workflow": "core",
                "operation": "train",
            },
        },
    )
    with pytest.raises(ValueError, match="run_contract schema"):
        read_run(tmp_path)


@pytest.mark.parametrize("status", [None, True, [], {}, "unknown"])
def test_invalid_status_is_rejected(tmp_path, status):
    write_json(
        tmp_path / "run.json",
        {"run_contract": run_identity("example", "train"), "status": status},
    )
    with pytest.raises(ValueError, match="recorded run status"):
        read_run(tmp_path)


def test_unknown_legacy_directory_name_does_not_determine_identity(tmp_path):
    root = tmp_path / "go1-training"
    root.mkdir()
    write_json(root / "run.json", {"status": "complete"})
    with pytest.raises(ValueError, match="Unrecognized legacy"):
        read_run(root)


def test_cli_dispatch_is_sdk_independent_and_reports_corruption_cleanly(tmp_path):
    write_json(
        tmp_path / "run.json",
        {"run_contract": run_identity("example", "train"), "status": "running"},
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from embodiedforge.cli import main; main(); assert not {'torch','mujoco','mjbatch','warp','embodiedforge.recipes','embodiedforge.h1'} & sys.modules.keys()",
            "runs",
            "inspect",
            "--run",
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["workflow"] == "example"
    (tmp_path / "run.json").write_text("{partial")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodiedforge",
            "runs",
            "inspect",
            "--run",
            str(tmp_path),
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Cannot inspect run" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_missing_run_is_a_read_only_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["inspect", "--run", str(tmp_path / "missing")])
    assert error.value.code == 1
    assert "Cannot inspect run" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_act_startup_record_can_be_inspected_before_lerobot_creates_the_run(tmp_path):
    root = tmp_path / "attempt"
    startup = tmp_path / ".attempt.act-run.json"
    write_json(
        startup, {"schema_version": 1, "status": "failed", "error": "startup failed"}
    )
    result = read_run(root)
    assert result["workflow"] == "act"
    assert result["manifest"] == "../.attempt.act-run.json"
    assert not root.exists()
    root.mkdir()
    write_json(root / "act-run.json", {"schema_version": 1, "status": "complete"})
    assert read_run(root)["manifest"] == "act-run.json"
    assert read_run(root)["status"] == "complete"


def test_dataset_status_aliases_do_not_relabel_other_workflows(tmp_path):
    write_json(
        tmp_path / "run.json",
        {"run_contract": run_identity("example", "train"), "status": "closed"},
    )
    with pytest.raises(ValueError, match="recorded run status"):
        read_run(tmp_path)


@pytest.mark.parametrize("operation", ["assess", "compare"])
@pytest.mark.parametrize("native", [False, True])
def test_old_microduck_offline_records_use_input_implementation_identity(
    tmp_path, operation, native
):
    source = (
        {"implementation": "embodiedforge.locomotion.microduck"}
        if native
        else {
            "revision": "53b8971b61baf5b7f3c16d135dd7cac37623de4b",
            "uv_lock_sha256": "a" * 64,
        }
    )
    provenance = {"source": source}
    manifest = {"schema": 1, "workflow": operation, "status": "rejected"}
    if operation == "compare":
        manifest.update(before=provenance, after=provenance)
    else:
        manifest["input"] = provenance
    write_json(tmp_path / "run.json", manifest)
    result = read_run(tmp_path)
    assert result["workflow"] == "microduck"
    assert result["operation"] == operation
    assert result["identity_source"] == "legacy"
    manifest["after" if operation == "compare" else "input"] = {
        "source": {"implementation": "other-task"}
    }
    write_json(tmp_path / "run.json", manifest)
    with pytest.raises(ValueError, match="Unrecognized legacy"):
        read_run(tmp_path)
