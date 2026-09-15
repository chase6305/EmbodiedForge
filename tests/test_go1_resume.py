"""Continuation reports distinguish changes from unavailable historical evidence."""

import copy

import pytest

from embodiedforge._go1_resume import build_resume_report, record_go1_resume


@pytest.fixture
def records():
    request = {
        "num_envs": 4,
        "horizon": 8,
        "seed": 0,
        "threads": 1,
        "go1_semantics": "transition-v2",
        "go1_reward_profile": "original",
        "go1_command_profile": "original",
        "go1_learning_rate": None,
        "resume_run": "/runs/previous",
        "input_run_manifest_sha256": "manifest-hash",
        "start_iteration": 3,
    }
    runtime = {
        "versions": {"numpy": "2.5.2", "torch": "2.9.0+cu128"},
        "launch_mode": "installed_packages",
    }
    snapshot = {
        "files": {"locomotion/go1.py": "task-hash", "locomotion/go1_ppo.py": "ppo-hash"}
    }
    previous = {
        "implementation": copy.deepcopy(snapshot),
        "request": dict(request),
        "result": {
            "checkpoint_iteration": 2,
            "runtime": copy.deepcopy(runtime),
            "task_semantics": "transition-v2",
        },
    }
    current = {
        "request": dict(request),
        "implementation": copy.deepcopy(snapshot),
        "input_sha256": "weights-hash",
    }
    return previous, current, runtime


def test_identical_inputs_do_not_claim_uninterrupted_training(records):
    report = build_resume_report(*records)
    assert report["code"]["status"] == "unchanged"
    assert report["code"]["files"] == []
    assert all(
        item["status"] == "unchanged" for item in report["configuration"].values()
    )
    assert all(
        item["status"] == "unchanged" for item in report["dependencies"].values()
    )
    assert report["continuation"]["equivalent_to_uninterrupted_training"] is False
    assert "random_generators" in report["continuation"]["reinitialized"]
    assert report["previous_iteration"] == 2
    assert report["start_iteration"] == 3


def test_code_and_override_changes_have_before_after_values(records):
    previous, current, runtime = records
    current["implementation"]["files"] = {
        "locomotion/go1.py": "new-task",
        "new.py": "added",
    }
    current["request"].update(
        go1_learning_rate=0.0003, seed=7, go1_reward_profile="balanced-v1"
    )
    runtime["versions"]["torch"] = "2.9.0+cpu"
    report = build_resume_report(previous, current, runtime)
    assert report["code"]["status"] == "changed"
    assert {item["path"]: item["change"] for item in report["code"]["files"]} == {
        "locomotion/go1.py": "modified",
        "locomotion/go1_ppo.py": "removed",
        "new.py": "added",
    }
    assert report["configuration"]["learning_rate_override"] == {
        "before": None,
        "after": 0.0003,
        "status": "changed",
    }
    assert report["configuration"]["seed"]["before"] == 0
    assert report["configuration"]["seed"]["after"] == 7
    assert report["dependencies"]["torch"]["status"] == "changed"


def test_legacy_missing_metadata_is_unknown_not_unchanged(records):
    previous, current, runtime = records
    previous.pop("implementation")
    previous.pop("request")
    previous["result"].pop("runtime")
    report = build_resume_report(previous, current, runtime)
    assert report["code"]["status"] == "unknown"
    assert report["configuration"]["seed"]["status"] == "unknown"
    assert report["dependencies"]["numpy"]["status"] == "unknown"
    # Learning-rate None is a known legacy default, not a missing value.
    assert report["configuration"]["learning_rate_override"]["status"] == "unchanged"


def test_modified_input_manifest_rejected_before_reporting(tmp_path):
    origin = tmp_path / "input-run.json"
    origin.write_text("{}")
    with pytest.raises(ValueError, match="input manifest SHA256 mismatch"):
        record_go1_resume(
            {"input_run_manifest": str(origin), "input_run_manifest_sha256": "wrong"}
        )
