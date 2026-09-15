"""Describe Go1 continuation without claiming an uninterrupted training replay."""

import json
from pathlib import Path

from .recipes import sha256, write_json


def _values(before, after):
    return {
        key: {
            "before": before.get(key),
            "after": after.get(key),
            "status": (
                "unknown"
                if key not in before or key not in after
                else "unchanged"
                if before[key] == after[key]
                else "changed"
            ),
        }
        for key in sorted(before.keys() | after.keys())
    }


def _code(before, after):
    old = (before.get("implementation") or {}).get("files")
    new = after["implementation"]["files"]
    if not old:
        return {
            "status": "unknown",
            "basis": "no_previous_snapshot_manifest",
            "files": [],
        }
    changes = [
        {
            "path": name,
            "before_sha256": old.get(name),
            "after_sha256": new.get(name),
            "change": "added"
            if name not in old
            else "removed"
            if name not in new
            else "modified",
        }
        for name in sorted(old.keys() | new.keys())
        if old.get(name) != new.get(name)
    ]
    return {
        "status": "changed" if changes else "unchanged",
        "basis": "recorded_snapshot_hashes_not_source_revalidation",
        "files": changes,
    }


def build_resume_report(previous, current, runtime):
    """Compare recorded inputs with the new snapshot and actual worker versions."""
    request = current["request"]
    old_request = previous.get("request", {})
    old_result = previous["result"]
    before, after = {}, {}
    for name in ("num_envs", "horizon", "seed", "threads"):
        if name in old_request:
            before[name] = old_request[name]
        after[name] = request[name]
    for name, key, default in (
        ("task_semantics", "go1_semantics", "upstream-v1"),
        ("reward_profile", "go1_reward_profile", "original"),
        ("command_profile", "go1_command_profile", "original"),
        ("learning_rate_override", "go1_learning_rate", None),
    ):
        # These defaults are the legacy checkpoint contract, not guessed values.
        before[name] = old_result.get(name, default)
        after[name] = request[key]
    return {
        "schema_version": 1,
        "source_run": request["resume_run"],
        "source_manifest": {
            "path": "input-run.json",
            "sha256": request["input_run_manifest_sha256"],
        },
        "checkpoint_sha256": current["input_sha256"],
        "previous_iteration": old_result["checkpoint_iteration"],
        "start_iteration": request["start_iteration"],
        "code": _code(previous, current),
        "configuration": _values(before, after),
        "dependencies": _values(
            old_result.get("runtime", {}).get("versions", {}), runtime["versions"]
        ),
        "launch_mode": {
            "before": old_result.get("runtime", {}).get("launch_mode"),
            "after": runtime["launch_mode"],
        },
        "continuation": {
            "restored": [
                "policy_parameters",
                "normalization_statistics",
                "optimizer_state",
            ],
            "reinitialized": ["simulation_state", "episode_state", "random_generators"],
            "learning_rate": "recomputed_from_iteration_and_current_override",
            "equivalent_to_uninterrupted_training": False,
        },
    }


def record_go1_resume(request):
    """Called only after the checkpoint and optimizer were successfully loaded."""
    source = Path(request["input_run_manifest"])
    if sha256(source) != request["input_run_manifest_sha256"]:
        raise ValueError("Resume input manifest SHA256 mismatch")
    previous = json.loads(source.read_text())
    current = json.loads(Path("run.json").read_text())
    runtime = json.loads(Path("runtime.json").read_text())
    report = build_resume_report(previous, current, runtime)
    write_json(Path("resume-report.json"), report)
    summary = {
        "code": report["code"]["status"],
        **{
            name: [
                key
                for key, value in report[name].items()
                if value["status"] == "changed"
            ]
            for name in ("configuration", "dependencies")
        },
    }
    print(json.dumps({"resume": summary, "report": "resume-report.json"}), flush=True)
    return {"path": "resume-report.json", "sha256": sha256(Path("resume-report.json"))}
