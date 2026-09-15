import argparse
import json
import os
import subprocess
import sys

import pytest

from embodiedforge import recipes


def progress_fixture():
    manifest = {
        "status": "running",
        "request": {
            "task": "go1-joystick",
            "command": "train",
            "start_iteration": 1800,
            "updates": 400,
            "num_envs": 512,
            "horizon": 24,
        },
    }
    row = {
        "iteration": 1811,
        "transitions": 12 * 512 * 24,
        "elapsed_seconds": 6.0,
        "ppo_kl": 0.01,
    }
    return manifest, row


def test_go1_live_status_preserves_checkpoint_completion_and_ignores_partial_tail(
    tmp_path, capsys
):
    manifest, row = progress_fixture()
    (tmp_path / "run.json").write_text(json.dumps(manifest))
    (tmp_path / "metrics.jsonl").write_bytes(
        (json.dumps(row) + '\n{"iteration":1812,"partial":"').encode() + b"\xe4\xb8"
    )
    recipes.main(["status", "--run", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "running"
    assert result["completed_updates"] is None
    assert result["checkpoint"] is None
    assert result["progress"] == {
        "observed_updates": 12,
        "requested_updates": 400,
        "cumulative_updates": 1812,
        "fraction": 0.03,
        "latest_metrics": row,
    }


def test_go1_progress_missing_incomplete_and_other_recipe(tmp_path):
    manifest, row = progress_fixture()
    assert recipes.go1_training_progress(tmp_path, manifest) is None
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row))
    assert recipes.go1_training_progress(tmp_path, manifest) is None
    (tmp_path / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    manifest["request"]["task"] = "wuji-reorient"
    assert recipes.go1_training_progress(tmp_path, manifest) is None
    manifest["request"]["task"] = "go1-joystick"
    manifest["request"]["command"] = "evaluate"
    assert recipes.go1_training_progress(tmp_path, manifest) is None


def test_go1_progress_reads_only_the_complete_tail(tmp_path):
    manifest, row = progress_fixture()
    (tmp_path / "metrics.jsonl").write_text(
        "x" * 100000 + "\n" + json.dumps(row) + "\n"
    )
    assert recipes.go1_training_progress(tmp_path, manifest)["latest_metrics"] == row


@pytest.mark.parametrize(
    "change",
    [
        {"iteration": True},
        {"iteration": 1799},
        {"iteration": 2200},
        {"transitions": 1},
        {"elapsed_seconds": -1},
        {"ppo_kl": float("nan")},
    ],
)
def test_go1_progress_rejects_invalid_complete_rows(tmp_path, change):
    manifest, row = progress_fixture()
    (tmp_path / "metrics.jsonl").write_text(json.dumps({**row, **change}) + "\n")
    with pytest.raises(ValueError, match="Go1"):
        recipes.go1_training_progress(tmp_path, manifest)


@pytest.mark.parametrize("line", ['{"iteration":', "[]", '{"metric":1e999}'])
def test_go1_progress_rejects_malformed_completed_json(tmp_path, line):
    manifest, _ = progress_fixture()
    (tmp_path / "metrics.jsonl").write_text(line + "\n")
    with pytest.raises(ValueError, match="Go1"):
        recipes.go1_training_progress(tmp_path, manifest)


@pytest.mark.parametrize(
    "invalid", [0, -0.1, float("nan"), float("inf"), True, "0.001"]
)
def test_go1_learning_rate_rejects_invalid_requested_and_inherited_values(invalid):
    with pytest.raises(ValueError, match="learning rate"):
        recipes.resolve_go1_learning_rate(invalid)
    with pytest.raises(ValueError, match="learning rate"):
        recipes.resolve_go1_learning_rate(None, {"learning_rate_override": invalid})


def test_go1_learning_rate_inheritance_and_result_validation(tmp_path):
    assert recipes.resolve_go1_learning_rate(None) is None
    assert recipes.resolve_go1_learning_rate(None, {}) is None
    prior = {"learning_rate_override": 0.0001}
    assert recipes.resolve_go1_learning_rate(None, prior) == 0.0001
    assert recipes.resolve_go1_learning_rate(0.0005, prior) == 0.0005
    with pytest.raises(ValueError, match="learning rate"):
        recipes.validate_result(
            {"task": "go1-joystick", "go1_learning_rate": 0.0001},
            {"cases": [{"learning_rate_override": None}]},
            tmp_path,
        )


def test_go1_command_profile_inheritance_and_validation(tmp_path):
    assert recipes.resolve_go1_command_profile(None) == "original"
    assert recipes.resolve_go1_command_profile(None, {}) == "original"
    previous = {"command_profile": "lateral-v1"}
    assert recipes.resolve_go1_command_profile(None, previous) == "lateral-v1"
    assert recipes.resolve_go1_command_profile("original", previous) == "original"
    assert recipes.resolve_go1_command_profile("lateral-v1", {}) == "lateral-v1"
    for requested, prior in (("bad", {}), (None, {"command_profile": "bad"})):
        with pytest.raises(ValueError, match="command profile"):
            recipes.resolve_go1_command_profile(requested, prior)
    for result in ({"command_profile": "original"}, {"cases": [{}]}):
        with pytest.raises(ValueError, match="command profile"):
            recipes.validate_result(
                {"go1_command_profile": "lateral-v1"}, result, tmp_path
            )


def test_go1_reward_profile_inheritance_and_validation(tmp_path):
    assert recipes.resolve_go1_reward_profile(None) == "original"
    assert recipes.resolve_go1_reward_profile(None, {}) == "original"
    assert (
        recipes.resolve_go1_reward_profile(None, {"reward_profile": "tracking-v1"})
        == "tracking-v1"
    )
    assert recipes.resolve_go1_reward_profile("tracking-v1", {}) == "tracking-v1"
    assert (
        recipes.resolve_go1_reward_profile(
            None, {"reward_profile": "tracking-moderate-v1"}
        )
        == "tracking-moderate-v1"
    )
    assert (
        recipes.resolve_go1_reward_profile(None, {"reward_profile": "tracking-turn-v1"})
        == "tracking-turn-v1"
    )
    assert (
        recipes.resolve_go1_reward_profile(
            "tracking-turn-v1", {"reward_profile": "tracking-v1"}
        )
        == "tracking-turn-v1"
    )
    assert (
        recipes.resolve_go1_reward_profile(
            None, {"reward_profile": "tracking-balanced-v1"}
        )
        == "tracking-balanced-v1"
    )
    with pytest.raises(ValueError):
        recipes.resolve_go1_reward_profile(None, {"reward_profile": "unknown"})
    with pytest.raises(ValueError, match="reward profile"):
        recipes.validate_result(
            {"go1_reward_profile": "tracking-v1"},
            {"reward_profile": "original"},
            tmp_path,
        )


@pytest.mark.parametrize("suite", recipes.SWITCHING_SUITES)
def test_settling_criterion_requires_switching_suite(suite):
    args = argparse.Namespace(
        task="go1-joystick", suite=suite, min_settled_fraction=0.8
    )
    assert recipes.criteria_for(args) == {"min_settled_fraction": 0.8}
    args.suite = "basic"
    with pytest.raises(ValueError, match="does not apply"):
        recipes.criteria_for(args)


def test_tracking_criterion_requires_switching_suite_and_valid_fraction():
    args = argparse.Namespace(
        task="go1-joystick", suite="switching", min_tracking_fraction=0.9
    )
    assert recipes.criteria_for(args) == {"min_tracking_fraction": 0.9}
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        args.min_tracking_fraction = invalid
        with pytest.raises(ValueError, match="threshold"):
            recipes.criteria_for(args)
    args.min_tracking_fraction = 0.9
    args.suite = "basic"
    with pytest.raises(ValueError, match="does not apply"):
        recipes.criteria_for(args)


def test_go1_semantics_inheritance_and_explicit_migration():
    assert recipes.resolve_go1_semantics(None) == "transition-v2"
    assert recipes.resolve_go1_semantics(None, {}) == "upstream-v1"
    assert (
        recipes.resolve_go1_semantics(None, {"task_semantics": "transition-v2"})
        == "transition-v2"
    )
    assert recipes.resolve_go1_semantics("transition-v2", {}) == "transition-v2"
    with pytest.raises(ValueError):
        recipes.resolve_go1_semantics(None, {"task_semantics": "unknown"})
    with pytest.raises(ValueError):
        recipes.resolve_go1_semantics("unknown")


def test_go1_result_must_match_requested_semantics(tmp_path):
    with pytest.raises(ValueError, match="task semantics"):
        recipes.validate_result(
            {"go1_semantics": "transition-v2"},
            {"cases": [{"task_semantics": "upstream-v1"}]},
            tmp_path,
        )


def test_recipe_catalog_import_does_not_load_sdks():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import embodiedforge.recipes, sys; "
            "assert not {'torch', 'warp', 'mujoco', 'mjbatch'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_snapshot_detects_modified_source_and_symlink(tmp_path):
    source = tmp_path / "recipe.py"
    source.write_text("original")
    metadata = {"files": {source.name: recipes.sha256(source)}}
    recipes.validate_snapshot(tmp_path, metadata)
    source.write_text("modified")
    with pytest.raises(ValueError, match="Cached source differs"):
        recipes.validate_snapshot(tmp_path, metadata)
    other = tmp_path / "other.py"
    other.write_text("original")
    source.unlink()
    source.symlink_to(other)
    with pytest.raises(ValueError, match="Cached source differs"):
        recipes.validate_snapshot(tmp_path, metadata)


def test_child_environment_isolates_sdk_paths(monkeypatch, tmp_path):
    for key in ("PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "VIRTUAL_ENV"):
        monkeypatch.setenv(key, "wrong")
    child = recipes.child_environment(tmp_path / "mjbatch")
    assert not {"PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX"} & child.keys()
    assert child["CUDA_VISIBLE_DEVICES"] == ""
    assert child["VIRTUAL_ENV"] == str(tmp_path / "mjbatch/.venv")
    assert child["PYTHONNOUSERSITE"] == "1"


def test_timeout_is_failure_even_when_child_handles_interrupt(tmp_path):
    script = "import signal, time; signal.signal(signal.SIGINT, lambda *a: exit(0)); print('ready', flush=True); time.sleep(20)"
    with pytest.raises(subprocess.TimeoutExpired):
        recipes.run_process(
            [sys.executable, "-c", script], tmp_path, os.environ.copy(), 0.3
        )
    assert "ready" in (tmp_path / "console.log").read_text()


@pytest.mark.parametrize(
    "name,value",
    [
        ("min_success_rate", 1.1),
        ("max_drop_rate", -0.1),
        ("max_drop_rate", float("nan")),
    ],
)
def test_invalid_probability_criteria(name, value):
    with pytest.raises(ValueError, match="Invalid acceptance"):
        recipes.criteria_for(argparse.Namespace(task="wuji-reorient", **{name: value}))


def test_criteria_require_the_correct_task_and_do_not_hide_failure():
    with pytest.raises(ValueError, match="does not apply"):
        recipes.criteria_for(
            argparse.Namespace(task="go1-joystick", min_success_rate=0.5)
        )
    result = recipes.assess(
        {"success_rate": 0.4, "drop_rate": 0.0},
        {"min_success_rate": 0.8, "max_drop_rate": 0.1},
    )
    assert result["passed"] is False
    assert [item["passed"] for item in result["checks"]] == [False, True]
    assert recipes.assess({}, {})["passed"] is None


def test_train_artifact_hash_and_update_count(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"checkpoint")
    request = {"command": "train", "updates": 3}
    result = {
        "checkpoint": model.name,
        "checkpoint_sha256": recipes.sha256(model),
        "completed_updates": 3,
        "checkpoint_iteration": 2,
        "finite_tensor_count": 1,
    }
    recipes.validate_result(request, result, tmp_path)
    with pytest.raises(ValueError, match="Incomplete"):
        recipes.validate_result(request, {**result, "completed_updates": 2}, tmp_path)
    model.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        recipes.validate_result(request, result, tmp_path)
    with pytest.raises(ValueError, match="outside"):
        recipes.validate_result(
            request, {**result, "checkpoint": "../model.pt"}, tmp_path
        )


def test_nan_in_nonacceptance_diagnostics_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        recipes.validate_result(
            {"command": "solve"}, {"nested": [float("nan")]}, tmp_path
        )


@pytest.mark.parametrize("suite", ["basic", "extended"])
def test_suite_checks_every_case_and_rejects_duplicate_reports(tmp_path, suite):
    import numpy as np

    from embodiedforge._go1_metrics import evaluation_commands
    from embodiedforge._h1_metrics import FirstEpisodeMetrics

    metrics = FirstEpisodeMetrics(2, 10, 0.02)
    for _ in range(10):
        metrics.update(
            np.zeros((2, 3)), np.zeros((2, 3)), np.zeros(2, bool), np.zeros(2, bool)
        )

    request = {
        "command": "evaluate",
        "task": "go1-joystick",
        "suite": suite,
        "seeds": [0, 1],
        "num_envs": 2,
        "steps": 10,
        "seed": 0,
    }
    cases = [
        {
            **metrics.report(),
            "seed": seed,
            "case": name,
            "command": command,
            "num_envs": 2,
            "steps_limit": 10,
            "survival_fraction": 1.0,
        }
        for seed in request["seeds"]
        for name, command in evaluation_commands(suite).items()
    ]
    recipes.validate_result(request, {"cases": cases}, tmp_path)
    cases[-1]["survival_fraction"] = 0.0
    acceptance = recipes.assess({"cases": cases}, {"min_survival_fraction": 0.8})
    assert acceptance["passed"] is False
    assert len(acceptance["checks"]) == 2 * len(evaluation_commands(suite))
    with pytest.raises(ValueError, match="duplicate"):
        recipes.validate_result(request, {"cases": cases[:-1] + [cases[0]]}, tmp_path)


def test_resume_uses_the_recorded_start_iteration(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    result = {
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": recipes.sha256(checkpoint),
        "completed_updates": 3,
        "checkpoint_iteration": 602,
        "finite_tensor_count": 1,
    }
    recipes.validate_result(
        {"command": "train", "updates": 3, "start_iteration": 600}, result, tmp_path
    )
    with pytest.raises(ValueError, match="Incomplete"):
        recipes.validate_result(
            {"command": "train", "updates": 3, "start_iteration": 599}, result, tmp_path
        )


def test_evaluation_input_rejects_wrong_task_and_tampering(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    data = {
        "schema": 1,
        "workflow": "recipe_train",
        "status": "complete",
        "recipe": "go1-joystick",
        "source": {"revision": recipes.SOURCES["mjbatch"]},
        "result": {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": recipes.sha256(checkpoint),
        },
    }
    recipes.write_json(tmp_path / "run.json", data)
    assert recipes.checkpoint_input(tmp_path, "go1-joystick")[0] == checkpoint
    with pytest.raises(ValueError, match="completed training"):
        recipes.checkpoint_input(tmp_path, "wuji-reorient")
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        recipes.checkpoint_input(tmp_path, "go1-joystick")


@pytest.mark.parametrize(
    "failure,status",
    [
        (KeyboardInterrupt(), "interrupted"),
        (RuntimeError("failed"), "failed"),
        (subprocess.TimeoutExpired("worker", 1), "timed_out"),
    ],
)
def test_lifecycle_records_failure(tmp_path, monkeypatch, failure, status):
    project = tmp_path / "cache/mjbatch"
    project.mkdir(parents=True)
    source = {"revision": recipes.SOURCES["mjbatch"], "files": {"uv.lock": "hash"}}
    recipes.write_json(project / "source.json", source)
    recipes.write_json(
        project / "ready.json", {"revision": source["revision"], "lock_sha256": "hash"}
    )
    monkeypatch.setattr(recipes, "validate_snapshot", lambda *a, **kw: None)

    def fail(*a, **kw):
        raise failure

    monkeypatch.setattr(recipes, "run_process", fail)
    output = tmp_path / "run"
    args = argparse.Namespace(
        command="train",
        task="go1-joystick",
        num_envs=32,
        horizon=24,
        updates=3,
        cache=tmp_path / "cache",
        output=output,
        timeout=1,
    )
    with pytest.raises(type(failure)):
        recipes.execute(args)
    assert json.loads((output / "run.json").read_text())["status"] == status
    with pytest.raises(FileExistsError):
        recipes.execute(args)


def test_native_implementation_snapshot_runs_independently_of_later_edits(tmp_path):
    source = tmp_path / "source" / "embodiedforge"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "marker.py").write_text("VALUE = 7\n")
    (source / "_recipe_worker.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
        "from embodiedforge.marker import VALUE\nprint(VALUE)\n"
    )
    assets = source / "locomotion"
    assets.mkdir()
    (assets / "scene.xml").write_text("<mujoco/>")
    (assets / "LICENSE.mjbatch").write_text("license text")
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "ignored.py").write_text("ignored")
    destination = tmp_path / "run" / "implementation" / "embodiedforge"
    snapshot = recipes.snapshot_implementation(source, destination)
    assert {"locomotion/scene.xml", "locomotion/LICENSE.mjbatch"} <= snapshot[
        "files"
    ].keys()
    assert not any("__pycache__" in name for name in snapshot["files"])
    (source / "marker.py").write_text("VALUE = 99\n")
    recipes.validate_snapshot(
        destination, snapshot, label="Run implementation snapshot"
    )
    executed = subprocess.run(
        [sys.executable, "-I", str(destination / "_recipe_worker.py")],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=True,
    )
    assert executed.stdout.strip() == "7"
    (destination / "marker.py").write_text("VALUE = 100\n")
    with pytest.raises(ValueError, match="Run implementation snapshot differs"):
        recipes.validate_snapshot(
            destination, snapshot, label="Run implementation snapshot"
        )


def test_native_snapshot_rejects_nested_output_and_symlinks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "__init__.py").write_text("")
    with pytest.raises(ValueError, match="outside"):
        recipes.snapshot_implementation(source, source / "nested")
    assert not (source / "nested").exists()
    outside = tmp_path / "outside.py"
    outside.write_text("")
    (source / "linked.py").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        recipes.snapshot_implementation(source, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_standalone_environment_uses_current_python_and_keeps_asset_cache(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "wrong")
    monkeypatch.setenv("VIRTUAL_ENV", "wrong")
    monkeypatch.setenv("MENAGERIE_CACHE_DIR", "/asset-cache")
    result = recipes.child_environment(None)
    assert "PYTHONPATH" not in result and "VIRTUAL_ENV" not in result
    assert result["MENAGERIE_CACHE_DIR"] == "/asset-cache"
    assert result["CUDA_VISIBLE_DEVICES"] == ""
    assert result["PATH"].split(os.pathsep)[0] == os.path.dirname(sys.executable)


def test_standalone_rejects_external_tasks_before_touching_cache(tmp_path):
    with pytest.raises(ValueError, match="Go1 only"):
        recipes.execute(argparse.Namespace(task="wuji-reorient", standalone=True))


def test_standalone_checkpoint_contract_and_hash_are_required(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    data = {
        "schema": 1,
        "workflow": "recipe_train",
        "status": "complete",
        "recipe": "go1-joystick",
        "source": dict(recipes.GO1_STANDALONE_SOURCE),
        "result": {
            "checkpoint": "model.pt",
            "checkpoint_sha256": recipes.sha256(checkpoint),
        },
    }
    recipes.write_json(tmp_path / "run.json", data)
    assert recipes.checkpoint_input(tmp_path, "go1-joystick")[0] == checkpoint
    data["source"]["contract_version"] = 999
    recipes.write_json(tmp_path / "run.json", data)
    with pytest.raises(ValueError, match="completed training"):
        recipes.checkpoint_input(tmp_path, "go1-joystick")
    data["source"] = dict(recipes.GO1_STANDALONE_SOURCE)
    recipes.write_json(tmp_path / "run.json", data)
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        recipes.checkpoint_input(tmp_path, "go1-joystick")
