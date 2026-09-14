"""Seed-level aggregation, configuration compatibility and batch recovery."""

import copy
import hashlib
import json
import math
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from embodiedforge import microduck
from embodiedforge._microduck_reports import (
    aggregate_evaluations,
    assess_evaluations,
    compare_evaluations,
    load_evaluation_run,
    validate_criteria,
)


def report(seed=0, *, checkpoint_hash="abc", rmse=3.0):
    return {
        "seed": seed,
        "task": microduck.TASK,
        "checkpoint_metadata": {"sha256": checkpoint_hash},
        "conditions": {
            "velocity_body_frame": [0.2, 0, 0],
            "policy_matmul_precision": "ieee",
        },
        "num_envs": 2,
        "steps_per_env": 4,
        "sim_seconds_per_env": 0.08,
        "transitions": 8,
        "onnx_parity": None,
        "finite_observations_actions_rewards": True,
        "termination_counts": {"nan_state": 0, "fell_over": 1},
        "mean_reward_per_transition": 0.1,
        "velocity_tracking": {
            "samples": 8,
            "mean_command": [0.2, 0, 0],
            "mean_actual": [0.1, 0, 0],
            "rmse": [rmse, 0, 0],
            "planar_velocity_rmse_m_s": rmse,
            "axes": ["vx", "vy", "wz"],
            "units": ["m/s", "m/s", "rad/s"],
            "frame": "root link body frame",
            "sampling": "pre-reset",
            "includes_terminal_steps": True,
        },
        "initial_episodes": {
            "duration_steps": [2, 4],
            "mean_observed_duration_seconds": 0.06,
            "survived_full_horizon_count": 1,
            "censored_before_horizon_count": 0,
        },
    }


def test_rmse_pooling_and_seed_variation_are_distinct():
    summary = aggregate_evaluations([report(0, rmse=3), report(1, rmse=4)])
    assert summary["transitions"] == 16
    assert summary["pooled_velocity_rmse"][0] == pytest.approx(math.sqrt(12.5))
    stat = summary["metrics"]["vx_rmse"]
    assert stat["mean"] == 3.5
    assert stat["sample_std"] == pytest.approx(math.sqrt(0.5))
    assert summary["seeds"] == [0, 1]
    assert summary["metrics"]["survived_full_horizon_count"]["mean"] == 1


def test_single_seed_does_not_claim_zero_uncertainty():
    summary = aggregate_evaluations([report()])
    assert summary["metrics"]["vx_rmse"]["sample_std"] is None


def test_mean_velocity_and_signed_bias_do_not_hide_tracking_error():
    first, second = report(0), report(1)
    first["velocity_tracking"]["mean_actual"] = [0.1, -0.1, 0.2]
    second["velocity_tracking"]["mean_actual"] = [0.3, 0.1, -0.2]
    summary = aggregate_evaluations([first, second])
    assert summary["metrics"]["vx_mean_actual"]["mean"] == pytest.approx(0.2)
    assert summary["metrics"]["vx_bias"]["mean"] == pytest.approx(0)
    assert summary["metrics"]["vx_bias"]["sample_std"] > 0
    assert summary["metrics"]["vx_rmse"]["mean"] == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_metadata", {"sha256": "different"}),
        ("conditions", {"policy_matmul_precision": "tf32"}),
        ("num_envs", 3),
        ("steps_per_env", 5),
        ("sim_seconds_per_env", 0.1),
    ],
)
def test_different_evaluation_conditions_cannot_be_combined(field, value):
    other = report(1)
    other[field] = value
    with pytest.raises(ValueError, match="Incompatible"):
        aggregate_evaluations([report(), other])


@pytest.mark.parametrize(
    "invalid",
    ["empty", "duplicate", "missing", "samples", "nan", "negative", "nan_state"],
)
def test_incomplete_and_invalid_results_are_rejected(invalid):
    reports = [report()]
    if invalid == "empty":
        reports = []
    elif invalid == "duplicate":
        reports.append(report())
    elif invalid == "missing":
        del reports[0]["velocity_tracking"]
    elif invalid == "samples":
        reports[0]["velocity_tracking"]["samples"] = 7
    elif invalid == "nan":
        reports[0]["velocity_tracking"]["rmse"][0] = float("nan")
    elif invalid == "negative":
        reports[0]["velocity_tracking"]["rmse"][0] = -1
    else:
        reports[0]["termination_counts"]["nan_state"] = 1
    with pytest.raises(ValueError):
        aggregate_evaluations(reports)


def test_different_onnx_models_cannot_be_combined():
    first, second = report(), report(1)
    first["onnx_parity"] = {
        "sha256": "a",
        "provider": "CPU",
        "atol": 1e-4,
        "rtol": 1e-4,
        "sampling": "rotating environment",
    }
    second["onnx_parity"] = dict(first["onnx_parity"], sha256="b")
    with pytest.raises(ValueError, match="ONNX"):
        aggregate_evaluations([first, second])


@pytest.fixture
def batch(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    args = Namespace(
        checkpoint=checkpoint,
        output=tmp_path / "batch",
        num_envs=2,
        steps=4,
        seed=0,
        seeds=[0, 1, 2],
        velocity=[0.2, 0, 0],
        no_pushes=True,
        onnx=None,
    )
    calls = []

    def run(command, **kwargs):
        if command[3] != "evaluate":
            return
        seed = int(command[7])
        calls.append(seed)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        Path(command[8]).write_text(json.dumps(report(seed, checkpoint_hash=digest)))

    monkeypatch.setattr(microduck, "run", run)
    return args, calls, run


def test_batch_preserves_individual_results_and_refuses_overwrite(batch):
    args, calls, _ = batch
    microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == [0, 1, 2]
    manifest = json.loads((args.output / "run.json").read_text())
    summary = json.loads((args.output / "summary.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["completed_seeds"] == [0, 1, 2]
    assert summary["seed_count"] == 3
    for path in summary["reports"]:
        assert (args.output / path).is_file()
    with pytest.raises(FileExistsError):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})


@pytest.mark.parametrize("interrupt", [False, True])
def test_batch_failure_preserves_completed_seeds_without_summary(
    batch, monkeypatch, interrupt
):
    args, calls, run = batch

    def fail(command, **kwargs):
        if command[3] == "evaluate" and command[7] == "1":
            if interrupt:
                raise KeyboardInterrupt
            raise subprocess.CalledProcessError(1, command)
        run(command, **kwargs)

    monkeypatch.setattr(microduck, "run", fail)
    with pytest.raises(
        KeyboardInterrupt if interrupt else subprocess.CalledProcessError
    ):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    manifest = json.loads((args.output / "run.json").read_text())
    assert manifest["status"] == ("interrupted" if interrupt else "failed")
    assert manifest["completed_seeds"] == [0]
    assert not (args.output / "summary.json").exists()
    assert (args.output / "seed-0/evaluation.json").is_file()
    assert not (args.output / "seed-2").exists()


def test_checkpoint_change_stops_batch(batch, monkeypatch):
    args, calls, run = batch

    def mutate(command, **kwargs):
        if command[3] == "evaluate" and command[7] == "1":
            args.checkpoint.write_bytes(b"replaced checkpoint")
        run(command, **kwargs)

    monkeypatch.setattr(microduck, "run", mutate)
    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == [0, 1]
    assert not (args.output / "summary.json").exists()


@pytest.mark.parametrize("seeds", [[1, 1], [-1], [2**32], []])
def test_invalid_seeds_do_not_create_output(batch, seeds):
    args, _, _ = batch
    args.seeds = seeds
    with pytest.raises(ValueError, match="seeds"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert not args.output.exists()


def test_aggregation_does_not_mutate_source_reports():
    reports = [report(), report(1)]
    before = copy.deepcopy(reports)
    aggregate_evaluations(reports)
    assert reports == before


def test_acceptance_includes_exact_boundary_and_zero_thresholds():
    assessment = assess_evaluations(
        [report()],
        {
            "max_planar_rmse": 3,
            "max_yaw_rmse": 0,
            "min_survival_fraction": 0.5,
        },
    )
    assert assessment["passed"]
    assert len(assessment["checks"]) == 3
    assert [check["unit"] for check in assessment["checks"]] == [
        "m/s",
        "rad/s",
        "fraction",
    ]
    assert assessment["num_envs"] == 2
    assert assessment["steps_per_env"] == 4
    assert assessment["failed_seeds"] == []


def test_passing_average_cannot_hide_failed_seed():
    reports = [report(0, rmse=1), report(1, rmse=3)]
    assert (
        aggregate_evaluations(reports)["metrics"]["planar_velocity_rmse_m_s"]["mean"]
        == 2
    )
    assessment = assess_evaluations(reports, {"max_planar_rmse": 2})
    assert not assessment["passed"]
    assert assessment["failed_seeds"] == [1]
    assert assessment["checks"][1]["actual"] == 3


def test_early_censoring_does_not_satisfy_full_horizon_acceptance():
    result = report()
    result["initial_episodes"].update(
        survived_full_horizon_count=0, censored_before_horizon_count=1
    )
    assessment = assess_evaluations([result], {"min_survival_fraction": 0.5})
    assert not assessment["passed"]
    assert assessment["checks"][0]["actual"] == 0


@pytest.mark.parametrize(
    "criteria",
    [
        {"max_planar_rmse": -1},
        {"max_yaw_rmse": float("nan")},
        {"max_planar_rmse": float("inf")},
        {"min_survival_fraction": 1.1},
        {"min_survival_fraction": True},
        {"unknown": 1},
    ],
)
def test_invalid_acceptance_criteria_are_rejected(criteria):
    with pytest.raises(ValueError, match="criterion"):
        validate_criteria(criteria)


def test_acceptance_cannot_silently_pass_without_criteria():
    with pytest.raises(ValueError, match="criterion"):
        assess_evaluations([report()], {})


@pytest.mark.parametrize("count", [-1, 3, 1.5, True])
def test_invalid_survivor_counts_cannot_pass_acceptance(count):
    result = report()
    result["initial_episodes"]["survived_full_horizon_count"] = count
    with pytest.raises(ValueError):
        assess_evaluations([result], {"min_survival_fraction": 0.5})


def test_rejected_batch_runs_every_seed_and_preserves_complete_summary(batch):
    args, calls, _ = batch
    args.max_planar_rmse = 2
    with pytest.raises(microduck.EvaluationRejected, match="seed 2"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == [0, 1, 2]
    manifest = json.loads((args.output / "run.json").read_text())
    assessment = json.loads((args.output / "acceptance.json").read_text())
    assert manifest["status"] == "rejected"
    assert manifest["completed_seeds"] == [0, 1, 2]
    assert manifest["acceptance_criteria"] == {"max_planar_rmse": 2}
    assert assessment["failed_seeds"] == [0, 1, 2]
    assert (args.output / "summary.json").is_file()
    for seed in calls:
        child = json.loads((args.output / f"seed-{seed}/run.json").read_text())
        assert child["status"] == "complete"
        assert child["acceptance_criteria"] == {}


def test_rejected_single_run_preserves_evaluation_and_acceptance(batch):
    args, calls, _ = batch
    args.seeds = None
    args.min_survival_fraction = 1
    with pytest.raises(microduck.EvaluationRejected, match="min_survival_fraction"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == [0]
    assert (args.output / "evaluation.json").is_file()
    assert not json.loads((args.output / "acceptance.json").read_text())["passed"]
    assert json.loads((args.output / "run.json").read_text())["status"] == "rejected"


def test_successful_acceptance_saves_limits_and_passes(batch):
    args, _, _ = batch
    args.max_planar_rmse = 3
    args.min_survival_fraction = 0.5
    microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert json.loads((args.output / "run.json").read_text())["status"] == "complete"
    assessment = json.loads((args.output / "acceptance.json").read_text())
    assert assessment["passed"]
    assert len(assessment["checks"]) == 6


def test_invalid_threshold_fails_before_any_gpu_work_or_directory_creation(batch):
    args, calls, _ = batch
    args.max_yaw_rmse = float("nan")
    with pytest.raises(ValueError, match="criterion"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == []
    assert not args.output.exists()


def test_cli_returns_distinct_code_for_policy_rejection(batch, monkeypatch, tmp_path):
    args, _, _ = batch
    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    (environment / "embodiedforge-source.json").write_text("{}")
    monkeypatch.setattr(microduck, "source_identity", lambda _: {})
    with pytest.raises(SystemExit) as exc:
        microduck.main(
            [
                "evaluate",
                "--repo",
                str(tmp_path),
                "--env-dir",
                str(environment),
                "--checkpoint",
                str(args.checkpoint),
                "--output",
                str(args.output),
                "--num-envs",
                "2",
                "--steps",
                "4",
                "--max-planar-rmse",
                "2",
            ]
        )
    assert exc.value.code == 3
    assert json.loads((args.output / "run.json").read_text())["status"] == "rejected"


@pytest.fixture
def saved_evaluation(tmp_path):
    root = tmp_path / "evaluation"
    options = {"velocity": [0.2, 0, 0], "no_pushes": True, "onnx": None}
    source = {"revision": "test-revision", "uv_lock_sha256": "test-lock"}

    def save(directory, seed):
        directory.mkdir(parents=True)
        result = report(seed)
        result["checkpoint"] = "/archived/model.pt"
        result["conditions"]["pushes_enabled"] = False
        manifest = {
            "schema": 1,
            "workflow": "evaluate",
            "status": "complete",
            "finished_at": "2026-09-12T00:00:00Z",
            "source": source,
            "task": microduck.TASK,
            "evaluation_options": options,
            "commands": [
                [
                    "python",
                    "-I",
                    "worker",
                    "evaluate",
                    result["checkpoint"],
                    "2",
                    "4",
                    str(seed),
                    "unused.json",
                    json.dumps(options),
                ]
            ],
        }
        (directory / "run.json").write_text(json.dumps(manifest))
        (directory / "evaluation.json").write_text(json.dumps(result))
        (directory / "runtime.json").write_text(
            json.dumps({"python": "3.12", "packages": {}})
        )

    save(root / "seed-0", 0)
    save(root / "seed-1", 1)
    manifest = {
        "schema": 1,
        "workflow": "evaluate_seeds",
        "status": "rejected",
        "finished_at": "2026-09-12T00:00:00Z",
        "source": source,
        "task": microduck.TASK,
        "evaluation_options": options,
        "checkpoint_sha256": "abc",
        "seeds": [0, 1],
        "completed_seeds": [0, 1],
    }
    (root / "run.json").write_text(json.dumps(manifest))
    return root


def test_offline_assessment_requires_no_upstream_or_gpu(
    saved_evaluation, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Offline assessment attempted to access an SDK or start a subprocess"
        )

    monkeypatch.setattr(microduck, "source_identity", forbidden)
    monkeypatch.setattr(microduck, "run", forbidden)
    output = tmp_path / "assessment"
    before = {
        str(path.relative_to(saved_evaluation)): path.read_bytes()
        for path in saved_evaluation.rglob("*.json")
    }
    microduck.main(
        [
            "assess",
            "--run",
            str(saved_evaluation),
            "--output",
            str(output),
            "--max-planar-rmse",
            "3",
        ]
    )
    assert json.loads((output / "acceptance.json").read_text())["passed"]
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["input"]["status"] == "rejected"
    for item in manifest["input"]["files"]:
        snapshot = output / "inputs" / item["path"]
        assert snapshot.read_bytes() == before[item["path"]]
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == item["sha256"]
        assert (saved_evaluation / item["path"]).read_bytes() == before[item["path"]]
    # A later change to the original report cannot alter the retained evidence.
    (saved_evaluation / "seed-0/evaluation.json").write_text("{}")
    assert (output / "inputs/seed-0/evaluation.json").read_bytes() == before[
        "seed-0/evaluation.json"
    ]


def test_offline_single_run_rejection_and_overwrite_protection(
    saved_evaluation, tmp_path
):
    output = tmp_path / "assessment"
    args = [
        "assess",
        "--run",
        str(saved_evaluation / "seed-0"),
        "--output",
        str(output),
        "--max-planar-rmse",
        "2",
    ]
    with pytest.raises(SystemExit) as exc:
        microduck.main(args)
    assert exc.value.code == 3
    assert json.loads((output / "run.json").read_text())["status"] == "rejected"
    original = (output / "acceptance.json").read_bytes()
    with pytest.raises(SystemExit) as exc:
        microduck.main(args)
    assert exc.value.code == 1
    assert (output / "acceptance.json").read_bytes() == original


@pytest.mark.parametrize(
    "invalid",
    [
        "running",
        "missing_seed",
        "duplicate_seed",
        "source",
        "checkpoint",
        "report_seed",
        "velocity",
        "missing_runtime",
    ],
)
def test_offline_input_inconsistency_is_rejected(saved_evaluation, invalid):
    root = saved_evaluation
    filename = "run.json"
    if invalid in ("source", "checkpoint"):
        filename = (
            "seed-0/run.json" if invalid == "source" else "seed-0/evaluation.json"
        )
    if invalid in ("report_seed", "velocity"):
        filename = "seed-0/evaluation.json"
    data = json.loads((root / filename).read_text())
    if invalid == "running":
        data["status"] = "running"
    elif invalid == "missing_seed":
        data["completed_seeds"] = [0]
    elif invalid == "duplicate_seed":
        data["seeds"] = data["completed_seeds"] = [0, 0]
    elif invalid == "source":
        data["source"] = {"revision": "changed"}
    elif invalid == "checkpoint":
        data["checkpoint_metadata"]["sha256"] = "changed"
    elif invalid == "report_seed":
        data["seed"] = 5
    elif invalid == "velocity":
        data["conditions"]["velocity_body_frame"] = [0, 0, 0]
    else:
        (root / "seed-0/runtime.json").unlink()
    (root / filename).write_text(json.dumps(data))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_evaluation_run(root)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e999"])
def test_offline_json_rejects_nonfinite_numbers_anywhere(saved_evaluation, value):
    (saved_evaluation / "seed-0/runtime.json").write_text(
        '{"diagnostic": ' + value + "}"
    )
    with pytest.raises(ValueError):
        load_evaluation_run(saved_evaluation)


def test_offline_requires_explicit_threshold_before_creating_output(
    saved_evaluation, tmp_path
):
    output = tmp_path / "assessment"
    with pytest.raises(SystemExit) as exc:
        microduck.main(
            ["assess", "--run", str(saved_evaluation), "--output", str(output)]
        )
    assert exc.value.code == 1
    assert not output.exists()


def test_offline_recomputes_from_reports_instead_of_trusting_summary(saved_evaluation):
    (saved_evaluation / "summary.json").write_text(
        '{"passed": true, "pooled_velocity_rmse": [0,0,0]}'
    )
    reports, _, inputs = load_evaluation_run(saved_evaluation)
    assert not assess_evaluations(reports, {"max_planar_rmse": 2})["passed"]
    assert "summary.json" not in inputs


def test_negative_curriculum_override_is_rejected_before_run(batch):
    args, calls, _ = batch
    args.curriculum_step = -1
    with pytest.raises(ValueError, match="curriculum-step"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == []
    assert not args.output.exists()


def comparable(seed, *, checkpoint_hash="abc", rmse=3):
    result = report(seed, checkpoint_hash=checkpoint_hash, rmse=rmse)
    result["conditions"].update(curriculum_start_step=0, curriculum_source="override")
    return result


def test_comparison_pairs_seed_ids_instead_of_list_positions():
    before = [comparable(0, rmse=3), comparable(1, rmse=4)]
    after = [
        comparable(1, checkpoint_hash="new", rmse=2),
        comparable(0, checkpoint_hash="new", rmse=1),
    ]
    result = compare_evaluations(before, after)
    assert result["metrics"]["vx_rmse"]["mean_delta"] == -2
    assert result["metrics"]["vx_rmse"]["delta_sample_std"] == 0
    assert result["before_checkpoint"]["sha256"] == "abc"
    assert result["after_checkpoint"]["sha256"] == "new"
    assert [row["seed"] for row in result["per_seed"]] == [0, 1]


@pytest.mark.parametrize(
    "difference", ["curriculum", "precision", "seeds", "sampling", "legacy"]
)
def test_comparison_rejects_confounding_conditions(difference):
    before, after = comparable(0), comparable(0, checkpoint_hash="new")
    if difference == "curriculum":
        after["conditions"]["curriculum_start_step"] = 24000
    elif difference == "precision":
        after["conditions"]["policy_matmul_precision"] = "tf32"
    elif difference == "seeds":
        after["seed"] = 1
    elif difference == "sampling":
        after["velocity_tracking"]["sampling"] = "after-reset"
    else:
        del after["conditions"]["curriculum_start_step"]
    with pytest.raises(ValueError):
        compare_evaluations([before], [after])


def test_comparison_allows_each_checkpoints_own_onnx_export():
    before, after = comparable(0), comparable(0, checkpoint_hash="new")
    parity = {
        "sha256": "old-onnx",
        "provider": "CPU",
        "atol": 1e-4,
        "rtol": 1e-4,
        "sampling": "rotating",
    }
    before["onnx_parity"] = parity
    after["onnx_parity"] = dict(parity, sha256="new-onnx")
    assert (
        compare_evaluations([before], [after])["metrics"]["vx_rmse"]["mean_delta"] == 0
    )
    after["onnx_parity"]["atol"] = 1e-2
    with pytest.raises(ValueError, match="ONNX"):
        compare_evaluations([before], [after])


def test_comparison_cli_retains_both_input_snapshots(saved_evaluation, tmp_path):
    for seed in (0, 1):
        path = saved_evaluation / f"seed-{seed}/evaluation.json"
        result = json.loads(path.read_text())
        result["conditions"].update(
            curriculum_start_step=0, curriculum_source="checkpoint"
        )
        result["checkpoint_metadata"]["common_step_counter"] = 0
        path.write_text(json.dumps(result))
    output = tmp_path / "comparison"
    microduck.main(
        [
            "compare",
            "--before",
            str(saved_evaluation),
            "--after",
            str(saved_evaluation),
            "--output",
            str(output),
        ]
    )
    result = json.loads((output / "comparison.json").read_text())
    assert all(metric["mean_delta"] == 0 for metric in result["metrics"].values())
    assert (output / "inputs/before/seed-0/evaluation.json").read_bytes() == (
        output / "inputs/after/seed-0/evaluation.json"
    ).read_bytes()
    assert json.loads((output / "run.json").read_text())["status"] == "complete"


def test_offline_rejects_a_reported_curriculum_that_disagrees_with_checkpoint(
    saved_evaluation,
):
    path = saved_evaluation / "seed-0/evaluation.json"
    result = json.loads(path.read_text())
    result["conditions"].update(curriculum_start_step=0, curriculum_source="checkpoint")
    result["checkpoint_metadata"]["common_step_counter"] = 24000
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="curriculum"):
        load_evaluation_run(saved_evaluation)


def test_offline_cannot_pass_with_a_stale_planar_rmse():
    result = report(rmse=3)
    result["velocity_tracking"]["planar_velocity_rmse_m_s"] = 0
    with pytest.raises(ValueError, match="Planar RMSE"):
        assess_evaluations([result], {"max_planar_rmse": 0.1})


@pytest.mark.parametrize(
    "invalid", [None, "samples", "shape", "nan", "totals", "bias", "planar"]
)
def test_per_environment_details_must_agree_with_totals(invalid):
    result = report()
    tracking = result["velocity_tracking"]
    tracking["mae"] = [2, 0, 0]
    detail = tracking["per_environment"] = {
        "samples_per_environment": 4,
        "mean_command": [[0.2, 0, 0]] * 2,
        "mean_actual": [[0.1, 0, 0]] * 2,
        "bias": [[-0.1, 0, 0]] * 2,
        "mae": [[2, 0, 0]] * 2,
        "rmse": [[3, 0, 0]] * 2,
        "planar_velocity_rmse_m_s": [3, 3],
    }
    if invalid is None:
        assert aggregate_evaluations([result])["metrics"]["vx_rmse"]["mean"] == 3
        return
    if invalid == "samples":
        detail["samples_per_environment"] = 3
    elif invalid == "shape":
        detail["rmse"] = [[3, 0, 0]]
    elif invalid == "nan":
        detail["rmse"] = [[float("nan"), 0, 0]] * 2
    elif invalid == "totals":
        detail["mean_actual"] = [[0.2, 0, 0]] * 2
    elif invalid == "bias":
        detail["bias"] = [[0, 0, 0]] * 2
    else:
        detail["planar_velocity_rmse_m_s"] = [0, 0]
    with pytest.raises(ValueError):
        aggregate_evaluations([result])
