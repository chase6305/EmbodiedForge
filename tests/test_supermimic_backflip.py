"""A backflip preset must never silently select the walking model/data."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodiedforge import supermimic_run as runner
from embodiedforge.workflows import supermimic_backflip as backflip
from embodiedforge.workflows.supermimic_inputs import smoke_training_clips


def registry(workspace):
    checkpoint = workspace / "runs/backflip-test/last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"dedicated-backflip-test-model")
    directory = backflip.task_directory(workspace)
    directory.mkdir(parents=True)
    record = {
        "task": "backflip",
        "training_status": "complete",
        "checkpoint": str(checkpoint.relative_to(workspace)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (directory / "current.json").write_text(json.dumps(record))
    return checkpoint


def test_first_backflip_training_uses_no_external_policy(tmp_path):
    args = runner.parser().parse_args(["train", "--task", "backflip"])
    plan = runner.resolve_options(args, tmp_path)
    assert plan["fresh"] and plan["checkpoint"] is None
    assert plan["seed"] == 137 and plan["warmup"] == 500
    assert plan["hours"] == 1.5
    assert plan["motions"] == str(backflip.task_directory(tmp_path) / "data/train")


def test_backflip_play_has_no_walking_fallback(tmp_path):
    args = runner.parser().parse_args(["play", "--task", "backflip"])
    plan = runner.resolve_options(args, tmp_path)
    assert plan["checkpoint"] is None and plan["clip"] == "backflip"


def test_smoke_selects_existing_backflip_instead_of_filtering_for_walk():
    assert smoke_training_clips({"backflip.npz": "hash"}) == ["backflip"]
    assert smoke_training_clips(
        {"dance.npz": "hash", "walk1_subject1.npz": "hash"}
    ) == ["walk1_subject1"]


def test_backflip_uses_its_own_registered_model_and_extended_eval(tmp_path):
    checkpoint = registry(tmp_path)
    args = runner.parser().parse_args(["eval", "--task", "backflip"])
    plan = runner.resolve_options(args, tmp_path)
    assert plan["checkpoint"] == str(checkpoint)
    assert plan["motions"].endswith("backflip/data/eval")
    assert plan["num_envs"] == 1
    checkpoint.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.resolve_options(args, tmp_path)


def test_small_smoke_does_not_resume_the_full_width_model(tmp_path):
    registry(tmp_path)
    args = runner.parser().parse_args(
        ["train", "--task", "backflip", "--profile", "smoke"]
    )
    plan = runner.resolve_options(args, tmp_path)
    assert plan["fresh"] and plan["checkpoint"] is None and plan["warmup"] is None


def test_fresh_flag_starts_an_independent_run(tmp_path):
    registry(tmp_path)
    args = runner.parser().parse_args(["train", "--task", "backflip", "--fresh"])
    assert runner.resolve_options(args, tmp_path)["checkpoint"] is None


def test_explicit_model_and_preparation_do_not_require_a_valid_old_registry(tmp_path):
    checkpoint = registry(tmp_path)
    checkpoint.write_bytes(b"bad old registry")
    replacement = tmp_path / "new.pt"
    replacement.write_bytes(b"new trusted model")
    args = runner.parser().parse_args(
        ["play", "--task", "backflip", "--checkpoint", str(replacement)]
    )
    assert runner.resolve_options(args, tmp_path)["checkpoint"] == str(replacement)
    args = runner.parser().parse_args(["prepare", "--task", "backflip"])
    assert runner.resolve_options(args, tmp_path)["checkpoint"] is None


def test_custom_motion_cannot_be_registered_as_the_pinned_backflip(tmp_path, capsys):
    assert runner.main(["train", "--task", "backflip", "--motions", str(tmp_path)]) == 2
    assert "pinned reference" in capsys.readouterr().err


def test_short_training_is_rejected_before_sdk_load(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/supermimic_reproduce.py"),
            "--root",
            str(tmp_path),
            "--motions",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--iterations",
            "100",
            "--num-envs",
            "8",
            "--warmup-iterations",
            "500",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2 and "warmup" in result.stderr
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("angle", [-360.0, 360.0, 0.0])
def test_rotation_metric_keeps_full_turns_and_quaternion_sign_invariance(angle):
    half = np.deg2rad(np.linspace(0, angle, 301)) / 2
    q = np.column_stack(
        (np.cos(half), np.zeros_like(half), np.sin(half), np.zeros_like(half))
    )
    assert backflip.lateral_rotation_degrees(q) == pytest.approx(angle)
    q[::2] *= -1
    assert backflip.lateral_rotation_degrees(q) == pytest.approx(angle)


def test_rotation_metric_is_relative_to_initial_heading():
    half = np.linspace(0, np.pi, 301)
    # A90degree fixed yaw followed by a360degree lateral rotation.
    q = np.column_stack(
        (np.cos(half), -np.sin(half), np.sin(half), np.cos(half))
    ) / np.sqrt(2)
    assert backflip.lateral_rotation_degrees(q) == pytest.approx(360.0)


def pose_inputs(samples=130):
    joints = np.zeros((samples, 29))
    ankles = np.zeros((samples, 2, 3))
    ankles[:, 1, 1] = 0.4
    return joints, joints.copy(), ankles, ankles.copy()


def test_fixed_pose_window_keeps_pose_and_foot_spacing_failures_visible():
    actual, target, feet, reference_feet = pose_inputs()
    perfect = backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
    assert perfect["pose_quality_passed"] is True
    assert perfect["hip_knee_rmse_deg"] == 0
    actual[90:118, [0, 1, 2, 3, 6, 7, 8, 9]] = np.deg2rad(30)
    failed = backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
    assert failed["hip_knee_rmse_deg"] == pytest.approx(30)
    assert failed["pose_quality_passed"] is False
    actual[:] = target
    feet[90:118, 1, 1] = 0.6
    failed = backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
    assert failed["hip_knee_rmse_deg"] == 0
    assert failed["max_ankle_separation_m"] == pytest.approx(0.6)
    assert failed["reference_max_ankle_separation_m"] == pytest.approx(0.4)
    assert failed["pose_quality_passed"] is False


def test_fixed_pose_window_does_not_shift_to_a_better_pose():
    actual, target, feet, reference_feet = pose_inputs()
    actual[90:118, :12] = np.pi
    measured = backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
    assert measured["hip_knee_rmse_deg"] == pytest.approx(180)
    assert measured["pose_quality_passed"] is False
    # A mismatch outside the prescribed window does not enter this diagnostic.
    actual[:] = 0
    actual[:90] = np.pi
    assert (
        backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)[
            "pose_quality_passed"
        ]
        is True
    )


def test_fixed_pose_window_missing_after_early_failure_is_unassessed():
    actual, target, feet, reference_feet = pose_inputs()
    measured = backflip.fixed_window_pose_quality(
        actual[:117], target, feet[:117], reference_feet
    )
    assert measured["pose_quality_passed"] is None
    assert measured["hip_knee_rmse_deg"] is None


def test_leg_asymmetry_is_reference_adjusted():
    actual, target, feet, reference_feet = pose_inputs()
    actual[:, 1] = target[:, 1] = 0.3
    measured = backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
    assert measured["reference_adjusted_leg_asymmetry_rmse_deg"] == 0
    actual[90:118, 1] += 0.2
    measured = backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
    assert measured["reference_adjusted_leg_asymmetry_rmse_deg"] == pytest.approx(
        np.degrees(0.2 / np.sqrt(6))
    )


def test_pose_diagnostic_rejects_nonfinite_or_mismatched_samples():
    actual, target, feet, reference_feet = pose_inputs()
    with pytest.raises(ValueError, match="sample counts"):
        backflip.fixed_window_pose_quality(actual, target, feet[:-1], reference_feet)
    with pytest.raises(ValueError, match="N29"):
        backflip.fixed_window_pose_quality(actual[:, :28], target, feet, reference_feet)
    actual[90, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        backflip.fixed_window_pose_quality(actual, target, feet, reference_feet)
