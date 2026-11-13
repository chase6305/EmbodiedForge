"""Missing/failed physical measurements must never turn into a quality pass."""

import pytest

from embodiedforge.workflows.supermimic_backflip_quality import combine_quality


def measured_inputs():
    digest = "a" * 64
    return {
        "complete": True,
        "initial_velocity": 0.0,
        "native_return": 0.9,
        "control_steps": 570,
        "state_samples": 571,
        "diagnostics": {
            "basic_checks_passed": True,
            "fixed_window_pose": {"pose_quality_passed": True},
        },
        "geometry": {
            "model_sha256": digest,
            "initial_pose_max_abs_error": 0.0,
            "all_geometry_checks_passed": True,
            "minimum_up_airborne": -1.0,
            "fully_airborne_seconds": 0.6,
            "relative_pitch_rotation_rad": -6.2,
        },
        "forces": {
            "compiled_cpu_geometry_model_sha256": digest,
            "force_balance_verified": True,
            "capacity_overflow_substeps": 0,
            "nonfoot_force_substeps": 0,
            "nonfoot_negative_distance_substeps": 0,
            "samples": 2280,
        },
    }


def test_complete_ported_quality_pass_is_not_claimed_as_v14_equivalence():
    result = combine_quality(**measured_inputs())
    assert result["ported_quality_passed"] is True
    assert not result["failed_checks"] and not result["unassessed_checks"]
    assert result["strict_v14_equivalence"] is False
    assert result["physical_robot_tested"] is False


def test_unrecorded_substep_contacts_are_unassessed_rather_than_zero():
    values = measured_inputs()
    del values["forces"]["nonfoot_negative_distance_substeps"]
    result = combine_quality(**values)
    assert result["ported_quality_passed"] is None
    assert "no200Hznegative_nonfoot_contact_distance" in result["unassessed_checks"]


def test_pose_failure_remains_a_failure_when_another_measurement_is_missing():
    values = measured_inputs()
    values["diagnostics"]["fixed_window_pose"]["pose_quality_passed"] = False
    del values["forces"]["nonfoot_negative_distance_substeps"]
    result = combine_quality(**values)
    assert result["ported_quality_passed"] is False
    assert "shared_fixed_window_leg_pose" in result["failed_checks"]
    assert result["unassessed_checks"]


@pytest.mark.parametrize(
    "key",
    [
        "capacity_overflow_substeps",
        "nonfoot_force_substeps",
        "nonfoot_negative_distance_substeps",
    ],
)
def test_one_bad_physics_substep_prevents_acceptance(key):
    values = measured_inputs()
    values["forces"][key] = 1
    assert combine_quality(**values)["ported_quality_passed"] is False


def test_incomplete_force_coverage_and_wrong_geometry_model_are_rejected():
    values = measured_inputs()
    values["forces"]["samples"] = 2276
    assert combine_quality(**values)["ported_quality_passed"] is False
    values["forces"]["samples"] = 2280
    values["forces"]["compiled_cpu_geometry_model_sha256"] = "b" * 64
    result = combine_quality(**values)
    assert result["ported_quality_passed"] is False
    assert "actual_physics_model_geometry_identity" in result["failed_checks"]


def test_no_flight_cannot_pass_even_with_a_good_standing_pose():
    values = measured_inputs()
    values["geometry"].update(
        minimum_up_airborne=None,
        fully_airborne_seconds=0.0,
        all_geometry_checks_passed=False,
    )
    assert combine_quality(**values)["ported_quality_passed"] is False


@pytest.mark.parametrize("native_return", [float("nan"), float("inf"), 0.74])
def test_invalid_or_low_native_return_cannot_pass(native_return):
    values = measured_inputs()
    values["native_return"] = native_return
    assert combine_quality(**values)["ported_quality_passed"] is False


def test_wrong_initial_pose_is_not_accepted_as_a_fresh_start():
    values = measured_inputs()
    values["geometry"]["initial_pose_max_abs_error"] = 0.001
    assert combine_quality(**values)["ported_quality_passed"] is False
