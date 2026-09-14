from copy import deepcopy

import numpy as np
import pytest

from embodiedforge._go1_metrics import (
    SWITCHING_SUITES,
    SwitchingMetrics,
    evaluation_commands,
    switching_schedule,
    validate_switching,
)
from embodiedforge.recipes import assess


@pytest.mark.parametrize("fall_at", [None, 50, 99])
@pytest.mark.parametrize("suite", SWITCHING_SUITES[1:])
def test_switching_schedule_tracking_and_no_reentry(fall_at, suite):
    metrics = SwitchingMetrics(2, 300, suite)
    assert evaluation_commands(suite) == {suite: None}
    expected = [
        [0, 0, 0],
        [0, 0.3, 0],
        [0, -0.3, 0],
        [-0.5, 0, 0],
        [0.5, 0.3, 0],
        [0, 0, 0],
    ]
    if suite in ("maneuver-switching", "maneuver-switching-mirrored"):
        expected = [
            [0, 0, 0],
            [-0.5, -0.3, 0],
            [0, 0.3, 0.5],
            [0, -0.3, -0.5],
            [0, 0, 0.8],
            [0, 0, 0],
        ]
        if suite == "maneuver-switching-mirrored":
            expected = [[vx, -vy, -yaw] for vx, vy, yaw in expected]
    assert [row["command"] for row in metrics.schedule] == expected
    for step in range(300):
        command = metrics.command()
        metrics.update(
            command, command, np.array([step == fall_at, False]), np.zeros(2, bool)
        )
    report = metrics.report()
    validate_switching(report, 300, suite)
    assert report["planar_rmse"] == report["yaw_rmse"] == 0
    assert report["survival_fraction"] == (1 if fall_at is None else 0.5)
    assert report["tracking_fraction"] == (1 if fall_at is None else 0.5)
    if fall_at is not None:
        assert report["per_env"]["observed_steps"][0] == fall_at + 1
        assert report["segments"][2]["per_env"]["observed_steps"][0] == 0
    for other_suite in SWITCHING_SUITES:
        if other_suite != suite:
            with pytest.raises(ValueError, match="schedule"):
                validate_switching(report, 300, other_suite)
    tampered = deepcopy(report)
    tampered["segments"][2]["command"][1] *= -1
    with pytest.raises(ValueError, match="schedule|identity"):
        validate_switching(tampered, 300, suite)
    with pytest.raises(ValueError, match="switching suite"):
        switching_schedule(300, "unknown")


def finish(metrics, velocity=None):
    while metrics.active.any() and metrics.steps < metrics.total.limit:
        command = metrics.command()
        measured = (
            command if velocity is None else np.broadcast_to(velocity, command.shape)
        )
        metrics.update(
            measured,
            command,
            np.zeros(len(command), bool),
            np.zeros(len(command), bool),
        )
    report = metrics.report()
    validate_switching(report, metrics.total.limit)
    return report


def test_switching_boundaries_and_perfect_tracking():
    metrics = SwitchingMetrics(2, 300)
    result = finish(metrics)
    assert metrics.schedule[1]["start_step"] == 50
    assert metrics.schedule[1]["end_step"] == 100
    assert result["planar_rmse"] == 0
    assert all(
        s["settled_fraction"] == 1 and s["mean_settle_time_s"] == 0.5
        for s in result["segments"]
    )
    acceptance = assess(result, {"max_planar_rmse": 0.1, "min_survival_fraction": 1})
    assert acceptance["passed"] and len(acceptance["checks"]) == 14


def test_a_fall_never_reenters_later_segments_and_missing_data_fails_acceptance():
    metrics = SwitchingMetrics(2, 300)
    command = metrics.command()
    metrics.update(command, command, np.ones(2, bool), np.zeros(2, bool))
    result = metrics.report()
    validate_switching(result, 300)
    assert result["segments"][0]["observed_frames"] == 2
    assert result["segments"][1]["observed_frames"] == 0
    assert result["segments"][1]["planar_rmse"] is None
    acceptance = assess(result, {"max_planar_rmse": 0.3})
    assert not acceptance["passed"]
    assert acceptance["checks"][2]["actual"] is None


def test_settling_requires_contiguous_hold_and_excludes_terminal_frame():
    metrics = SwitchingMetrics(2, 300)
    for step in range(50):
        command = metrics.command()
        velocity = command.copy()
        if step == 24:
            velocity[0, 0] = 0.2
        fell = np.array([False, step == 24])
        metrics.update(velocity, command, fell, np.zeros(2, bool))
    result = finish(metrics)
    row = result["segments"][0]
    assert row["per_env"]["settle_time_s"] == [1.0, None]
    assert row["settled_fraction"] == 0.5
    assert all(s["entered_count"] == 1 for s in result["segments"][1:])


def test_wrong_command_is_rejected_before_advancing_metrics():
    metrics = SwitchingMetrics(1, 300)
    with pytest.raises(ValueError, match="Observed command"):
        metrics.update(
            np.zeros((1, 3)), np.ones((1, 3)), np.zeros(1, bool), np.zeros(1, bool)
        )
    assert metrics.steps == 0


def test_response_criterion_catches_persistent_error_hidden_by_loose_rmse():
    metrics = SwitchingMetrics(2, 300)
    while metrics.steps < 300:
        command = metrics.command()
        velocity = command + [0.11, 0, 0]
        metrics.update(velocity, command, np.zeros(2, bool), np.zeros(2, bool))
    result = metrics.report()
    validate_switching(result, 300)
    assert assess(result, {"max_planar_rmse": 0.3})["passed"]
    response = assess(result, {"min_settled_fraction": 0.8})
    assert not response["passed"]
    assert len(response["checks"]) == 7
    assert all(check["actual"] == 0 for check in response["checks"])


@pytest.mark.parametrize("kind", ["order", "count", "coverage", "settled", "survival"])
def test_report_validation_rejects_inconsistent_segments(kind):
    report = deepcopy(finish(SwitchingMetrics(2, 300)))
    row = report["segments"][1]
    if kind == "order":
        row["start_step"] += 1
    elif kind == "count":
        row["per_env"]["observed_steps"][0] -= 1
    elif kind == "coverage":
        row["observed_frames"] -= 1
    elif kind == "settled":
        row["settled_fraction"] = 0
    else:
        row["survival_fraction"] = 0
    with pytest.raises(ValueError):
        validate_switching(report, 300)


def test_switching_requires_complete_segments():
    for steps in (0, 299, 500):
        with pytest.raises(ValueError):
            switching_schedule(steps)


def test_tracking_fraction_detects_drift_after_initial_settling():
    metrics = SwitchingMetrics(2, 300)
    while metrics.steps < 300:
        command = metrics.command()
        velocity = command.copy()
        if metrics.steps % 50 >= 25:
            velocity += [0.2, 0, 0]
        metrics.update(velocity, command, np.zeros(2, bool), np.zeros(2, bool))
    result = metrics.report()
    validate_switching(result, 300)
    assert result["settled_fraction"] == 1
    assert result["tracking_fraction"] == 0.5
    assert all(
        row["per_env"]["tracking_steps"] == [25, 25] for row in result["segments"]
    )
    assert not assess(result, {"min_tracking_fraction": 0.9})["passed"]


def test_tracking_denominator_keeps_fallen_environments_and_unobserved_time():
    metrics = SwitchingMetrics(2, 300)
    for step in range(50):
        command = metrics.command()
        metrics.update(
            command, command, np.array([step == 24, False]), np.zeros(2, bool)
        )
    result = finish(metrics)
    assert result["segments"][0]["per_env"]["tracking_steps"] == [24, 50]
    assert result["segments"][0]["tracking_fraction"] == 0.74
    assert all(row["tracking_fraction"] == 0.5 for row in result["segments"][1:])
    assert result["tracking_fraction"] == 0.5


@pytest.mark.parametrize(
    "kind",
    ["missing", "negative", "float", "too_many", "fraction", "overall", "protocol"],
)
def test_tracking_validation_rejects_inconsistent_or_partial_reports(kind):
    report = finish(SwitchingMetrics(2, 300))
    row = report["segments"][0]
    if kind == "missing":
        row["per_env"].pop("tracking_steps")
    elif kind == "negative":
        row["per_env"]["tracking_steps"][0] = -1
    elif kind == "float":
        row["per_env"]["tracking_steps"][0] = 50.0
    elif kind == "too_many":
        row["per_env"]["tracking_steps"][0] = 51
    elif kind == "fraction":
        row["tracking_fraction"] = 0.9
    elif kind == "overall":
        report["tracking_fraction"] = 0.9
    else:
        report.pop("tracking_protocol")
    with pytest.raises(ValueError, match="tracking"):
        validate_switching(report, 300)


def test_legacy_switching_reports_remain_readable():
    report = finish(SwitchingMetrics(2, 300))
    report.pop("tracking_fraction")
    report.pop("tracking_protocol")
    for row in report["segments"]:
        row.pop("tracking_fraction")
        row["per_env"].pop("tracking_steps")
    validate_switching(report, 300)


def test_extended_commands_include_basic_and_both_directions_without_shared_state():
    from embodiedforge._go1_metrics import evaluation_commands

    basic = evaluation_commands("basic")
    extended = evaluation_commands("extended")
    assert len(extended) == 10
    assert all(extended[key] == value for key, value in basic.items())
    assert extended["backward"] == [-0.5, 0.0, 0.0]
    assert extended["fast_forward"] == [1.0, 0.0, 0.0]
    np.testing.assert_array_equal(
        extended["strafe_left"], -np.array(extended["strafe_right"])
    )
    np.testing.assert_array_equal(
        extended["spin_left"], -np.array(extended["spin_right"])
    )
    extended["forward"][0] = 99
    assert evaluation_commands("basic")["forward"][0] == 0.5
    assert evaluation_commands("switching") == {"switching": None}
    with pytest.raises(ValueError, match="suite"):
        evaluation_commands("unknown")


def test_segment_rmse_must_pool_to_the_reported_overall_error():
    report = finish(SwitchingMetrics(2, 300))
    report["segments"][1]["yaw_rmse"] = 0.1
    with pytest.raises(ValueError, match="pool"):
        validate_switching(report, 300)


def test_settled_environment_needs_at_least_a_hold_of_tracking_steps():
    report = finish(SwitchingMetrics(2, 300))
    row = report["segments"][0]
    row["per_env"]["tracking_steps"] = [0, 50]
    row["tracking_fraction"] = 0.5
    report["tracking_fraction"] = 0.5
    with pytest.raises(ValueError, match="lacks tracking"):
        validate_switching(report, 300)


def test_settle_time_cannot_include_the_terminal_frame():
    metrics = SwitchingMetrics(1, 300)
    for step in range(25):
        command = metrics.command()
        metrics.update(command, command, np.array([step == 24]), np.zeros(1, bool))
    report = metrics.report()
    report["segments"][0]["per_env"]["settle_time_s"] = [0.5]
    report["segments"][0]["settled_fraction"] = 1
    report["segments"][0]["mean_settle_time_s"] = 0.5
    with pytest.raises(ValueError, match="settle time"):
        validate_switching(report, 300)
