import numpy as np
import pytest

from embodiedforge._wuji_diagnostics import TrialDiagnostics, quaternion_angle


def test_requested_video_must_exist(tmp_path):
    from embodiedforge._wuji_diagnostics import verify_video

    with pytest.raises(ValueError, match="no requested video"):
        verify_video({"record_video": None}, tmp_path)


def test_video_validation_requires_all_trial_frames(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from embodiedforge import _wuji_diagnostics as module

    (tmp_path / "evaluation.mp4").write_bytes(b"fixture")
    stream = {
        "width": 640,
        "height": 368,
        "r_frame_rate": "20/1",
        "nb_read_frames": "3",
    }
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout=json.dumps({"streams": [stream]}), stderr=""
        ),
    )
    result = {
        "record_video": "evaluation.mp4",
        "diagnostics": {"trials": [{"steps": 2}]},
    }
    assert module.verify_video(result, tmp_path)["nb_read_frames"] == "3"
    stream["nb_read_frames"] = "2"
    with pytest.raises(ValueError, match="frame count mismatch"):
        module.verify_video(result, tmp_path)


def evaluation_fixture():
    from embodiedforge._wuji_diagnostics import validate_evaluation

    audit = TrialDiagnostics()
    audit.start([0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1])
    audit.update([0, 0, 0], [1, 0, 0, 0], np.zeros(20))
    request = {"seed": 0, "steps": 1, "num_trials": 1, "policy": "zero"}
    result = {
        "protocol": {
            "seed": 0,
            "environment_seed": 0,
            "policy": "zero",
            "control_dt_s": 0.05,
            "trial_timeout_s": 0.05,
        },
        "num_trials": 1,
        "trials": [
            {
                "trial_idx": 0,
                "status": "timeout",
                "goal_reaches": 0,
                "time_to_first_success_s": None,
                "final_orientation_error_rad": np.pi,
                "min_orientation_error_rad": np.pi,
            }
        ],
        "diagnostics": audit.report(),
        "success_rate": 0,
        "drop_rate": 0,
        "timeout_rate": 1,
    }
    validate_evaluation(request, result)
    return request, result


@pytest.mark.parametrize(
    "kind", ["policy", "rate", "index", "steps", "action", "success", "error"]
)
def test_evaluation_rejects_inconsistent_trial_accounting(kind):
    from embodiedforge._wuji_diagnostics import validate_evaluation

    request, result = evaluation_fixture()
    if kind == "policy":
        result["protocol"]["policy"] = "trained"
    elif kind == "rate":
        result["success_rate"] = 1
    elif kind == "index":
        result["trials"][0]["trial_idx"] = 1
    elif kind == "steps":
        result["diagnostics"]["trials"][0]["steps"] = 0
    elif kind == "action":
        result["diagnostics"]["trials"][0]["action_rms"] = 0.1
    elif kind == "success":
        result["trials"][0]["goal_reaches"] = 1
    else:
        result["trials"][0]["final_orientation_error_rad"] = float("nan")
    with pytest.raises(ValueError):
        validate_evaluation(request, result)


def test_quaternion_angle_handles_sign_scale_and_rejects_invalid():
    assert quaternion_angle([1, 0, 0, 0], [-2, 0, 0, 0]) == 0
    assert quaternion_angle([1, 0, 0, 0], [0, 1, 0, 0]) == pytest.approx(np.pi)
    with pytest.raises(ValueError):
        quaternion_angle([0, 0, 0, 0], [1, 0, 0, 0])


def test_motion_and_progress_are_distinct_from_action_magnitude():
    audit = TrialDiagnostics()
    initial, goal = [1, 0, 0, 0], [0, 0, 0, 1]
    audit.start([0, 0, 0], initial, goal)
    audit.update([0, 0, 0], initial, np.ones(20))
    audit.update([0, 0, 0], initial, np.ones(20))
    first = audit.report()["trials"][0]
    assert first["action_rms"] == 1
    assert first["action_delta_rms"] == 0
    assert first["max_cube_rotation_rad"] == 0
    assert first["best_error_reduction_rad"] == 0


def test_trial_boundary_does_not_count_action_reset_as_jerk():
    audit = TrialDiagnostics()
    initial = np.array([1.0, 0, 0, 0])
    audit.start([0, 0, 0], initial, [0, 0, 0, 1])
    initial[:] = 0  # Initial poses must be owned snapshots.
    audit.update([0, 0, 0], [1, 0, 0, 0], np.ones(20))
    audit.start([0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1])
    audit.update([0.1, 0, 0], [2**-0.5, 0, 0, 2**-0.5], -np.ones(20))
    audit.update([0.1, 0, 0], [2**-0.5, 0, 0, 2**-0.5], -np.ones(20))
    rows = audit.report()["trials"]
    assert rows[0]["action_delta_rms"] is None
    assert rows[1]["action_delta_rms"] == 0
    assert rows[1]["max_cube_rotation_rad"] == pytest.approx(np.pi / 2)
    assert rows[1]["final_error_reduction_rad"] == pytest.approx(np.pi / 2)
    assert rows[1]["max_cube_displacement_m"] == pytest.approx(0.1)
    assert audit.report()["trials"] == rows  # Re-reading must not append again.


def test_incomplete_and_nonfinite_trials_fail():
    audit = TrialDiagnostics()
    with pytest.raises(ValueError, match="No active"):
        audit.update([0, 0, 0], [1, 0, 0, 0], np.zeros(20))
    audit.start([0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1])
    with pytest.raises(ValueError, match="no physics steps"):
        audit.finish()
    with pytest.raises(ValueError, match="finite"):
        audit.update([0, 0, 0], [1, 0, 0, 0], np.full(20, np.nan))
