"""Verify recorder coverage against actual simulator time, including bad clocks."""

import numpy as np
import pytest

from embodiedforge.workflows.supermimic_contacts import (
    ContactRecorder,
    validate_substep_clock,
)


def test_device_float32_clock_accumulation_is_accepted():
    # The actual device accumulates this clock, rather than multiplying a step ID.
    time = np.float32(0)
    times = []
    for _ in range(2280):
        time += np.float32(0.005)
        times.append(time)
    actual = validate_substep_clock(times, 0.005)
    assert actual["actual_clock_verified"]
    assert actual["actual_last_time_s"] == float(time)
    assert actual["actual_max_step_error_s"] < 1e-6


@pytest.mark.parametrize(
    "times", [[0.005, 0.005], [0.005, 0.015], [0.005, 0.0], [0.005, float("nan")], []]
)
def test_duplicate_skipped_reversed_or_nonfinite_physics_clock_fails(times):
    with pytest.raises(ValueError, match="clock|timestamps"):
        validate_substep_clock(times, 0.005)


def test_saved_geometry_hash_does_not_materialize_the_model_bytes(
    tmp_path, monkeypatch
):
    import hashlib
    from pathlib import Path

    mujoco = pytest.importorskip("mujoco")
    recorder = ContactRecorder.__new__(ContactRecorder)
    recorder.model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/>"
        '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
    )
    recorder.dt = 0.005
    recorder.iteration_flags = 1
    recorder.debug_contacts, recorder.nonfoot_debug = [], []
    recorder.samples = [
        {
            "actual_data_time_s": (i + 1) * recorder.dt,
            "total_ground_force_N": np.zeros(3),
            "root_constraint_N": np.zeros(3),
            "nonfoot_force_N": np.zeros(3),
            "foot_forces_N": np.zeros((2, 3)),
            "nonfoot_force_contacts": 0,
            "nonfoot_negative_distance_contacts": 0,
            "max_nonfoot_depth_m": 0.0,
            "overflow": 0,
        }
        for i in range(3)
    ]

    def reject_whole_file_read(path):
        raise AssertionError(f"Whole-file read during contact audit: {path}")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)
    report = recorder.save(tmp_path)
    with (tmp_path / "physics-model.mjb").open("rb") as stream:
        expected = hashlib.sha256(stream.read()).hexdigest()
    assert report["compiled_cpu_geometry_model_sha256"] == expected
    assert report["actual_clock_verified"]
