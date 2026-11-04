"""Reference-informed backflip geometry and solved-force acceptance.

Thresholds and airborne geometry follow the pinned backflip reference evaluator
at5b313d75e804cdd2ff1ce9bdcfa899a98eca856b, with additional conservative
fixed-window and200Hz checks. See LICENSE.supermimic-backflip. This is a mjlab
port with different assets and571timestamps, not exact IsaacLab3V14 acceptance.
"""

import math

from embodiedforge.runtime.hashing import snapshot_file_hashes


def airborne_geometry(state, reference, model_path):
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation

    state = np.asarray(state, dtype=np.float64)
    if (
        state.ndim != 2
        or state.shape[1] != 71
        or not len(state)
        or not np.isfinite(state).all()
    ):
        raise ValueError("Expected finite nonempty actual N71 simulator states")
    model_sha = snapshot_file_hashes(model_path)[0]
    model = mujoco.MjModel.from_binary_path(str(model_path))
    if model.nq != 36 or model.nv != 35:
        raise ValueError("Backflip geometry requires the actual29jointG1 model")
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    if len(planes) != 1:
        raise ValueError("Backflip geometry requires one flat ground plane")
    floor = int(planes[0])
    feet, ankle_ids = [], []
    for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bodies = [i for i in range(model.nbody) if model.body(i).name.endswith(name)]
        if len(bodies) != 1:
            raise ValueError(f"Cannot identify one {name}")
        ankle_ids.append(bodies[0])
        group = np.flatnonzero(model.geom_bodyid == bodies[0]).tolist()
        if not group:
            raise ValueError("Missing foot geometry")
        feet.append(group)
    all_feet = set(feet[0]) | set(feet[1])
    with np.load(reference, allow_pickle=False) as data:
        target_joints = data["joint_pos"].astype(np.float64)
        names = list(map(str, data["body_names"]))
        target_feet = data["body_pos_w"][
            :,
            [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")],
        ].astype(np.float64)
        initial_target = np.concatenate(
            (data["body_pos_w"][0, 0], data["body_quat_w"][0, 0], target_joints[0])
        )
    if len(target_joints) != 571:
        raise ValueError("Physical backflip evaluation requires571reference timestamps")
    quat_error = min(
        float(np.abs(state[0, 3:7] - initial_target[3:7]).max()),
        float(np.abs(state[0, 3:7] + initial_target[3:7]).max()),
    )
    initial_error = max(
        quat_error,
        float(np.abs(state[0, :3] - initial_target[:3]).max()),
        float(np.abs(state[0, 7:36] - initial_target[7:]).max()),
    )
    data = mujoco.MjData(model)
    air, bad, separation = [], [], []
    for row in state:
        data.qpos[:], data.qvel[:] = row[:36], row[36:]
        mujoco.mj_forward(
            model, data
        )  # Geometry only; forces come from the device recorder.
        gaps = [
            min(
                mujoco.mj_geomDistance(model, data, geom, floor, 10, None)
                for geom in group
            )
            for group in feet
        ]
        collision = False
        for contact in data.contact:
            one, two = map(int, contact.geom)
            if contact.dist < 0 and (
                (one == floor and two not in all_feet)
                or (two == floor and one not in all_feet)
            ):
                collision = True
        air.append(min(gaps) > 0.03)
        bad.append(collision)
        separation.append(
            float(np.linalg.norm(data.xpos[ankle_ids[0]] - data.xpos[ankle_ids[1]]))
        )
    rotations = Rotation.from_quat(state[:, [4, 5, 6, 3]])
    matrix = rotations.as_matrix()
    phase = np.unwrap(
        np.arctan2(
            -(matrix[:, :, 0] @ matrix[0, :, 2]), matrix[:, :, 0] @ matrix[0, :, 0]
        )
    )
    up = matrix[:, 2, 2]
    native = min(321, len(state))
    full_air = np.asarray(air[:native]) & ~np.asarray(bad[:native])
    edges = np.diff(np.r_[False, full_air, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    result = {
        "recording_hz": 50,
        "model_sha256": model_sha,
        "model_source": str(model_path),
        "initial_pose_max_abs_error": initial_error,
        "whole_reference_sagittal_rotation_deg": float(
            np.degrees(phase[native - 1] - phase[0])
        ),
        "fully_airborne_seconds": float(full_air.sum() / 50),
        "longest_airborne_window_start_inclusive_end_exclusive": None,
        "aerial_rotation_deg": None,
        "aerial_axis_min_alignment": None,
        "aerial_hip_knee_rmse_deg": None,
        "aerial_max_ankle_separation_m": None,
        "same_window_reference_max_ankle_separation_m": None,
        "minimum_up_airborne": float(up[:native][full_air].min())
        if full_air.any()
        else None,
        "relative_pitch_rotation_rad": float(
            (rotations[:-1].inv() * rotations[1:]).as_rotvec()[:320, 1].sum()
        ),
        "nonfoot_floor_geometry_frames": int(sum(bad)),
        "hold_nonfoot_geometry_frames": int(sum(bad[321:])),
        "scope": "Actual longest geometric-airborne window in the first321states, foot gap>.03m. CPU geometry does not generate contact forces;200Hzactual forces are assessed separately.",
    }
    if len(starts):
        index = int(np.argmax(ends - starts))
        start, end = int(starts[index]), int(ends[index])
        ids = [0, 1, 2, 3, 6, 7, 8, 9]
        result.update(
            longest_airborne_window_start_inclusive_end_exclusive=[start, end],
            aerial_rotation_deg=float(np.degrees(phase[end - 1] - phase[start])),
            aerial_axis_min_alignment=float(
                (matrix[start:end, :, 1] @ matrix[0, :, 1]).min()
            ),
            aerial_hip_knee_rmse_deg=float(
                np.degrees(
                    np.sqrt(
                        np.mean(
                            (
                                state[start:end, 7:36][:, ids]
                                - target_joints[start:end, ids]
                            )
                            ** 2
                        )
                    )
                )
            ),
            aerial_max_ankle_separation_m=max(separation[start:end]),
            same_window_reference_max_ankle_separation_m=float(
                np.linalg.norm(
                    target_feet[start:end, 0] - target_feet[start:end, 1], axis=1
                ).max()
            ),
        )
    result["geometry_quality_checks"] = {
        "full_backward_turn": bool(
            len(state) >= 321
            and -370 <= result["whole_reference_sagittal_rotation_deg"] <= -350
        ),
        "airborne_rotation": bool(
            result["aerial_rotation_deg"] is not None
            and result["aerial_rotation_deg"] <= -250
            and result["aerial_axis_min_alignment"] > 0.8
        ),
        "dynamic_leg_posture": bool(
            result["aerial_hip_knee_rmse_deg"] is not None
            and result["aerial_hip_knee_rmse_deg"] < 25
            and result["aerial_max_ankle_separation_m"]
            <= result["same_window_reference_max_ankle_separation_m"] + 0.15
        ),
        "no_nonfoot_geometry": not any(bad),
    }
    result["all_geometry_checks_passed"] = all(
        result["geometry_quality_checks"].values()
    )
    if snapshot_file_hashes(model_path)[0] != model_sha:
        raise ValueError("Actual geometry model changed during analysis")
    return result


def combine_quality(
    *,
    complete,
    initial_velocity,
    native_return,
    control_steps,
    state_samples,
    diagnostics,
    geometry,
    forces,
):
    def zero(key):
        value = forces.get(key)
        return None if value is None else value == 0

    model_sha = geometry.get("model_sha256")
    expected_sha = forces.get("compiled_cpu_geometry_model_sha256")
    checks = {
        "complete_reference_and_extra_hold": complete,
        "zero_initial_velocity": initial_velocity is not None
        and math.isfinite(initial_velocity)
        and initial_velocity < 1e-6,
        "correct_initial_pose": None
        if geometry.get("initial_pose_max_abs_error") is None
        else geometry["initial_pose_max_abs_error"] < 1e-5,
        "shared_rotation_geometry_hold": diagnostics.get("basic_checks_passed"),
        "shared_fixed_window_leg_pose": diagnostics.get("fixed_window_pose", {}).get(
            "pose_quality_passed"
        ),
        "dynamic_geometry": geometry.get("all_geometry_checks_passed"),
        "inverted_while_airborne": geometry.get("minimum_up_airborne") is not None
        and geometry["minimum_up_airborne"] < -0.6,
        "at_least60msairborne": geometry.get("fully_airborne_seconds", 0) >= 0.06,
        "relative_pitch_below_minus5rad": geometry.get("relative_pitch_rotation_rad", 0)
        < -5,
        "native320step_tracking_return": native_return is not None
        and math.isfinite(native_return)
        and native_return >= 0.75,
        "solved_force_balance": forces.get("force_balance_verified"),
        "no_contact_capacity_overflow": zero("capacity_overflow_substeps"),
        "no_nonfoot_force": zero("nonfoot_force_substeps"),
        "no200Hznegative_nonfoot_contact_distance": zero(
            "nonfoot_negative_distance_substeps"
        ),
        "actual_physics_model_geometry_identity": None
        if model_sha is None or expected_sha is None
        else model_sha == expected_sha,
        "complete200Hzforce_coverage": forces.get("samples") == 2280
        and control_steps == 570
        and state_samples == 571,
    }
    checks = {
        key: None if value is None else bool(value) for key, value in checks.items()
    }
    failed = [key for key, value in checks.items() if value is False]
    missing = [key for key, value in checks.items() if value is None]
    return {
        "checks": checks,
        "failed_checks": failed,
        "unassessed_checks": missing,
        "ported_quality_passed": False if failed else (None if missing else True),
        "strict_v14_equivalence": False,
        "physical_robot_tested": False,
        "scope": "Reference-informed thresholds on current mjlab G1, with conservative fixed-window/200Hzgeometry/model-identity checks. One evaluation does not establish independent-training robustness or exact IsaacLab3V14 reproduction.",
    }
