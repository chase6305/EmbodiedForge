"""Backflip reference preparation for SuperMimic/mjlab, not an iLQR controller.

Recipe reviewed from liudi159/supermimic-backflip-5090 at 5b313d75e804cdd2ff1ce9bdcfa899a98eca856b.
The Apache-2.0 mjbatch file supplies motion targets only. No trained weights
or reference poses are substituted for the simulated policy state.
"""

import json
import os
import shutil
import urllib.request
from pathlib import Path
from uuid import uuid4

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import copy_file_snapshot, snapshot_file_hashes
from embodiedforge.workflows.supermimic_inputs import inspect_motion_files

SOURCE_REVISION = "77966f85bcd8f7ef4351cb4a1a6f42e133d19725"
SOURCE_URL = f"https://raw.githubusercontent.com/kevinzakka/mjbatch/{SOURCE_REVISION}/examples/assets/flip.npz"
SOURCE_SHA256 = "f9ad1275f978bf891ee141e272d8279eb56685e4209f42d5cd7dc23529bcb4d6"
CSV_SHA256 = "0d7dcb9ac79834bec3f3db16b5a0d8e582b71b3854435d09ad3b886631adb007"
REFERENCE_REVISION = "5b313d75e804cdd2ff1ce9bdcfa899a98eca856b"


def task_directory(workspace):
    return Path(workspace) / "runs/supermimic/tasks/backflip"


def prepared(directory):
    directory = Path(directory)
    record = json.loads((directory / "provenance.json").read_text())
    if record["source_sha256"] != SOURCE_SHA256 or record["csv_sha256"] != CSV_SHA256:
        raise ValueError("Backflip reference identity differs from the pinned recipe")
    if snapshot_file_hashes(directory / "backflip.csv")[0] != CSV_SHA256:
        raise ValueError("Backflip canonical CSV changed")
    for phase in ("train", "eval"):
        inspect_motion_files(directory / phase, record[phase + "_files"])
    return record


def latest_checkpoint(workspace, *, recipe="cuboi"):
    """Read one recipe's registry; weights never cross implementation formats."""
    if recipe not in {"cuboi", "reference"}:
        raise ValueError(f"Unknown backflip recipe: {recipe}")
    directory = task_directory(workspace)
    path = directory / (
        "reference/current.json" if recipe == "reference" else "current.json"
    )
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    if (
        record.get("task") != "backflip"
        or record.get("training_status") != "complete"
        or record.get("recipe", "cuboi") != recipe
    ):
        raise ValueError(f"Invalid {recipe} backflip checkpoint registry")
    checkpoint = Path(workspace) / record["checkpoint"]
    if snapshot_file_hashes(checkpoint)[0] != record["checkpoint_sha256"]:
        raise ValueError(f"{recipe} backflip checkpoint changed")
    return {**record, "checkpoint": str(checkpoint)}


def register_checkpoint(workspace, output, worker):
    checkpoint = Path(output) / "training/training/ckpt_latest.pt"
    digest = snapshot_file_hashes(checkpoint)[0]
    if (
        worker["status"] != "complete"
        or digest != worker["checkpoint_sha256"]
        or not worker.get("finite_model_tensors")
        or not worker.get("checkpoint_reload_action_parity")
        or worker.get("new_updates", 0) <= 0
    ):
        raise ValueError("Cannot register an unverified backflip checkpoint")
    record = {
        "task": "backflip",
        "training_status": "complete",
        "skill_success": None,
        "checkpoint": str(checkpoint.relative_to(Path(workspace)))
        if checkpoint.is_relative_to(Path(workspace))
        else str(checkpoint),
        "checkpoint_sha256": digest,
        "iteration": worker["final_iteration"],
        "updates": worker["final_updates"],
        "selection": "Last completed full-width backflip run; not selected by reward",
        "scope": "Training completion does not establish a successful backflip or strict acceptance.",
    }
    target = task_directory(workspace)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "current.json", record)


def prepare(workspace, root):
    """Execute only in the isolated SDK: identical CSV, then CPU G1 FK at50Hz."""
    import sys

    import numpy as np

    workspace, root = Path(workspace), Path(root)
    destination = task_directory(workspace) / "data"
    if destination.exists():
        return prepared(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name("data-preparing-" + uuid4().hex)
    staging.mkdir()
    # Build privately; publish data/ only after all files are verified.
    source = workspace / ".cache/external/mjbatch/examples/assets/flip.npz"
    target = staging / "source.npz"
    try:
        if source.is_file():
            copy_file_snapshot(source, target)
        else:
            with (
                urllib.request.urlopen(SOURCE_URL, timeout=30) as response,
                target.open("xb") as stream,
            ):
                shutil.copyfileobj(response, stream)
        if snapshot_file_hashes(target)[0] != SOURCE_SHA256:
            raise ValueError("Backflip source SHA-256 mismatch")
        with np.load(target, allow_pickle=False) as clip:
            if float(clip["fps"][0]) != 50 or clip["joint_pos"].shape != (358, 29):
                raise ValueError("Unexpected G1 backflip source layout")
            root_pos = clip["body_pos_w"][:, 0]
            quaternion_xyzw = clip["body_quat_w"][:, 0, [1, 2, 3, 0]]
            full = np.column_stack((root_pos, quaternion_xyzw, clip["joint_pos"]))
        selected = full[30:241]
        raw = np.vstack(
            (
                np.repeat(selected[:1], 10, axis=0),
                selected,
                np.repeat(selected[-1:], 100, axis=0),
            )
        )
        if raw.shape != (321, 36) or not np.isfinite(raw).all():
            raise ValueError("Backflip recipe produced invalid frames")
        np.savetxt(staging / "backflip.csv", raw, delimiter=",", fmt="%.10g")
        if snapshot_file_hashes(staging / "backflip.csv")[0] != CSV_SHA256:
            raise ValueError(
                "Backflip CSV differs from the pinned reference experiment"
            )
        extended = np.vstack((raw, np.repeat(raw[-1:], 250, axis=0)))
        np.savetxt(staging / "backflip_eval.csv", extended, delimiter=",", fmt="%.10g")
        sys.path.insert(0, str(root / "src"))
        from supermimic.data.motion_convert import (
            G1_JOINT_NAMES,
            convert_csv_to_npz,
            load_g1_model,
        )

        model = load_g1_model()  # Verifies all29 joint names against the actual MJCF.
        for phase, csv_name, frames in (
            ("train", "backflip.csv", 321),
            ("eval", "backflip_eval.csv", 571),
        ):
            output = staging / phase / "backflip.npz"
            summary = convert_csv_to_npz(
                staging / csv_name, output, input_fps=50, output_fps=50, model=model
            )
            if summary["output_frames"] != frames:
                raise ValueError("50Hz reference was unexpectedly resampled")
            with np.load(output, allow_pickle=False) as converted:
                if not np.allclose(converted["joint_vel"][:8], 0) or not np.allclose(
                    converted["joint_vel"][-98:], 0
                ):
                    raise ValueError("Reference start/end hold velocities are not zero")
                arrays = {name: converted[name].copy() for name in converted.files}
            arrays["source"] = np.array(str(destination / csv_name))
            np.savez(output, **arrays)
        record = {
            "status": "complete",
            "method": "SuperMimic",
            "backend": "mjlab",
            "reference_project": "https://github.com/liudi159/supermimic-backflip-5090",
            "reference_revision": REFERENCE_REVISION,
            "source_url": SOURCE_URL,
            "source_revision": SOURCE_REVISION,
            "source_sha256": SOURCE_SHA256,
            "source_license": "Apache-2.0 (mjbatch)",
            "source_usage": "Motion targets only; no mjbatch iLQR controller or external trained policy is run.",
            "csv_sha256": CSV_SHA256,
            "csv_quaternion": "xyzw",
            "npz_quaternion": "wxyz",
            "control_hz": 50,
            "source_frames_inclusive": [30, 240],
            "initial_hold_frames": 10,
            "final_hold_frames": 100,
            "training_frames": 321,
            "extra_evaluation_hold_frames": 250,
            "evaluation_frames": 571,
            "joint_names": list(G1_JOINT_NAMES),
            "train_files": inspect_motion_files(staging / "train"),
            "eval_files": inspect_motion_files(staging / "eval"),
            "external_policy_weights_used": False,
            "scope": "Same root/joint CSV as reference; FK recomputed for the current pinned mjlab G1. Different backend, networks, observations, collisions and reduced batches; not a strict V14 reproduction.",
        }
        write_json(staging / "provenance.json", record)
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        os.rename(staging, destination)
        return prepared(destination)
    except BaseException:
        # Keep failed preparation evidence separate; never publish it as data/.
        raise


def lateral_rotation_degrees(quaternions):
    """Accumulate small SO(3) increments about the initial body's lateral axis."""
    import numpy as np

    q = np.asarray(quaternions, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or len(q) == 0:
        raise ValueError("Expected a nonempty wxyz quaternion sequence")
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    if q.ndim != 2 or q.shape[1] != 4 or not np.isfinite(q).all() or (norms <= 0).any():
        raise ValueError("Expected finite nonzero wxyz quaternions")
    q = q / norms
    a, b = q[1:], q[:-1] * [1, -1, -1, -1]
    w = a[:, :1] * b[:, :1] - np.sum(a[:, 1:] * b[:, 1:], axis=1, keepdims=True)
    v = a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + np.cross(a[:, 1:], b[:, 1:])
    v = np.where(w < 0, -v, v)
    w = np.abs(w)
    length = np.linalg.norm(v, axis=1, keepdims=True)
    factor = np.divide(
        2 * np.arctan2(length, w),
        length,
        out=np.zeros_like(length),
        where=length > 1e-12,
    )
    qw, x, y, z = q[0]
    axis = np.array(
        [2 * (x * y - qw * z), 1 - 2 * (x * x + z * z), 2 * (y * z + qw * x)]
    )
    return float(np.degrees(np.sum((v * factor) @ axis)))


def fixed_window_pose_quality(joints, target_joints, ankles, target_ankles):
    """Fixed90:118 pose diagnostic; this is not the dynamic airborne V14 gate."""
    import numpy as np

    values = [
        np.asarray(value, dtype=np.float64)
        for value in (joints, target_joints, ankles, target_ankles)
    ]
    for value, shape in zip(values, ((29,), (29,), (2, 3), (2, 3)), strict=True):
        if value.ndim != len(shape) + 1 or value.shape[1:] != shape:
            raise ValueError("Expected N29 joints and N2x3 ankle positions")
        if not np.isfinite(value).all():
            raise ValueError("Pose diagnostics require finite arrays")
    actual, target, feet, target_feet = values
    if len(actual) != len(feet) or len(target) != len(target_feet):
        raise ValueError("Joint and ankle sample counts must agree")
    result = {
        "window_frames_start_inclusive_end_exclusive": [90, 118],
        "window_hz": 50,
        "hip_knee_rmse_deg": None,
        "reference_adjusted_leg_asymmetry_rmse_deg": None,
        "max_ankle_separation_m": None,
        "reference_max_ankle_separation_m": None,
        "hip_knee_rmse_limit_deg": 25.0,
        "ankle_separation_allowance_m": 0.15,
        "pose_quality_passed": None,
        "scope": "Fixed reference-time window, no motion retiming. This metric excludes dynamic airborne and200Hzcontact-force gates; use the separate physical-quality record when provided.",
    }
    if min(map(len, values)) < 118:
        return result
    actual, target, feet, target_feet = [value[90:118] for value in values]
    ids = [0, 1, 2, 3, 6, 7, 8, 9]
    error = actual[:, ids] - target[:, ids]
    rmse = float(np.degrees(np.sqrt(np.mean(error**2))))
    mirror = np.array([1, -1, -1, 1, 1, -1])
    asymmetry = (actual[:, :6] - actual[:, 6:12] * mirror) - (
        target[:, :6] - target[:, 6:12] * mirror
    )
    separation = float(np.max(np.linalg.norm(feet[:, 0] - feet[:, 1], axis=1)))
    reference_separation = float(
        np.max(np.linalg.norm(target_feet[:, 0] - target_feet[:, 1], axis=1))
    )
    result.update(
        hip_knee_rmse_deg=rmse,
        reference_adjusted_leg_asymmetry_rmse_deg=float(
            np.degrees(np.sqrt(np.mean(asymmetry**2)))
        ),
        max_ankle_separation_m=separation,
        reference_max_ankle_separation_m=reference_separation,
        pose_quality_passed=bool(
            rmse < 25.0 and separation <= reference_separation + 0.15
        ),
    )
    return result


def evaluate(policy, paths, cfg, output):
    """One real policy rollout with50Hz diagnostics, not the reference200Hz gate."""
    import numpy as np
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import ClipResult

    if len(paths) != 1:
        raise ValueError("Backflip evaluation requires exactly one reference")
    with np.load(paths[0], allow_pickle=False) as data:
        frames = len(data["joint_pos"])
    env = make_env(
        SuperMimicEnvCfgArgs(
            npz_files=[str(paths[0])],
            num_envs=1,
            future_frames=cfg["env"]["future_frames"],
            episode_length_s=1e5,
            action_noise_std=0.0,
            obs_noise=False,
            reset_noise=False,
            domain_randomization=False,
            sampling_mode="start",
            record_transitions=False,
            seed=0,
            anchor_obs=cfg["env"].get("anchor_obs", False),
        ),
        device="cuda:0",
    )
    states, reward, sums = [], 0.0, None
    terminated = timed_out = False
    try:
        command = env.command_manager.get_term("motion")
        command.forced_frame[:] = command.library.clip_start[0]
        obs, _ = env.reset()
        state_io = StateIO(env)
        names = list(env.reward_manager.active_terms)
        with torch.no_grad():
            for _ in range(frames + 2):
                states.append(state_io.read()[0].cpu().numpy().copy())
                obs, rew, term, tout, _ = env.step(policy(obs["policy"], obs["goal"]))
                reward += float(rew[0])
                values = env.reward_manager.term_rewards[0].cpu().numpy().copy()
                sums = values if sums is None else sums + values
                terminated, timed_out = bool(term[0]), bool(tout[0])
                if terminated or timed_out:
                    break  # Auto-reset state is never appended to the trajectory.
    finally:
        env.close()
    state = np.stack(states)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite backflip rollout state")
    steps = len(state)
    result = ClipResult(
        name="backflip",
        frames=frames,
        steps=steps,
        norm_return=reward / (0.1 * steps),
        failed=terminated or not timed_out,
        fail_step=steps if terminated or not timed_out else -1,
        terms={n: float(sums[i]) / steps for i, n in enumerate(names)},
    )
    np.savez(
        Path(output) / "backflip-rollout.npz", state=state, time_s=np.arange(steps) / 50
    )
    diagnostics = diagnose_rollout(
        state, paths[0], output, not result.failed and steps >= frames
    )
    return [result], diagnostics


def diagnose_rollout(state, reference_path, output, complete):
    """Shared saved-state diagnostics for both SuperMimic implementations."""
    import mujoco
    import numpy as np
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

    state = np.asarray(state)
    if (
        state.ndim != 2
        or state.shape[1] != 71
        or not len(state)
        or not np.isfinite(state).all()
    ):
        raise ValueError("Expected finite nonempty N71 actual simulator states")
    steps = len(state)
    with np.load(reference_path, allow_pickle=False) as data:
        reference_quat = data["body_quat_w"][:, 0].copy()
        reference_joints = data["joint_pos"].copy()
        body_names = [str(name) for name in data["body_names"]]
        foot_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        reference_ankles = data["body_pos_w"][
            :, [body_names.index(name) for name in foot_names]
        ].copy()
    if complete and steps < len(reference_joints):
        raise ValueError("Completed rollout must cover all reference timestamps")
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.worldbody.add_geom(
        name="ef_backflip_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1]
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ef_backflip_floor")
    feet = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in ("left_ankle_roll_link", "right_ankle_roll_link")
    }
    if -1 in feet:
        raise ValueError("Cannot identify G1 feet for contact diagnostics")
    ankle_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_names
    ]
    ankle_positions = []
    penetration_frames = 0
    max_penetration = 0.0
    for row in state:
        data.qpos[:] = row[:36]
        data.qvel[:] = row[36:]
        mujoco.mj_forward(
            model, data
        )  # Geometry at saved poses; no new trajectory step.
        ankle_positions.append(data.xpos[ankle_ids].copy())
        bad = False
        for contact in data.contact:
            one, two = map(int, contact.geom)
            if floor not in (one, two) or contact.dist >= 0:
                continue
            other = two if one == floor else one
            if int(model.geom_bodyid[other]) not in feet:
                bad = True
                max_penetration = max(max_penetration, float(-contact.dist))
        penetration_frames += int(bad)
    rotation = lateral_rotation_degrees(state[:, 3:7])
    target_rotation = lateral_rotation_degrees(reference_quat)
    hold = None
    if complete and steps >= 100:
        end = state[-100:]
        linear = np.linalg.norm(end[:, 36:39], axis=1)
        angular = np.linalg.norm(end[:, 39:42], axis=1)
        q = end[:, 3:7] / np.linalg.norm(end[:, 3:7], axis=1, keepdims=True)
        hold = {
            "horizontal_drift_m": float(
                np.max(np.linalg.norm(end[:, :2] - end[:1, :2], axis=1))
            ),
            "linear_rms_m_s": float(np.sqrt(np.mean(linear**2))),
            "linear_max_m_s": float(np.max(linear)),
            "angular_rms_rad_s": float(np.sqrt(np.mean(angular**2))),
            "min_upright_projection": float(
                np.min(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
            ),
        }
    stable = (
        hold is not None
        and hold["horizontal_drift_m"] < 0.1
        and hold["linear_rms_m_s"] < 0.1
        and hold["linear_max_m_s"] < 0.2
        and hold["angular_rms_rad_s"] < 0.2
        and hold["min_upright_projection"] > 0.99
    )
    basic = bool(
        complete
        and abs(rotation - target_rotation) < 20
        and penetration_frames == 0
        and stable
    )
    pose = fixed_window_pose_quality(
        state[:, 7:36], reference_joints, ankle_positions, reference_ankles
    )
    quality = basic and pose["pose_quality_passed"] is True
    diagnostics = {
        "recording_hz": 50,
        "evaluation_seed": 0,
        "completed_extended_reference": complete,
        "actual_lateral_rotation_deg": rotation,
        "reference_lateral_rotation_deg": target_rotation,
        "nonfoot_floor_penetration_frames": penetration_frames,
        "max_nonfoot_floor_penetration_m": max_penetration,
        "last2s_hold": hold,
        "basic_checks_passed": basic,
        "fixed_window_pose": pose,
        "quality_checks_passed": quality,
        "skill_success": None if quality else False,
        "strict_acceptance": "not_assessed",
        "scope": "One actual policy rollout, no pose substitution or post-failure reset samples. Saved-pose geometry, hold and fixed-window leg diagnostics only. Dynamic airborne and200Hzsolved-force acceptance, when available, is a separate physical-quality record. These diagnostics alone do not establish robust success or exactV14equivalence.",
    }
    write_json(Path(output) / "backflip-diagnostics.json", diagnostics)
    return diagnostics
