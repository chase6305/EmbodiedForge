"""Optional GMR worker: BVH -> IK -> existing robot replay NPZ."""

import importlib.metadata
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from ._h1_motion import MotionRecorder, render_motion
from ._motion_sdk import load_request
from .h1 import sha256, write_json


def bvh_timing(path):
    """Preserve the file's actual frame interval instead of a rounded FPS."""
    frames = interval = None
    with Path(path).open() as stream:
        for line in stream:
            if match := re.fullmatch(r"\s*Frames:\s*(\d+)\s*", line):
                frames = int(match[1])
            if match := re.fullmatch(r"\s*Frame Time:\s*(\S+)\s*", line):
                interval = float(match[1])
                break
    if not frames or interval is None or not np.isfinite(interval) or interval <= 0:
        raise ValueError("BVH requires positive Frames and finite positive Frame Time")
    return frames, interval


def human_frames(path, source_format):
    if source_format != "xsens":
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames, _ = load_bvh_file(str(path), format=source_format)
        return frames

    # The high-level Xsens loader imports a Qt editor and implicitly reads
    # offsets.json from cwd. Use its pinned BVH parser with explicit zero offsets.
    from general_motion_retargeting.utils.xsens_vendor.BVHParser import (
        BVHParser,
        quat_fk,
    )

    parser = BVHParser(axis_order="zxy", scale=0.01)
    rotations, positions = parser.parse(Path(path).read_text())
    quats, positions, _, parents = parser._MOTION_data_post_processing(
        rotations, positions, reset_to_zero=True
    )
    orientations, positions = quat_fk(quats, positions, parents)
    frames = []
    for q, p in zip(orientations, positions, strict=True):
        frame = {name: (p[index], q[index]) for index, name in enumerate(parser.names)}
        frame["LeftFootMod"] = frame["LeftAnkle"]
        frame["RightFootMod"] = frame["RightAnkle"]
        frames.append(frame)
    return frames


def retarget(request):
    repo = Path(request["repository"])
    sys.path.insert(0, str(repo))
    import general_motion_retargeting as gmr
    import mujoco
    from general_motion_retargeting.params import IK_CONFIG_DICT

    if Path(gmr.__file__).resolve().parent != repo / "general_motion_retargeting":
        raise ValueError("GMR imported from a different checkout")
    output, options = Path(request["output"]), request["options"]
    source = request["inputs"]["source.bvh"]
    count, dt = bvh_timing(source["path"])
    frames = human_frames(source["path"], options["format"])
    if len(frames) != count:
        raise ValueError(f"BVH declares {count} frames, decoder returned {len(frames)}")
    solver = gmr.GeneralMotionRetargeting(
        src_human=f"bvh_{options['format']}",
        tgt_robot=options["robot"],
        actual_human_height=options["human_height"],
        solver="daqp",
        verbose=False,
        use_velocity_limit=False,
    )
    model = solver.model
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    if (
        len(free) != 1
        or len(joints) + 1 != model.njnt
        or model.jnt_qposadr[free[0]] != 0
    ):
        raise ValueError("Expected one leading floating root and scalar hinge joints")
    root = int(model.jnt_bodyid[free[0]])
    bodies = [root] + [i for i in range(1, model.nbody) if i != root]
    lookup = {body: index for index, body in enumerate(bodies)}
    metadata = {
        "schema": 1,
        "source_kind": "retargeted_reference",
        "title": f"GMR {options['robot']} reference",
        "robot": options["robot"],
        "quaternion_order": "xyzw",
        "coordinate_frame": "world",
        "fps": 1 / dt,
        "source_frame_zero_time": dt,
        "joint_names": [model.joint(int(i)).name for i in joints],
        "body_names": [model.body(i).name for i in bodies],
        "edges": [
            [lookup[int(model.body_parentid[i])], lookup[i]]
            for i in bodies
            if int(model.body_parentid[i]) in lookup
        ],
        "source": source,
        "gmr_revision": request["revision"],
    }
    recorder = MotionRecorder(metadata)
    data = mujoco.MjData(model)
    started = time.monotonic()
    max_violation = 0.0
    limited = joints[model.jnt_limited[joints].astype(bool)]
    for index, frame in enumerate(frames):
        qpos = solver.retarget(frame)
        if not np.isfinite(qpos).all() or not np.isclose(
            np.linalg.norm(qpos[3:7]), 1, atol=1e-6
        ):
            raise ValueError(f"Invalid IK configuration at frame {index}")
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        if len(limited):
            values = qpos[model.jnt_qposadr[limited]]
            bounds = model.jnt_range[limited]
            max_violation = max(
                max_violation,
                float(np.maximum(bounds[:, 0] - values, values - bounds[:, 1]).max()),
            )
        recorder.append(
            (index + 1) * dt,
            data.xpos[bodies],
            np.roll(data.xquat[bodies], -1, axis=-1),
            qpos[model.jnt_qposadr[joints]],
            False,
            False,
        )
        if (index + 1) % 500 == 0 or index + 1 == count:
            print(f"Retargeted {index + 1}/{count} frames", flush=True)
    destination = output / "motion.npz"
    recorder.save(destination)
    render_motion(destination, output / "motion.html")
    ik_path = IK_CONFIG_DICT[f"bvh_{options['format']}"][options["robot"]]
    result = {
        "status": "complete",
        "frames": count,
        "fps": 1 / dt,
        "source_duration_seconds": (count - 1) * dt,
        "retarget_seconds": time.monotonic() - started,
        "robot": options["robot"],
        "joints": len(joints),
        "bodies": len(bodies),
        "motion": str(destination),
        "motion_sha256": sha256(destination),
        "model": str(solver.xml_file),
        "model_sha256": sha256(Path(solver.xml_file)),
        "ik_config_sha256": sha256(ik_path),
        "max_joint_limit_violation_rad": max_violation,
        "solver": "daqp",
        "velocity_limits": False,
        "xsens_offsets": "zero; initial root XY/yaw normalized"
        if options["format"] == "xsens"
        else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "mujoco",
                "mink",
                "qpsolvers",
                "daqp",
            )
        },
    }
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    retarget(load_request(sys.argv[1]))
