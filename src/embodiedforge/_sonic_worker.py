"""Execute each bundled clip using upstream SONIC's policy and control loop."""

import importlib.metadata
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ._microduck_process import run_process
from ._motion_sdk import load_request
from .h1 import sha256, write_json


def assets(repo):
    xml = repo / "assets/mjcf/x2_ultra.xml"
    tree = ET.parse(xml)
    meshdir = xml.parent / tree.find("compiler").get("meshdir", "")
    paths = {xml} | {
        meshdir / node.attrib["file"]
        for node in tree.iter("mesh")
        if "file" in node.attrib
    }
    result = []
    for path in sorted(paths):
        if not path.is_file():
            raise ValueError(
                f"Missing X2 asset: {path}; install the official X2 v1.3.0 meshes"
            )
        with path.open("rb") as stream:
            if stream.read(80).startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ):
                raise ValueError(f"Unresolved LFS asset: {path}")
        result.append({"path": str(path.resolve()), "sha256": sha256(path)})
    return result


def episode_result(text):
    """The pinned player emits rounded text metrics; keep that precision explicit."""
    rows = [
        line.strip() for line in text.splitlines() if line.lstrip().startswith("[end]")
    ]
    if len(rows) != 1:
        raise ValueError("Expected exactly one terminal episode report from SONIC")
    match = re.fullmatch(
        r"\[end\] ep=0 ran ([\d.]+)s, reason: ([^|\r\n]+) \| "
        r"joint MAE ([\d.]+) rad \(max ([\d.]+)\) \| pelvis-z MAE ([\d.]+) m",
        rows[0],
    )
    if match is None:
        raise ValueError("Invalid first terminal episode report from SONIC")
    seconds, reason, joint, maximum, pelvis = match.groups()
    if not all(
        math.isfinite(float(value)) for value in (seconds, joint, maximum, pelvis)
    ):
        raise ValueError("Nonfinite terminal episode metrics from SONIC")
    if reason == "motion_end":
        status = "success"
    elif reason.startswith(("pelvis_z=", "gravity_body[z]=")):
        status = "failed"
    elif reason.startswith("reached --max-episode="):
        status = "truncated"
    else:
        raise ValueError(f"Unknown SONIC termination reason: {reason}")
    return {
        "status": status,
        "reason": reason,
        "simulated_seconds": float(seconds),
        "joint_mae_rad": float(joint),
        "max_joint_error_rad": float(maximum),
        "pelvis_z_mae_m": float(pelvis),
    }


def validate_clip(name, clip, init_frame):
    if not isinstance(name, str) or not name or not isinstance(clip, dict):
        raise ValueError("Motion bank must contain named clip dictionaries")
    fps = float(clip["fps"])
    joints = np.asarray(clip["dof"])
    if not np.isfinite(fps) or fps <= 0 or joints.ndim != 2 or joints.shape[1] != 31:
        raise ValueError(f"Invalid X2 rate or joint layout: {name}")
    count = len(joints)
    if not 0 <= init_frame < count - 1:
        raise ValueError(f"--init-frame must leave at least two frames in {name}")
    for key, shape in (
        ("dof", (count, 31)),
        ("root_trans_offset", (count, 3)),
        ("root_rot", (count, 4)),
    ):
        value = np.asarray(clip[key])
        if (
            value.shape != shape
            or value.dtype.kind not in "iuf"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid {key} in {name}")
    if not np.allclose(np.linalg.norm(clip["root_rot"], axis=-1), 1, atol=1e-3, rtol=0):
        raise ValueError(f"Nonunit root quaternion in {name}")
    return count, fps


def evaluate(request):
    import joblib
    import onnxruntime as ort

    repo, output = Path(request["repository"]), Path(request["output"])
    options, inputs = request["options"], request["inputs"]
    asset_manifest = assets(repo)
    # Only the pinned checkout's bundled motions are accepted by the public CLI.
    bank = joblib.load(inputs["motions.pkl"]["path"])
    if not isinstance(bank, dict) or not bank:
        raise ValueError("Empty or invalid SONIC motion bank")
    names = [options["clip"]] if options["clip"] is not None else list(bank)
    if any(name not in bank for name in names):
        raise ValueError(f"Unknown exact clip; available: {list(bank)}")
    layouts = {
        name: validate_clip(name, bank[name], options["init_frame"]) for name in names
    }
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = options["threads"]
    session = ort.InferenceSession(
        inputs["policy.onnx"]["path"],
        session_options,
        providers=["CPUExecutionProvider"],
    )
    ins, outs = session.get_inputs(), session.get_outputs()
    if (
        len(ins) != 1
        or len(outs) != 1
        or ins[0].shape[-1] != 1670
        or outs[0].shape[-1] != 31
    ):
        raise ValueError(
            "Expected a fused X2 policy with 1670 observations and 31 actions"
        )
    policy = {
        "bytes": Path(inputs["policy.onnx"]["path"]).stat().st_size,
        "sha256": inputs["policy.onnx"]["sha256"],
        "input_shape": ins[0].shape,
        "output_shape": outs[0].shape,
        "providers": session.get_providers(),
    }
    del session
    report = {
        "status": "running",
        "model": options["model"],
        "policy": policy,
        "headless": True,
        "physics": "mujoco",
        "control_hz": 50,
        "metric_source": "upstream terminal text: time 2dp, joint MAE 4dp, max joint/pelvis MAE 3dp",
        "assets": asset_manifest,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "mujoco",
                "onnxruntime",
                "joblib",
            )
        },
        "clips": [],
    }
    write_json(output / "result.json", report)
    env = os.environ.copy()
    env["ORT_NUM_THREADS"] = str(options["threads"])
    for index, name in enumerate(names):
        frames, fps = layouts[name]
        motion = output / "inputs" / f"clip_{index:03d}.pkl"
        joblib.dump({name: bank[name]}, motion)
        # A positive episode cap makes upstream exit after the first terminal
        # event. Leave room for its wrap detection at the next 50 Hz tick.
        cap = (frames - options["init_frame"]) / fps + 0.1
        command = [
            sys.executable,
            str(repo / "scripts/eval_x2_mujoco_onnx.py"),
            "--onnx",
            inputs["policy.onnx"]["path"],
            "--motion",
            str(motion),
            "--no-viewer",
            "--max-episode",
            str(cap),
            "--init-frame",
            str(options["init_frame"]),
        ]
        if options["model"] == "transfer-v2":
            command += ["--tuning", "", "--action-clip", "20", "--freeze-wrist"]
        else:
            command += ["--tuning", inputs["tuning.yaml"]["path"]]
        log = output / f"clip_{index:03d}.log"
        started = time.monotonic()
        run_process(command, cwd=output, env=env, log_path=log)
        result = episode_result(log.read_text())
        result.update(
            name=name,
            frames=frames,
            fps=fps,
            init_frame=options["init_frame"],
            wall_seconds=time.monotonic() - started,
            command=command,
            log=str(log),
            motion_sha256=sha256(motion),
        )
        report["clips"].append(result)
        write_json(output / "result.json", report)
    if assets(repo) != asset_manifest:
        raise ValueError("X2 assets changed during evaluation")
    report["status"] = "complete"
    report["success_rate"] = sum(
        row["status"] == "success" for row in report["clips"]
    ) / len(names)
    write_json(output / "result.json", report)
    print(
        json.dumps(
            {key: report[key] for key in ("status", "model", "success_rate", "clips")},
            indent=2,
        )
    )


if __name__ == "__main__":
    evaluate(load_request(sys.argv[1]))
