"""Inspect reference motion clips and score aligned recorded rollouts on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from ._motion_clips import MotionClips, import_weave


def compare_clips(reference, rollout):
    """Score each first episode, using failed termination before clip completion.

    Recorded frames must align to reference frames from time zero, in the same
    world frame, before simulator reset. No resampling or retargeting is inferred.
    """
    for key in (
        "robot",
        "joint_names",
        "body_names",
        "coordinate_frame",
        "quaternion_order",
    ):
        if reference.metadata[key] != rollout.metadata[key]:
            raise ValueError(f"Motion layouts differ: {key}")
    if reference.fps != rollout.fps:
        raise ValueError(
            "Motion frame rates differ; resample explicitly before scoring"
        )
    if rollout.metadata.get("reference_sha256") != reference.sha256:
        raise ValueError("Rollout reference_sha256 does not match the reference file")
    if set(reference.names) != set(rollout.names):
        raise ValueError("Reference and rollout must contain the same named clips")
    if "terminated" not in rollout.arrays:
        raise ValueError("Rollout has no termination flags")
    lookup = {name: i for i, name in enumerate(rollout.names)}
    clips = []
    for index, name in enumerate(reference.names):
        other = lookup[name]
        count, target = int(rollout.lengths[other]), int(reference.lengths[index])
        if reference.objects[index] != rollout.objects[other] or count > target:
            raise ValueError(
                f"Object mismatch or rollout longer than reference: {name}"
            )
        a = slice(reference.starts[index], reference.starts[index] + count)
        b = slice(rollout.starts[other], rollout.starts[other + 1])
        last = rollout.starts[other + 1] - 1
        failed, timed_out, reached_end = (
            bool(rollout.arrays[key][last])
            for key in ("terminated", "truncated", "clip_end")
        )
        if reached_end and count != target:
            raise ValueError(f"clip_end before the reference ends: {name}")
        status = (
            "failed"
            if failed
            else "success"
            if reached_end
            else "truncated"
            if timed_out
            else "incomplete"
        )
        metrics = {}
        for field, label in (
            ("joint_pos", "joint_rmse_rad"),
            ("body_pos_w", "body_position_rmse_m"),
            ("object_pos_w", "object_position_rmse_m"),
        ):
            error = (
                rollout.arrays[field][b].astype(np.float64) - reference.arrays[field][a]
            )
            squared = np.square(error)
            if field != "joint_pos":
                squared = squared.sum(axis=-1)
            metrics[label] = float(np.sqrt(squared.mean()))
        for field, label in (
            ("body_quat_w", "body_rotation_rmse_rad"),
            ("object_quat_w", "object_rotation_rmse_rad"),
        ):
            qa, qb = (
                reference.arrays[field][a].astype(np.float64),
                rollout.arrays[field][b].astype(np.float64),
            )
            qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
            qb /= np.linalg.norm(qb, axis=-1, keepdims=True)
            # Align equivalent signs, then use chord lengths: their ratio is
            # tan(angle / 4). Unlike acos(dot), this preserves small rotations.
            qb *= np.where((qa * qb).sum(axis=-1, keepdims=True) < 0, -1.0, 1.0)
            angle = 4 * np.arctan2(
                np.linalg.norm(qa - qb, axis=-1),
                np.linalg.norm(qa + qb, axis=-1),
            )
            metrics[label] = float(np.sqrt(np.square(angle).mean()))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError(f"Nonfinite tracking metrics: {name}")
        clips.append(
            {
                "name": name,
                "object": reference.objects[index],
                "status": status,
                "frames": count,
                "reference_frames": target,
                "progress": count / target,
                **metrics,
            }
        )
    keys = tuple(metrics)

    def average(rows):
        return {
            key: float(np.mean([row[key] for row in rows])) if rows else None
            for key in keys
        }

    return {
        "schema": "ef-motion-evaluation-v1",
        "status": "incomplete"
        if any(row["status"] == "incomplete" for row in clips)
        else "complete",
        "reference_sha256": reference.sha256,
        "rollout_sha256": rollout.sha256,
        "num_clips": len(clips),
        "fps": reference.fps,
        "success_rate": sum(row["status"] == "success" for row in clips) / len(clips),
        "aggregation": "arithmetic mean of per-clip RMSE; all clips weighted equally",
        "metrics_all": average(clips),
        "metrics_success": average(
            [row for row in clips if row["status"] == "success"]
        ),
        "clips": clips,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser(
        "import-weave", help="Import numeric/string NPZ with explicit robot layout"
    )
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--layout", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--input", type=Path, required=True)
    score = commands.add_parser(
        "compare", help="Score recorded trajectories, without simulation"
    )
    score.add_argument("--reference", type=Path, required=True)
    score.add_argument("--rollout", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "import-weave":
            library = import_weave(args.input, json.loads(args.layout.read_text()))
            library.save(args.output)
            result = {"output": str(args.output), "num_clips": len(library.names)}
        elif args.command == "inspect":
            library = MotionClips.load(args.input)
            result = {
                "metadata": library.metadata,
                "sha256": library.sha256,
                "fps": library.fps,
                "num_clips": len(library.names),
                "frames": int(library.starts[-1]),
                "clips": [
                    {"name": name, "object": obj, "frames": int(length)}
                    for name, obj, length in zip(
                        library.names, library.objects, library.lengths, strict=True
                    )
                ],
            }
        else:
            result = compare_clips(
                MotionClips.load(args.reference), MotionClips.load(args.rollout)
            )
            encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
            with args.output.open("x") as stream:
                stream.write(encoded)
        print(json.dumps(result, indent=2, allow_nan=False))
        if result.get("status") == "incomplete":
            raise SystemExit(1)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Motion operation failed: {exc}\n")


if __name__ == "__main__":
    main()
