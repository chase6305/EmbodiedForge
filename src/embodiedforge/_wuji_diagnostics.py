"""Streaming per-trial diagnostics; no simulation or training SDK dependencies."""

import hashlib
import json
import subprocess

import numpy as np


def verify_video(result, output):
    """Decode the requested video and check it covers every recorded step."""
    value = result.get("record_video")
    if not value:
        raise ValueError("Wuji rendering produced no requested video")
    path = (output / value).resolve()
    if not path.is_relative_to(output.resolve()) or not path.is_file():
        raise ValueError("Wuji video is missing or outside its run")
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_read_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    streams = json.loads(probe.stdout)["streams"]
    expected = sum(row["steps"] + 1 for row in result["diagnostics"]["trials"])
    if len(streams) != 1 or probe.stderr.strip():
        raise ValueError("Wuji video failed decoding")
    stream = streams[0]
    if (
        stream["width"],
        stream["height"],
        stream["r_frame_rate"],
        int(stream["nb_read_frames"]),
    ) != (640, 368, "20/1", expected):
        raise ValueError("Wuji video dimensions, frame rate or frame count mismatch")
    return {
        "path": str(path.relative_to(output.resolve())),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **stream,
    }


def validate_evaluation(request, result):
    """Check trial accounting and controller identity before accepting a report."""
    protocol = result["protocol"]
    if (
        protocol["policy"] != request.get("policy", "trained")
        or protocol["seed"] != request["seed"]
        or protocol["environment_seed"] != request["seed"]
        or not np.isclose(protocol["control_dt_s"], 0.05, rtol=0, atol=1e-12)
        or not np.isclose(
            protocol["trial_timeout_s"], request["steps"] * 0.05, rtol=0, atol=1e-12
        )
    ):
        raise ValueError("Wuji evaluation protocol differs from request")
    trials, diagnostics = result["trials"], result["diagnostics"]["trials"]
    if (
        len(trials) != request["num_trials"]
        or len(diagnostics) != len(trials)
        or result["num_trials"] != len(trials)
    ):
        raise ValueError("Incomplete Wuji trial accounting")
    counts = {status: 0 for status in ("success", "drop", "timeout")}
    for index, (trial, diagnostic) in enumerate(zip(trials, diagnostics, strict=True)):
        status = trial["status"]
        if (
            trial["trial_idx"] != index
            or diagnostic["trial_idx"] != index
            or status not in counts
        ):
            raise ValueError("Invalid Wuji trial index or status")
        counts[status] += 1
        steps = diagnostic["steps"]
        if (
            type(steps) is not int
            or not 1 <= steps <= request["steps"]
            or (status == "timeout" and steps != request["steps"])
        ):
            raise ValueError("Incomplete Wuji trial steps")
        final, minimum = (
            trial["final_orientation_error_rad"],
            trial["min_orientation_error_rad"],
        )
        if not 0 <= minimum <= final <= np.pi + 1e-5 or not np.isclose(
            final, diagnostic["final_error_rad"], atol=1e-4, rtol=0
        ):
            raise ValueError("Invalid Wuji orientation error accounting")
        success_time = trial["time_to_first_success_s"]
        if status == "success":
            if (
                trial["goal_reaches"] != 1
                or success_time is None
                or not np.isclose(success_time, steps * 0.05, rtol=0, atol=1e-10)
            ):
                raise ValueError("Invalid Wuji success accounting")
        elif trial["goal_reaches"] != 0 or success_time is not None:
            raise ValueError("Non-successful Wuji trial reports success")
        if protocol["policy"] == "zero" and diagnostic["action_rms"] != 0:
            raise ValueError("Zero-action baseline emitted nonzero controls")
    for status, count in counts.items():
        if not np.isclose(
            result[status + "_rate"], count / len(trials), rtol=0, atol=1e-12
        ):
            raise ValueError("Wuji summary rates differ from trial outcomes")


def quaternion_angle(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        raise ValueError("Expected finite quaternions of shape (4,)")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        raise ValueError("Zero quaternion is invalid")
    return float(2 * np.arccos(np.clip(abs(np.dot(a, b)) / norm, 0, 1)))


class TrialDiagnostics:
    """Track emitted actions and actual cube motion, independently of success."""

    def __init__(self):
        self.trials = []
        self.current = None

    def start(self, position, quaternion, goal):
        self.finish()
        position, quaternion, goal = (
            np.asarray(x, dtype=float).copy() for x in (position, quaternion, goal)
        )
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Invalid initial cube position")
        error = quaternion_angle(quaternion, goal)
        self.current = {
            "trial_idx": len(self.trials),
            "initial_position": position.tolist(),
            "initial_quaternion": quaternion.tolist(),
            "goal": goal.tolist(),
            "initial_error_rad": error,
            "final_error_rad": error,
            "min_error_rad": error,
            "steps": 0,
            "action_square_sum": 0.0,
            "action_delta_square_sum": 0.0,
            "action_max_abs": 0.0,
            "max_cube_rotation_rad": 0.0,
            "max_cube_displacement_m": 0.0,
        }
        self.previous_action = None

    def update(self, position, quaternion, action):
        if self.current is None:
            raise ValueError("No active diagnostic trial")
        position, action = np.asarray(position), np.asarray(action)
        if (
            position.shape != (3,)
            or action.shape != (20,)
            or not np.isfinite(position).all()
            or not np.isfinite(action).all()
        ):
            raise ValueError("Expected finite cube position and 20-dimensional action")
        row = self.current
        error = quaternion_angle(quaternion, row["goal"])
        row["final_error_rad"] = error
        row["min_error_rad"] = min(error, row["min_error_rad"])
        row["max_cube_rotation_rad"] = max(
            row["max_cube_rotation_rad"],
            quaternion_angle(quaternion, row["initial_quaternion"]),
        )
        row["max_cube_displacement_m"] = max(
            row["max_cube_displacement_m"],
            float(np.linalg.norm(position - row["initial_position"])),
        )
        row["action_square_sum"] += float(np.square(action.astype(float)).sum())
        row["action_max_abs"] = max(row["action_max_abs"], float(np.abs(action).max()))
        if self.previous_action is not None:
            row["action_delta_square_sum"] += float(
                np.square(action - self.previous_action).sum()
            )
        self.previous_action = action.astype(float).copy()
        row["steps"] += 1

    def finish(self):
        if self.current is None:
            return
        row = self.current
        if not row["steps"]:
            raise ValueError("Diagnostic trial has no physics steps")
        row["action_rms"] = float(
            np.sqrt(row.pop("action_square_sum") / (20 * row["steps"]))
        )
        delta = row.pop("action_delta_square_sum")
        row["action_delta_rms"] = (
            float(np.sqrt(delta / (20 * (row["steps"] - 1))))
            if row["steps"] > 1
            else None
        )
        row["final_error_reduction_rad"] = (
            row["initial_error_rad"] - row["final_error_rad"]
        )
        row["best_error_reduction_rad"] = (
            row["initial_error_rad"] - row["min_error_rad"]
        )
        self.trials.append(row)
        self.current = None

    def report(self):
        self.finish()
        if not self.trials:
            raise ValueError("No diagnostic trials")
        keys = (
            "action_rms",
            "max_cube_rotation_rad",
            "max_cube_displacement_m",
            "final_error_reduction_rad",
            "best_error_reduction_rad",
        )
        return {
            "schema": 1,
            "frame": "palm tag coordinates; quaternions wxyz",
            "summary": {
                "mean_" + key: float(np.mean([r[key] for r in self.trials]))
                for key in keys
            },
            "trials": self.trials,
        }
