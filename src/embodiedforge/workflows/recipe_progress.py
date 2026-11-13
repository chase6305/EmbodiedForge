"""Read bounded live recipe metrics independently of checkpoint progress."""

import json
import math
import os
from pathlib import Path


def go1_training_progress(run, manifest):
    """Read the latest complete metrics row without loading the SDK or checkpoint.

    The live log may be ahead of the periodically saved checkpoint. Only a
    bounded tail is read, and an unfinished final line is left for the next poll.
    """
    request = manifest.get("request") or {}
    if request.get("task") != "go1-joystick" or request.get("command") != "train":
        return None
    try:
        with (Path(run) / "metrics.jsonl").open("rb") as stream:
            size = stream.seek(0, os.SEEK_END)
            offset = max(0, size - 65536)
            stream.seek(offset)
            tail = stream.read(65536)
    except FileNotFoundError:
        return None
    if offset:
        tail = tail.partition(b"\n")[2]
    lines = [line for line in tail.split(b"\n")[:-1] if line.strip()]
    if not lines:
        return None

    def reject_constant(value):
        raise ValueError(f"Non-finite Go1 progress value: {value}")

    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            reject_constant(value)
        return parsed

    try:
        latest = json.loads(
            lines[-1], parse_constant=reject_constant, parse_float=finite_float
        )
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid complete Go1 metrics row") from exc
    if not isinstance(latest, dict):
        raise ValueError("Invalid Go1 progress record")
    start = request.get("start_iteration", 0)
    counts = [request.get(key) for key in ("updates", "num_envs", "horizon")]
    iteration, transitions = latest.get("iteration"), latest.get("transitions")
    elapsed = latest.get("elapsed_seconds")
    if (
        type(start) is not int
        or start < 0
        or any(type(value) is not int or value <= 0 for value in counts)
        or type(iteration) is not int
        or type(transitions) is not int
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("Invalid Go1 progress counts or elapsed time")
    updates, num_envs, horizon = counts
    observed = iteration - start + 1
    if not 1 <= observed <= updates or transitions != observed * num_envs * horizon:
        raise ValueError("Go1 progress differs from the requested training budget")
    return {
        "observed_updates": observed,
        "requested_updates": updates,
        "cumulative_updates": iteration + 1,
        "fraction": observed / updates,
        "latest_metrics": latest,
    }
