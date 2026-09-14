"""Deterministic command switching and first-episode response metrics."""

import numpy as np

from ._h1_metrics import BASIC_COMMANDS, FirstEpisodeMetrics

SWITCHING_SUITES = (
    "switching",
    "lateral-switching",
    "maneuver-switching",
    "maneuver-switching-mirrored",
)

TRACKING_PROTOCOL = {
    "version": 1,
    "planar_tolerance_mps": 0.1,
    "yaw_tolerance_radps": 0.15,
    "denominator": "all initial environments times scheduled segment steps",
    "terminal_frame_counts_as_tracking": False,
    "overall": "minimum segment tracking fraction",
}


def evaluation_commands(suite, velocity=None):
    """Return independent command lists for a versioned, deterministic suite."""
    if suite in SWITCHING_SUITES:
        return {suite: None}
    if suite is None:
        return {"custom": list(velocity)}
    if suite not in ("basic", "extended"):
        raise ValueError("Unknown Go1 evaluation suite")
    commands = {name: command.copy() for name, command in BASIC_COMMANDS.items()}
    if suite == "extended":
        commands.update(
            fast_forward=[1.0, 0.0, 0.0],
            backward=[-0.5, 0.0, 0.0],
            strafe_left=[0.0, 0.3, 0.0],
            strafe_right=[0.0, -0.3, 0.0],
            spin_left=[0.0, 0.0, 0.8],
            spin_right=[0.0, 0.0, -0.8],
        )
    return commands


def switching_schedule(steps, suite="switching"):
    if suite not in SWITCHING_SUITES:
        raise ValueError("Unknown Go1 switching suite")
    if type(steps) is not int or steps < 300 or steps % 6:
        raise ValueError("Switching suite requires steps divisible by 6, at least 300")
    commands = (
        ("stand", [0.0, 0.0, 0.0]),
        ("forward", [0.5, 0.0, 0.0]),
        ("stop", [0.0, 0.0, 0.0]),
        ("turn_left", [0.5, 0.0, 0.5]),
        ("turn_right", [0.5, 0.0, -0.5]),
        ("stop_after_turn", [0.0, 0.0, 0.0]),
    )
    if suite == "lateral-switching":
        commands = (
            ("stand", [0.0, 0.0, 0.0]),
            ("strafe_left", [0.0, 0.3, 0.0]),
            ("strafe_right", [0.0, -0.3, 0.0]),
            ("backward", [-0.5, 0.0, 0.0]),
            ("diagonal_forward", [0.5, 0.3, 0.0]),
            ("stop_after_diagonal", [0.0, 0.0, 0.0]),
        )
    elif suite in ("maneuver-switching", "maneuver-switching-mirrored"):
        commands = (
            ("stand", [0.0, 0.0, 0.0]),
            ("diagonal_backward", [-0.5, -0.3, 0.0]),
            ("lateral_turn_left", [0.0, 0.3, 0.5]),
            ("lateral_turn_right", [0.0, -0.3, -0.5]),
            ("spin_left", [0.0, 0.0, 0.8]),
            ("stop_after_spin", [0.0, 0.0, 0.0]),
        )
        if suite == "maneuver-switching-mirrored":
            names = {
                "diagonal_backward": "diagonal_backward_left",
                "lateral_turn_left": "lateral_turn_right",
                "lateral_turn_right": "lateral_turn_left",
                "spin_left": "spin_right",
            }
            commands = tuple(
                (names.get(name, name), [vx, -vy, -yaw])
                for name, (vx, vy, yaw) in commands
            )
    width = steps // len(commands)
    return [
        {
            "segment": name,
            "start_step": i * width,
            "end_step": (i + 1) * width,
            "command": command,
        }
        for i, (name, command) in enumerate(commands)
    ]


class SwitchingMetrics:
    """Keep fallen rows inactive across segment boundaries, including terminal frames."""

    def __init__(self, num_envs, steps, suite="switching"):
        self.schedule = switching_schedule(steps, suite)
        self.total = FirstEpisodeMetrics(num_envs, steps, 0.02)
        self.width = steps // 6
        self.count = np.zeros((6, num_envs), dtype=int)
        self.tracking_count = np.zeros((6, num_envs), dtype=int)
        self.error = np.zeros((6, num_envs, 3))
        self.entered = np.zeros((6, num_envs), dtype=bool)
        self.survived = np.zeros((6, num_envs), dtype=bool)
        self.streak = np.zeros(num_envs, dtype=int)
        self.settled_at = np.full((6, num_envs), np.nan)

    @property
    def active(self):
        return self.total.active

    @property
    def steps(self):
        return self.total.steps

    def command(self):
        if self.steps >= self.total.limit:
            raise ValueError("Switching schedule is exhausted")
        return np.broadcast_to(
            self.schedule[self.steps // self.width]["command"], (len(self.active), 3)
        )

    def update(self, velocity, command, terminated, truncated):
        expected = self.command()
        if not np.array_equal(command, expected):
            raise ValueError("Observed command differs from switching schedule")
        segment, offset = divmod(self.steps, self.width)
        active = self.active.copy()
        # Validate before updating segment data; total includes the terminal frame.
        self.total.update(velocity, command, terminated, truncated)
        if offset == 0:
            self.entered[segment] = active
            self.streak[:] = 0
        error = (np.asarray(velocity)[active] - expected[active]) ** 2
        self.error[segment, active] += error
        self.count[segment, active] += 1
        within = np.zeros(len(active), dtype=bool)
        within[active] = (error[:, :2].sum(1) <= 0.1**2) & (error[:, 2] <= 0.15**2)
        within &= self.active  # A fall on the hold-completion frame is not success.
        self.tracking_count[segment] += within
        self.streak = np.where(within, self.streak + 1, 0)
        newly = (self.streak >= 25) & np.isnan(self.settled_at[segment])
        self.settled_at[segment, newly] = (offset + 1) * 0.02
        self.survived[segment] = self.active

    def report(self):
        result = self.total.report()
        rows = []
        for i, entry in enumerate(self.schedule):
            frames = int(self.count[i].sum())
            pooled = self.error[i].sum(0) / frames if frames else None
            settled = np.isfinite(self.settled_at[i])
            rows.append(
                {
                    **entry,
                    "observed_frames": frames,
                    "entered_count": int(self.entered[i].sum()),
                    "survival_fraction": float(self.survived[i].mean()),
                    "planar_rmse": float(np.sqrt(pooled[:2].sum())) if frames else None,
                    "yaw_rmse": float(np.sqrt(pooled[2])) if frames else None,
                    "settled_fraction": float(settled.mean()),
                    "tracking_fraction": float(
                        self.tracking_count[i].mean() / self.width
                    ),
                    "mean_settle_time_s": float(self.settled_at[i, settled].mean())
                    if settled.any()
                    else None,
                    "per_env": {
                        "observed_steps": self.count[i].tolist(),
                        "tracking_steps": self.tracking_count[i].tolist(),
                        "settle_time_s": [
                            float(x) if np.isfinite(x) else None
                            for x in self.settled_at[i]
                        ],
                    },
                }
            )
        result.update(
            settled_fraction=min(row["settled_fraction"] for row in rows),
            tracking_fraction=min(row["tracking_fraction"] for row in rows),
            tracking_protocol=TRACKING_PROTOCOL.copy(),
            command_schedule=self.schedule,
            segments=rows,
            settling_protocol={
                "planar_tolerance_mps": 0.1,
                "yaw_tolerance_radps": 0.15,
                "hold_steps": 25,
                "denominator": "all initial environments",
                "time": "hold completion from segment start",
            },
        )
        return result


def validate_switching(case, steps, suite="switching"):
    schedule = switching_schedule(steps, suite)
    if case.get("command_schedule") != schedule or len(case.get("segments", [])) != 6:
        raise ValueError("Incomplete or incorrect switching schedule")
    tracking = "tracking_protocol" in case
    if tracking and case["tracking_protocol"] != TRACKING_PROTOCOL:
        raise ValueError("Incorrect switching tracking protocol")
    if not tracking and (
        "tracking_fraction" in case
        or any(
            "tracking_fraction" in segment or "tracking_steps" in segment["per_env"]
            for segment in case["segments"]
        )
    ):
        raise ValueError("Missing switching tracking protocol")
    counts = np.zeros(case["num_envs"], dtype=int)
    final_counts = np.asarray(case["per_env"]["observed_steps"])
    ended = np.asarray(case["per_env"]["terminated"]) | np.asarray(
        case["per_env"]["truncated"]
    )
    for expected, segment in zip(schedule, case["segments"], strict=True):
        if any(segment.get(key) != value for key, value in expected.items()):
            raise ValueError("Switching segment identity mismatch")
        observed = np.asarray(segment["per_env"]["observed_steps"])
        remaining = np.maximum(
            np.asarray(case["per_env"]["observed_steps"]) - expected["start_step"], 0
        )
        wanted = np.minimum(remaining, expected["end_step"] - expected["start_step"])
        if observed.shape != counts.shape or not np.array_equal(observed, wanted):
            raise ValueError("Switching segment frame accounting mismatch")
        counts += observed
        survived = (final_counts >= expected["end_step"]) & ~(
            ended & (final_counts <= expected["end_step"])
        )
        if not np.isclose(
            segment["survival_fraction"], survived.mean(), rtol=0, atol=1e-12
        ):
            raise ValueError("Switching segment survival accounting mismatch")
        if segment["observed_frames"] != int(observed.sum()) or segment[
            "entered_count"
        ] != int((observed > 0).sum()):
            raise ValueError("Switching segment coverage mismatch")
        terminal = (
            ended
            & (final_counts > expected["start_step"])
            & (final_counts <= expected["end_step"])
        )
        if tracking:
            tracked = np.asarray(segment["per_env"].get("tracking_steps"))
            fraction = segment.get("tracking_fraction")
            if (
                tracked.shape != counts.shape
                or not np.issubdtype(tracked.dtype, np.integer)
                or np.any(tracked < 0)
                or np.any(tracked > observed - terminal)
                or not isinstance(fraction, (int, float))
                or not np.isclose(
                    fraction,
                    tracked.sum() / (len(counts) * (steps // 6)),
                    rtol=0,
                    atol=1e-12,
                )
            ):
                raise ValueError("Switching tracking accounting mismatch")
        for key in ("planar_rmse", "yaw_rmse"):
            value = segment[key]
            if (value is None) != (observed.sum() == 0) or (
                value is not None and (not np.isfinite(value) or value < 0)
            ):
                raise ValueError("Invalid switching segment metric")
        times = segment["per_env"]["settle_time_s"]
        if len(times) != len(counts):
            raise ValueError("Switching settle-time count mismatch")
        finite = []
        for index, (value, count) in enumerate(
            zip(times, observed - terminal, strict=True)
        ):
            if value is not None:
                if not np.isfinite(value) or not 0.5 <= value <= count * 0.02 + 1e-12:
                    raise ValueError("Invalid switching settle time")
                if tracking and tracked[index] < 25:
                    raise ValueError(
                        "Switching settled environment lacks tracking steps"
                    )
                finite.append(value)
        mean = segment["mean_settle_time_s"]
        if (
            not np.isclose(
                segment["settled_fraction"],
                len(finite) / len(counts),
                rtol=0,
                atol=1e-12,
            )
            or (mean is None) != (len(finite) == 0)
            or (finite and not np.isclose(mean, np.mean(finite), rtol=0, atol=1e-12))
        ):
            raise ValueError("Switching settle-time summary mismatch")
    if counts.tolist() != case["per_env"]["observed_steps"]:
        raise ValueError("Switching segments do not cover the first episode")
    for key in ("planar_rmse", "yaw_rmse"):
        pooled = np.sqrt(
            sum(
                row[key] ** 2 * row["observed_frames"]
                for row in case["segments"]
                if row["observed_frames"]
            )
            / counts.sum()
        )
        if not np.isclose(case[key], pooled, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Switching {key} does not pool its segments")
    if case["settled_fraction"] != min(
        row["settled_fraction"] for row in case["segments"]
    ):
        raise ValueError("Switching overall settled fraction must be the worst segment")
    if tracking and case.get("tracking_fraction") != min(
        row["tracking_fraction"] for row in case["segments"]
    ):
        raise ValueError(
            "Switching overall tracking fraction must be the worst segment"
        )
