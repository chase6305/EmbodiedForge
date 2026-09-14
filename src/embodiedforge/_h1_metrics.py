"""First-episode H1 metrics, including the terminal frame before auto-reset."""

import numpy as np

BASIC_COMMANDS = {
    "stand": [0.0, 0.0, 0.0],
    "forward": [0.5, 0.0, 0.0],
    "turn_left": [0.5, 0.0, 0.5],
    "turn_right": [0.5, 0.0, -0.5],
}


class FirstEpisodeMetrics:
    def __init__(self, num_envs: int, steps: int, dt: float):
        if num_envs <= 0 or steps <= 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid evaluation dimensions or timestep")
        self.limit, self.dt = steps, dt
        self.active = np.ones(num_envs, dtype=bool)
        self.count = np.zeros(num_envs, dtype=int)
        self.error_sum = np.zeros((num_envs, 3))
        self.velocity_sum = np.zeros((num_envs, 3))
        self.terminated = np.zeros(num_envs, dtype=bool)
        self.truncated = np.zeros(num_envs, dtype=bool)
        self.steps = 0

    def update(self, velocity, command, terminated, truncated):
        velocity, command = np.asarray(velocity), np.asarray(command)
        terminated, truncated = np.asarray(terminated), np.asarray(truncated)
        n = len(self.active)
        if velocity.shape != (n, 3) or command.shape != (n, 3):
            raise ValueError("Expected velocity and command arrays of shape (N, 3)")
        if terminated.shape != (n,) or truncated.shape != (n,):
            raise ValueError("Expected termination arrays of shape (N,)")
        if self.steps >= self.limit:
            raise ValueError("Evaluation exceeded its step limit")
        if (
            not np.isfinite(velocity[self.active]).all()
            or not np.isfinite(command[self.active]).all()
        ):
            raise ValueError("Non-finite evaluation state")
        self.error_sum[self.active] += (
            velocity[self.active] - command[self.active]
        ) ** 2
        self.velocity_sum[self.active] += velocity[self.active]
        self.count[self.active] += 1
        self.terminated |= self.active & terminated.astype(bool)
        self.truncated |= self.active & truncated.astype(bool)
        self.active &= ~(terminated.astype(bool) | truncated.astype(bool))
        self.steps += 1

    def report(self):
        if (self.count == 0).any() or (self.steps < self.limit and self.active.any()):
            raise ValueError("Incomplete first-episode evaluation")
        means = self.error_sum / self.count[:, None]
        pooled = self.error_sum.sum(axis=0) / self.count.sum()
        return {
            "num_envs": len(self.count),
            "steps_limit": self.limit,
            "steps_executed": self.steps,
            "dt": self.dt,
            "horizon_seconds": self.limit * self.dt,
            "survival_fraction": float(self.active.mean()),
            "fall_fraction": float(self.terminated.mean()),
            "truncation_fraction": float(self.truncated.mean()),
            "mean_observed_seconds": float(self.count.mean() * self.dt),
            "planar_rmse": float(np.sqrt(pooled[:2].sum())),
            "yaw_rmse": float(np.sqrt(pooled[2])),
            "mean_velocity": (
                self.velocity_sum.sum(axis=0) / self.count.sum()
            ).tolist(),
            "per_env": {
                "observed_steps": self.count.tolist(),
                "terminated": self.terminated.tolist(),
                "truncated": self.truncated.tolist(),
                "planar_rmse": np.sqrt(means[:, :2].sum(axis=1)).tolist(),
                "yaw_rmse": np.sqrt(means[:, 2]).tolist(),
                "mean_velocity": (self.velocity_sum / self.count[:, None]).tolist(),
            },
        }


def assess(reports: list[dict], criteria: dict) -> dict:
    if not reports:
        raise ValueError("No evaluation reports")
    checks = []
    for report in reports:
        for name, limit in criteria.items():
            metric = {
                "max_planar_rmse": "planar_rmse",
                "max_yaw_rmse": "yaw_rmse",
                "min_survival_fraction": "survival_fraction",
            }[name]
            value = report[metric]
            if not np.isfinite(value) or not np.isfinite(limit):
                raise ValueError("Non-finite acceptance metric or limit")
            passed = value >= limit if name.startswith("min_") else value <= limit
            checks.append(
                {
                    "seed": report["seed"],
                    "case": report.get("case", "custom"),
                    "criterion": name,
                    "value": value,
                    "limit": limit,
                    "passed": bool(passed),
                }
            )
    return {
        "criteria": criteria,
        "checks": checks,
        "passed": all(check["passed"] for check in checks) if checks else None,
    }


def validate_first_episode(report):
    """Check frame coverage and pooled summaries against per-environment data."""
    n, limit, executed = (
        report.get(key) for key in ("num_envs", "steps_limit", "steps_executed")
    )
    dt = report.get("dt")
    if (
        any(type(value) is not int or value <= 0 for value in (n, limit, executed))
        or executed > limit
        or not isinstance(dt, (int, float))
        or isinstance(dt, bool)
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError("Invalid first-episode dimensions or timestep")
    rows = report.get("per_env", {})
    counts = np.asarray(rows.get("observed_steps"))
    terminated = np.asarray(rows.get("terminated"))
    truncated = np.asarray(rows.get("truncated"))
    if (
        counts.shape != (n,)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 1)
        or np.any(counts > executed)
        or terminated.shape != (n,)
        or terminated.dtype != np.bool_
        or truncated.shape != (n,)
        or truncated.dtype != np.bool_
    ):
        raise ValueError("Invalid first-episode per-environment frame accounting")
    active = ~(terminated | truncated)
    if np.any(active & (counts != limit)):
        raise ValueError("Incomplete first-episode coverage")
    expected = {
        "horizon_seconds": limit * dt,
        "mean_observed_seconds": counts.mean() * dt,
        "survival_fraction": active.mean(),
        "fall_fraction": terminated.mean(),
        "truncation_fraction": truncated.mean(),
    }
    for key in ("planar_rmse", "yaw_rmse", "mean_velocity"):
        values = np.asarray(rows.get(key))
        shape = (n, 3) if key == "mean_velocity" else (n,)
        if (
            values.shape != shape
            or not np.issubdtype(values.dtype, np.number)
            or np.iscomplexobj(values)
            or not np.isfinite(values).all()
            or (key != "mean_velocity" and np.any(values < 0))
        ):
            raise ValueError(f"Invalid first-episode per-environment {key}")
        expected[key] = (
            np.average(values, axis=0, weights=counts)
            if key == "mean_velocity"
            else np.sqrt(np.average(values**2, weights=counts))
        )
    for key, wanted in expected.items():
        value = np.asarray(report.get(key))
        if (
            value.shape != np.shape(wanted)
            or not np.issubdtype(value.dtype, np.number)
            or np.iscomplexobj(value)
            or not np.isfinite(value).all()
            or not np.allclose(value, wanted, rtol=1e-10, atol=1e-12)
        ):
            raise ValueError(f"First-episode {key} summary mismatch")
