"""Dependency-free aggregation of comparable Microduck evaluation reports."""

import hashlib
import json
import math
import statistics
from pathlib import Path


def load_evaluation_run(directory: Path) -> tuple[list[dict], dict, dict[str, bytes]]:
    """Read a completed evaluation once; retain exact inputs for an offline audit."""
    directory = directory.expanduser().resolve()
    inputs = {}

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite value in evaluation JSON")
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    def read(relative):
        raw = (directory / relative).read_bytes()
        value = json.loads(
            raw, parse_float=finite_float, parse_constant=invalid_constant
        )
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object: {relative}")
        inputs[relative] = raw
        return value

    def completed(manifest, workflow):
        if manifest.get("schema") != 1 or manifest.get("workflow") != workflow:
            raise ValueError(f"Expected schema-1 {workflow} run")
        if manifest.get("status") not in ("complete", "rejected") or not manifest.get(
            "finished_at"
        ):
            raise ValueError(
                "Evaluation run did not finish; incomplete runs cannot be assessed"
            )

    def read_seed(prefix, manifest):
        completed(manifest, "evaluate")
        report = read(prefix + "evaluation.json")
        read(prefix + "runtime.json")
        try:
            commands = [
                command
                for command in manifest["commands"]
                if len(command) >= 10 and command[3] == "evaluate"
            ]
            if len(commands) != 1:
                raise ValueError("Missing or ambiguous evaluation command")
            command = commands[0]
            for key, value in (
                ("seed", int(command[7])),
                ("num_envs", int(command[5])),
                ("steps_per_env", int(command[6])),
                ("checkpoint", command[4]),
            ):
                if report[key] != value:
                    raise ValueError(
                        f"Evaluation report disagrees with run manifest: {key}"
                    )
            options = manifest["evaluation_options"]
            if json.loads(command[9]) != options:
                raise ValueError("Evaluation options disagree with recorded command")
            conditions = report["conditions"]
            if conditions["velocity_body_frame"] != options["velocity"]:
                raise ValueError("Reported velocity differs from requested velocity")
            if options["no_pushes"] and conditions["pushes_enabled"]:
                raise ValueError("Reported pushes differ from requested settings")
            if bool(options["onnx"]) != (report["onnx_parity"] is not None):
                raise ValueError("Reported ONNX check differs from requested settings")
            if (
                options.get("curriculum_step") is not None
                and conditions.get("curriculum_start_step")
                != options["curriculum_step"]
            ):
                raise ValueError("Reported curriculum differs from requested settings")
            if "curriculum_start_step" in conditions:
                override = options.get("curriculum_step")
                expected = (
                    report["checkpoint_metadata"]["common_step_counter"]
                    if override is None
                    else override
                )
                actual = conditions["curriculum_start_step"]
                if (
                    type(actual) is not int
                    or actual < 0
                    or actual != expected
                    or conditions.get("curriculum_source")
                    != ("checkpoint" if override is None else "override")
                ):
                    raise ValueError(
                        "Reported curriculum source or counter is inconsistent"
                    )
            if report["task"] != manifest["task"]:
                raise ValueError("Reported task differs from run manifest")
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError(f"Incomplete evaluation manifest/report: {exc}") from exc
        return report

    manifest = read("run.json")
    workflow = manifest.get("workflow")
    if workflow == "evaluate":
        reports = [read_seed("", manifest)]
    elif workflow == "evaluate_seeds":
        completed(manifest, workflow)
        seeds = manifest.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
        ):
            raise ValueError("Invalid batch seeds")
        if len(set(seeds)) != len(seeds) or manifest.get("completed_seeds") != seeds:
            raise ValueError("Incomplete or duplicate batch seeds")
        reports = []
        for seed in seeds:
            prefix = f"seed-{seed}/"
            child = read(prefix + "run.json")
            for key in ("source", "task", "evaluation_options"):
                if child.get(key) != manifest.get(key):
                    raise ValueError(f"Child run disagrees with batch: {key}")
            report = read_seed(prefix, child)
            if report["seed"] != seed or report.get("checkpoint_metadata", {}).get(
                "sha256"
            ) != manifest.get("checkpoint_sha256"):
                raise ValueError("Child seed or checkpoint disagrees with batch")
            reports.append(report)
    else:
        raise ValueError("Expected an evaluate or evaluate_seeds run directory")
    aggregate_evaluations(reports)
    provenance = {
        "directory": str(directory),
        "source": manifest.get("source"),
        "status": manifest["status"],
        "files": [
            {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}
            for name, raw in inputs.items()
        ],
    }
    return reports, provenance, inputs


def validate_criteria(criteria: dict) -> dict:
    limits = {
        "max_planar_rmse": None,
        "max_yaw_rmse": None,
        "min_survival_fraction": 1.0,
    }
    for name, value in criteria.items():
        if name not in limits:
            raise ValueError(f"Unknown evaluation criterion: {name}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or (limits[name] is not None and value > limits[name])
        ):
            raise ValueError(f"Invalid evaluation criterion {name}: {value}")
    return dict(criteria)


def compare_evaluations(before: list[dict], after: list[dict]) -> dict:
    """Pair identical evaluation seeds/conditions across two checkpoints."""
    baseline, candidate = aggregate_evaluations(before), aggregate_evaluations(after)
    if (
        "curriculum_start_step" not in baseline["conditions"]
        or "curriculum_start_step" not in candidate["conditions"]
    ):
        raise ValueError(
            "Comparison requires reports with an explicit curriculum start; rerun with --curriculum-step"
        )
    for key in (
        "task",
        "conditions",
        "num_envs",
        "steps_per_env",
        "sim_seconds_per_env",
        "velocity_axes",
        "velocity_units",
    ):
        if baseline[key] != candidate[key]:
            raise ValueError(
                f"Comparison conditions differ: {key}; use identical evaluation settings"
            )
    for key in ("frame", "sampling", "includes_terminal_steps"):
        if before[0]["velocity_tracking"][key] != after[0]["velocity_tracking"][key]:
            raise ValueError(f"Comparison sampling differs: {key}")
    if set(baseline["seeds"]) != set(candidate["seeds"]):
        raise ValueError("Comparison requires identical seed sets")

    def parity_settings(summary):
        value = summary["onnx_parity_configuration"]
        return (
            {key: item for key, item in value.items() if key != "sha256"}
            if value is not None
            else None
        )

    if parity_settings(baseline) != parity_settings(candidate):
        raise ValueError("Comparison ONNX inference settings differ")
    candidates = {row["seed"]: row for row in candidate["per_seed"]}
    pairs = []
    for row in baseline["per_seed"]:
        other = candidates[row["seed"]]
        pairs.append(
            {
                "seed": row["seed"],
                "before": {key: value for key, value in row.items() if key != "seed"},
                "after": {key: value for key, value in other.items() if key != "seed"},
                "delta": {
                    key: other[key] - value
                    for key, value in row.items()
                    if key != "seed"
                },
            }
        )
    metrics = {}
    for key in pairs[0]["delta"]:
        changes = [pair["delta"][key] for pair in pairs]
        metrics[key] = {
            "before_mean": baseline["metrics"][key]["mean"],
            "after_mean": candidate["metrics"][key]["mean"],
            "mean_delta": statistics.mean(changes),
            "delta_sample_std": statistics.stdev(changes) if len(changes) > 1 else None,
        }
    return {
        "schema": 1,
        "definition": "Paired delta = after - before; no significance or overall pass/fail is inferred.",
        "before_checkpoint": baseline["checkpoint_metadata"],
        "after_checkpoint": candidate["checkpoint_metadata"],
        "conditions": baseline["conditions"],
        "task": baseline["task"],
        "seeds": baseline["seeds"],
        "num_envs": baseline["num_envs"],
        "steps_per_env": baseline["steps_per_env"],
        "sim_seconds_per_env": baseline["sim_seconds_per_env"],
        "metrics": metrics,
        "per_seed": pairs,
    }


def assess_evaluations(reports: list[dict], criteria: dict) -> dict:
    """Apply explicit limits to EVERY seed; averages cannot mask a failed seed."""
    criteria = validate_criteria(criteria)
    if not criteria:
        raise ValueError("At least one evaluation criterion is required")
    summary = aggregate_evaluations(reports)
    checks = []
    for row in summary["per_seed"]:
        survivors = row["survived_full_horizon_count"]
        if type(survivors) is not int or not 0 <= survivors <= summary["num_envs"]:
            raise ValueError("Invalid full-horizon survivor count")
        values = {
            "max_planar_rmse": row["planar_velocity_rmse_m_s"],
            "max_yaw_rmse": row["wz_rmse"],
            "min_survival_fraction": survivors / summary["num_envs"],
        }
        for name, threshold in criteria.items():
            actual = values[name]
            if actual < 0:
                raise ValueError(f"Invalid negative metric for {name}")
            minimum = name == "min_survival_fraction"
            checks.append(
                {
                    "seed": row["seed"],
                    "criterion": name,
                    "actual": actual,
                    "unit": {
                        "max_planar_rmse": "m/s",
                        "max_yaw_rmse": "rad/s",
                        "min_survival_fraction": "fraction",
                    }[name],
                    "operator": ">=" if minimum else "<=",
                    "threshold": threshold,
                    "passed": actual >= threshold if minimum else actual <= threshold,
                }
            )
    return {
        "schema": 1,
        "criteria": criteria,
        "scope": "Every seed must meet every criterion; survival concerns initial episodes over the configured horizon.",
        "checkpoint_metadata": summary["checkpoint_metadata"],
        "conditions": summary["conditions"],
        "sim_seconds_per_env": summary["sim_seconds_per_env"],
        "num_envs": summary["num_envs"],
        "steps_per_env": summary["steps_per_env"],
        "seeds": summary["seeds"],
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "failed_seeds": sorted(
            {check["seed"] for check in checks if not check["passed"]}
        ),
    }


def aggregate_evaluations(reports: list[dict]) -> dict:
    """Use seeds as replicates; pool squared errors, never average RMSE as a pool."""
    if not reports:
        raise ValueError("No evaluation reports to aggregate")
    try:
        return _aggregate(reports)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(f"Incomplete evaluation report: {exc}") from exc


def _validate_per_environment(tracking: dict, num_envs: int, steps: int) -> None:
    detail = tracking.get("per_environment")
    if detail is None:
        return  # Older reports contain only the batch accumulators.
    if (
        type(detail["samples_per_environment"]) is not int
        or detail["samples_per_environment"] != steps
    ):
        raise ValueError("Incomplete per-environment sample count")
    for name in ("mean_command", "mean_actual", "bias", "mae", "rmse"):
        values = detail[name]
        if len(values) != num_envs or any(len(row) != 3 for row in values):
            raise ValueError("Invalid per-environment velocity shape")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (name in ("mae", "rmse") and value < 0)
            for row in values
            for value in row
        ):
            raise ValueError("Invalid per-environment velocity metric")

    def close(actual, expected):
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in (actual, expected)
        ):
            raise ValueError("Invalid per-environment aggregate metric")
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Per-environment metrics disagree with reported totals")

    for axis in range(3):
        for name in ("mean_command", "mean_actual", "mae"):
            close(
                statistics.mean(row[axis] for row in detail[name]), tracking[name][axis]
            )
        close(
            math.sqrt(statistics.mean(row[axis] ** 2 for row in detail["rmse"])),
            tracking["rmse"][axis],
        )
        for index in range(num_envs):
            close(
                detail["bias"][index][axis],
                detail["mean_actual"][index][axis]
                - detail["mean_command"][index][axis],
            )
    planar = detail["planar_velocity_rmse_m_s"]
    if len(planar) != num_envs:
        raise ValueError("Invalid per-environment planar RMSE shape")
    for value, row in zip(planar, detail["rmse"], strict=True):
        close(value, math.hypot(*row[:2]))


def _aggregate(reports: list[dict]) -> dict:
    first = reports[0]
    invariants = (
        "task",
        "checkpoint_metadata",
        "conditions",
        "num_envs",
        "steps_per_env",
        "sim_seconds_per_env",
    )
    seeds = [report["seed"] for report in reports]
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Invalid evaluation seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate evaluation seeds")
    if any(
        type(first[key]) is not int or first[key] <= 0
        for key in ("num_envs", "steps_per_env")
    ):
        raise ValueError("Invalid evaluation counts")

    def number(value):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Nonfinite or invalid evaluation metric")
        return value

    def vector(value):
        if len(value) != 3:
            raise ValueError("Expected three velocity axes")
        return [number(item) for item in value]

    def parity_identity(report):
        parity = report["onnx_parity"]
        if parity is None:
            return None
        return {
            key: parity[key]
            for key in ("sha256", "provider", "atol", "rtol", "sampling")
        }

    rows = []
    for report in reports:
        for key in invariants:
            if report[key] != first[key]:
                raise ValueError(f"Incompatible evaluation reports: {key}")
        if parity_identity(report) != parity_identity(first):
            raise ValueError("Incompatible ONNX parity configuration")
        samples = report["num_envs"] * report["steps_per_env"]
        tracking = report["velocity_tracking"]
        if report["transitions"] != samples or tracking["samples"] != samples:
            raise ValueError("Incomplete evaluation sample count")
        _validate_per_environment(tracking, report["num_envs"], report["steps_per_env"])
        for key in ("axes", "units", "frame", "sampling", "includes_terminal_steps"):
            if tracking[key] != first["velocity_tracking"][key]:
                raise ValueError(f"Incompatible velocity tracking: {key}")
        if (
            report["finite_observations_actions_rewards"] is not True
            or report["termination_counts"]["nan_state"] != 0
        ):
            raise ValueError("Evaluation reported nonfinite simulation state")
        initial = report["initial_episodes"]
        if len(initial["duration_steps"]) != report["num_envs"]:
            raise ValueError("Incomplete initial episode observations")
        rmse = vector(tracking["rmse"])
        if any(value < 0 for value in rmse):
            raise ValueError("Negative RMSE")
        planar = number(tracking["planar_velocity_rmse_m_s"])
        if not math.isclose(planar, math.hypot(*rmse[:2]), rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Planar RMSE disagrees with velocity axes")
        row = {
            "seed": report["seed"],
            "mean_reward_per_transition": number(report["mean_reward_per_transition"]),
            "planar_velocity_rmse_m_s": number(tracking["planar_velocity_rmse_m_s"]),
            "initial_mean_observed_duration_seconds": number(
                initial["mean_observed_duration_seconds"]
            ),
            "survived_full_horizon_count": number(
                initial["survived_full_horizon_count"]
            ),
            "censored_before_horizon_count": number(
                initial["censored_before_horizon_count"]
            ),
            "falls": number(report["termination_counts"]["fell_over"]),
        }
        for axis, value in zip(("vx", "vy", "wz"), rmse, strict=True):
            row[f"{axis}_rmse"] = value
        for axis, command, actual in zip(
            ("vx", "vy", "wz"),
            vector(tracking["mean_command"]),
            vector(tracking["mean_actual"]),
            strict=True,
        ):
            row[f"{axis}_mean_command"] = command
            row[f"{axis}_mean_actual"] = actual
            row[f"{axis}_bias"] = actual - command
        rows.append(row)

    def stats(values):
        return {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
        }

    return {
        "schema": 1,
        **{key: first[key] for key in invariants},
        "seeds": seeds,
        "seed_count": len(seeds),
        "transitions": sum(report["transitions"] for report in reports),
        "aggregation": "Equal-size seed runs; sample standard deviation is across seeds, not environments. No confidence interval is implied.",
        "metrics": {
            key: stats([row[key] for row in rows]) for key in rows[0] if key != "seed"
        },
        "pooled_velocity_rmse": [
            math.sqrt(
                statistics.mean(
                    report["velocity_tracking"]["rmse"][axis] ** 2 for report in reports
                )
            )
            for axis in range(3)
        ],
        "velocity_axes": first["velocity_tracking"]["axes"],
        "velocity_units": first["velocity_tracking"]["units"],
        "onnx_parity_configuration": parity_identity(first),
        "per_seed": rows,
    }
