"""Task-specific recipe result and acceptance checks, independent of the CLI."""

import json
import math

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import file_sha256 as sha256
from embodiedforge.runtime.records import RunRejected

from .recipe_catalog import SWITCHING_SUITES


class EvaluationRejected(RunRejected):
    """The simulation completed but did not meet the requested criteria."""


def criteria_for(args):
    names = (
        ("min_success_rate", "max_drop_rate")
        if args.task.startswith("wuji-")
        else ("min_survival_fraction", "max_planar_rmse", "max_yaw_rmse")
    )
    if args.task == "go1-joystick" and getattr(args, "suite", None) in SWITCHING_SUITES:
        names += ("min_settled_fraction", "min_tracking_fraction")
    criteria = {}
    for name in (
        "min_success_rate",
        "max_drop_rate",
        "min_survival_fraction",
        "max_planar_rmse",
        "max_yaw_rmse",
        "min_settled_fraction",
        "min_tracking_fraction",
    ):
        value = getattr(args, name, None)
        if value is None:
            continue
        if name not in names:
            raise ValueError(f"{name} does not apply to {args.task}")
        if (
            not math.isfinite(value)
            or value < 0
            or (("rate" in name or "fraction" in name) and value > 1)
        ):
            raise ValueError(f"Invalid acceptance threshold: {name}")
        criteria[name] = value
    return criteria


def assess(result, criteria):
    if "cases" in result:
        checks = []
        for case in result["cases"]:
            checks.extend(
                {**item, "seed": case["seed"], "case": case["case"]}
                for item in assess(case, criteria)["checks"]
            )
        return {
            "criteria": criteria,
            "checks": checks,
            "passed": all(item["passed"] for item in checks) if checks else None,
        }
    if "segments" in result:
        checks = assess({k: v for k, v in result.items() if k != "segments"}, criteria)[
            "checks"
        ]
        for segment in result["segments"]:
            checks.extend(
                {**item, "segment": segment["segment"]}
                for item in assess(segment, criteria)["checks"]
            )
        return {
            "criteria": criteria,
            "checks": checks,
            "passed": all(item["passed"] for item in checks) if checks else None,
        }
    checks = []
    for name, threshold in criteria.items():
        metric = name.split("_", 1)[1]
        actual = result[metric]
        unobserved = actual is None and result.get("observed_frames") == 0
        if not unobserved and not math.isfinite(actual):
            raise ValueError(f"Non-finite evaluation metric: {metric}")
        passed = (
            False
            if unobserved
            else actual >= threshold
            if name.startswith("min_")
            else actual <= threshold
        )
        checks.append(
            {
                "criterion": name,
                "actual": actual,
                "threshold": threshold,
                "passed": passed,
            }
        )
    return {
        "criteria": criteria,
        "checks": checks,
        "passed": all(item["passed"] for item in checks) if checks else None,
    }


def validate_result(request, result, output):
    if request.get("task") == "go1-joystick":
        for item in result.get("cases", [result]):
            if item.get("learning_rate_override") != request.get("go1_learning_rate"):
                raise ValueError("Go1 result learning rate differs from request")
    if request.get("go1_command_profile"):
        for item in result.get("cases", [result]):
            if item.get("command_profile") != request["go1_command_profile"]:
                raise ValueError("Go1 result command profile differs from request")
    if request.get("go1_reward_profile"):
        for item in result.get("cases", [result]):
            if item.get("reward_profile") != request["go1_reward_profile"]:
                raise ValueError("Go1 result reward profile differs from request")
    if request.get("go1_semantics"):
        for item in result.get("cases", [result]):
            if item.get("task_semantics") != request["go1_semantics"]:
                raise ValueError("Go1 result task semantics differs from request")
    if request["command"] == "train":
        path = (output / result["checkpoint"]).resolve()
        if not path.is_relative_to(output) or not path.is_file():
            raise ValueError("Worker checkpoint is missing or outside its run")
        if sha256(path) != result["checkpoint_sha256"]:
            raise ValueError("Worker checkpoint SHA256 mismatch")
        if (
            result["completed_updates"] != request["updates"]
            or result["checkpoint_iteration"]
            != request.get("start_iteration", 0) + request["updates"] - 1
            or result["finite_tensor_count"] <= 0
        ):
            raise ValueError("Incomplete training result")
    elif request["command"] == "evaluate":
        if request["task"].startswith("wuji-"):
            from .._wuji_diagnostics import validate_evaluation

            validate_evaluation(request, result)
            if request.get("record_video"):
                from .._wuji_diagnostics import verify_video

                write_json(
                    output / "video-validation.json", verify_video(result, output)
                )
        else:
            from .._go1_metrics import evaluation_commands
            from .._h1_metrics import validate_first_episode

            commands = evaluation_commands(
                request.get("suite"), request.get("velocity")
            )
            expected = {
                (seed, case)
                for seed in (request.get("seeds") or [request["seed"]])
                for case in commands
            }
            cases = result.get("cases", [result])
            found = [(case["seed"], case["case"]) for case in cases]
            if set(found) != expected or len(found) != len(expected):
                raise ValueError("Missing, duplicate or unexpected Go1 evaluation case")
            for case in cases:
                validate_first_episode(case)
                if (
                    case["num_envs"] != request["num_envs"]
                    or case["steps_limit"] != request["steps"]
                    or case["command"] != commands[case["case"]]
                ):
                    raise ValueError(
                        "Evaluation dimensions or command differ from request"
                    )
                if request.get("suite") in SWITCHING_SUITES:
                    from .._go1_metrics import validate_switching

                    validate_switching(case, request["steps"], request["suite"])
                if request.get("record_motion"):
                    from .._h1_motion import load_motion

                    motion = case["motion"]
                    path = (output / motion["path"]).resolve()
                    if (
                        not path.is_relative_to(output)
                        or sha256(path) != motion["sha256"]
                    ):
                        raise ValueError("Motion SHA256 mismatch")
                    trace = load_motion(path)
                    if len(trace["time"]) != case["per_env"]["observed_steps"][0]:
                        raise ValueError("Motion frame count differs from evaluation")
                    metadata = trace["metadata"]
                    if (
                        request.get("suite") in SWITCHING_SUITES
                        and metadata.get("command_schedule") != case["command_schedule"]
                    ):
                        raise ValueError(
                            "Motion command schedule differs from evaluation"
                        )
                    if (
                        metadata["case"] != case["case"]
                        or metadata["seed"] != case["seed"]
                        or metadata["env_id"] != 0
                        or metadata["quaternion_order"] != "xyzw"
                        or metadata["velocity_command"] != case["command"]
                        or not math.isclose(metadata["dt"], case["dt"], abs_tol=1e-12)
                        or trace["terminated"] != case["per_env"]["terminated"][0]
                        or trace["truncated"] != case["per_env"]["truncated"][0]
                    ):
                        raise ValueError(
                            "Motion metadata or final flags differ from evaluation"
                        )
    json.dumps(result, allow_nan=False)
