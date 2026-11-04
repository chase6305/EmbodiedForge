"""CPU environment layout versions, shared by evaluation and episode storage."""

import math

ENVIRONMENT_SPEC_VERSION = 1
DONE_HANDLING = frozenset({"freeze", "reset_before_next_step"})


def _finite_number(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_environment_spec(spec: dict) -> None:
    """Validate without mutation; unversioned legacy layouts remain supported.

    Version 1 describes owned NumPy observations on CPU, explicit reset, and
    post-action/pre-reset transitions. It requires declared done handling so a
    consumer cannot infer row freezing from the shared method signatures.
    This checks a declared layout, not physics, assets or policy compatibility.
    """
    if not isinstance(spec, dict):
        raise ValueError("Environment specification must be a mapping")
    if "schema_version" in spec:
        version = spec["schema_version"]
        if type(version) is not int or version != ENVIRONMENT_SPEC_VERSION:
            raise ValueError("Unsupported environment specification schema_version")
        if "done_handling" not in spec:
            raise ValueError(
                "Versioned environment specification requires done_handling"
            )
    if "done_handling" in spec and (
        not isinstance(spec["done_handling"], str)
        or spec["done_handling"] not in DONE_HANDLING
    ):
        raise ValueError("Unsupported environment done_handling")
    if spec.get("device") != "cpu":
        raise ValueError("A CPU environment specification is required")
    if type(spec.get("num_envs")) is not int or spec["num_envs"] <= 0:
        raise ValueError("Environment count must be a positive integer")
    for name in ("action_shape", "proprio_shape"):
        shape = spec.get(name)
        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] <= 0
        ):
            raise ValueError(f"Invalid environment {name}")
    for name in ("task", "action_units"):
        if not isinstance(spec.get(name), str) or not spec[name].strip():
            raise ValueError(f"Environment {name} must be a nonempty name")
    interval = spec.get("control_dt")
    if not _finite_number(interval) or interval <= 0:
        raise ValueError("Environment control_dt must be finite and positive")
    bounds = spec.get("action_range")
    if "action_range" not in spec or (
        bounds is not None
        and (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or not all(_finite_number(value) for value in bounds)
            or bounds[0] >= bounds[1]
        )
    ):
        raise ValueError(
            "Environment action_range must be ordered finite bounds or None"
        )
