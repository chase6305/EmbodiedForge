"""Read installation metadata without importing SDKs or allocating devices."""

from importlib import metadata

from .backends import NEWTON_VERSION, execution_plan
from .core import Config


def installation_report(config: Config) -> dict:
    """Report selected built-in dependencies and optional training availability.

    This is a metadata preflight, not a solver, CUDA or binary ABI smoke test.
    Custom adapters have unknown dependencies and require their own validation.
    Newton and mjbatch have exact SDK version contracts checked here.
    """
    plan = execution_plan(config)
    dependencies = {
        "numpy": ("numpy",),
        "mujoco": ("numpy", "mujoco"),
        "mjbatch": ("numpy", "mujoco", "mjbatch"),
        "newton": ("numpy", "newton", "warp-lang"),
    }
    required = dependencies.get(config.physics, ("numpy",))
    packages = []
    for name in (*required, "torch"):
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        expected = NEWTON_VERSION if name == "newton" else None
        if config.physics == "mjbatch":
            expected = {"mjbatch": "0.1.0", "mujoco": "3.11.0"}.get(name)
        status = "present"
        if installed is None:
            status = "missing"
        elif expected and installed != expected:
            status = "version_mismatch"
        packages.append(
            {
                "name": name,
                "installed": installed,
                "expected": expected,
                "required_for_simulation": name in required,
                "status": status,
            }
        )
    known = config.physics in dependencies and config.render in ("null", "raster")
    return {
        "plan": plan,
        "check_scope": "installation_metadata_only",
        "dependencies_known": known,
        "metadata_ok": known
        and all(
            item["status"] == "present"
            for item in packages
            if item["required_for_simulation"]
        ),
        "packages": packages,
    }
