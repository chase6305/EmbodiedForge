"""Record the loaded training entry points without importing optional SDKs."""

import hashlib
import inspect
import json
import sys
from importlib import metadata
from pathlib import Path


def component(value):
    """Describe the actual class/function/module, not a configured module name."""
    if not (
        inspect.isclass(value) or inspect.isroutine(value) or inspect.ismodule(value)
    ):
        value = type(value)
    module = value if inspect.ismodule(value) else inspect.getmodule(value)
    name = getattr(module, "__name__", getattr(value, "__module__", "unknown"))
    source = getattr(module, "__file__", None)
    path = Path(source).resolve() if source else None
    return {
        "module": name,
        "symbol": getattr(value, "__qualname__", getattr(value, "__name__", None)),
        "project_namespace": name == "embodiedforge"
        or name.startswith("embodiedforge."),
        "file": str(path) if path else None,
        "python_source_sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path and path.suffix == ".py"
            else None
        ),
    }


def record_training_runtime(
    directory,
    *,
    environment,
    task,
    learner,
    physics_adapter,
    physics,
    core_vector_env,
    packages,
):
    """Write one report per new run; this is provenance, not a checkpoint lock."""
    versions = {}
    for package in sorted(set(packages)):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "schema_version": 1,
        "scope": "loaded_entry_points_not_full_dependency_graph",
        "python_executable": sys.executable,
        "environment": component(environment),
        "task": component(task),
        "learner": component(learner),
        "physics_adapter": component(physics_adapter),
        "physics": physics,
        "simulation_device": "cpu",
        "learner_device": "cpu",
        "core_vector_env": core_vector_env,
        "versions": versions,
    }
    path = Path(directory) / "training-runtime.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return report
