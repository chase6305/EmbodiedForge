"""Environment policy for isolated conda-based SDK subprocesses."""

import os
from pathlib import Path


def child_environment(environment: Path) -> dict[str, str]:
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(name, None)
    env.update(
        CONDA_PREFIX=str(environment),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        WANDB_MODE="disabled",
    )
    env["PATH"] = str(environment / "bin") + os.pathsep + env.get("PATH", "")
    return env
