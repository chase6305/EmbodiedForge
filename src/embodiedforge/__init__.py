"""Public core exports, loaded when requested by a caller.

Importing a runtime or provenance utility does not load NumPy, an environment,
or a simulation/training SDK. Core types keep their original defining modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import (
        Config,
        PhysicsBackend,
        Policy,
        RenderBackend,
        StepResult,
        Task,
        TaskSpec,
    )
    from .env import VectorEnv

__all__ = [
    "Config",
    "PhysicsBackend",
    "Policy",
    "RenderBackend",
    "StepResult",
    "Task",
    "TaskSpec",
    "VectorEnv",
]
__version__ = "0.1.0"


def __getattr__(name):
    if name == "VectorEnv":
        from .env import VectorEnv

        value = VectorEnv
    elif name in __all__:
        from . import core

        value = getattr(core, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
