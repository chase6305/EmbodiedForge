"""Explicit registration, with lazy loading of optional SDKs."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from embodiedforge.core import Capabilities, Config, ControlSpec, SceneSpec, TaskSpec

NEWTON_VERSION = "1.6.0rc1"
NEWTON_SOURCE = (
    "https://github.com/newton-physics/newton/archive/refs/tags/v1.6.0rc1.zip"
)


@dataclass(frozen=True)
class Registration:
    """Lazy construction recipe and its advertised, SDK-independent features."""

    factory: Callable
    capabilities: Capabilities
    sdk_version: str | None = None
    source: str | None = None


PHYSICS: dict[str, Registration] = {}
RENDER: dict[str, Registration] = {}


def register(
    kind: str,
    name: str,
    factory: Callable,
    capabilities: Capabilities,
    *,
    sdk_version: str | None = None,
    source: str | None = None,
) -> None:
    """Register at startup; never replace an existing adapter implicitly."""
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("Backend registration requires a name and callable factory")
    if kind not in ("physics", "render"):
        raise ValueError(f"Unknown backend kind: {kind}")
    table = PHYSICS if kind == "physics" else RENDER
    if name in table:
        raise ValueError(f"Backend already registered: {kind}/{name}")
    table[name] = Registration(factory, capabilities, sdk_version, source)


def _lazy(module, name):
    return lambda: getattr(import_module(module), name)()


register(
    "physics",
    "numpy",
    _lazy("embodiedforge.backends.physics", "NumpyPhysics"),
    Capabilities(control=ControlSpec(2, "newtons_xy")),
)
register(
    "physics",
    "mujoco",
    _lazy("embodiedforge.backends.physics", "MujocoPhysics"),
    Capabilities(control=ControlSpec(2, "newtons_xy")),
)
register(
    "physics",
    "mjbatch",
    _lazy("embodiedforge.backends.mjbatch", "MjbatchPhysics"),
    Capabilities(control=ControlSpec(2, "newtons_xy")),
    sdk_version="0.1.0",
    source="https://github.com/kevinzakka/mjbatch",
)
register(
    "physics",
    "newton",
    _lazy("embodiedforge.backends.newton", "NewtonPhysics"),
    Capabilities(control=ControlSpec(2, "newtons_xy")),
    sdk_version=NEWTON_VERSION,
    source=NEWTON_SOURCE,
)
register(
    "render",
    "null",
    _lazy("embodiedforge.backends.render", "NullRenderer"),
    Capabilities(),
)
register(
    "render",
    "raster",
    _lazy("embodiedforge.backends.render", "RasterRenderer"),
    Capabilities(
        channels=frozenset({"rgb", "depth", "instance_id", "semantic_id"}),
        required_entities=frozenset({"agent", "target"}),
    ),
)


def resolve(
    config: Config, scene: SceneSpec, task: TaskSpec
) -> tuple[Registration, Registration]:
    """Resolve metadata without constructing a backend or importing its SDK."""
    from embodiedforge.sensors import CHANNEL_DTYPES

    unsupported = set(config.channels) - (CHANNEL_DTYPES.keys() - {"depth_valid"})
    if unsupported:
        raise ValueError(
            f"Sensor pipeline does not support channels: {sorted(unsupported)}"
        )
    selected = []
    for kind, name, table in (
        ("physics", config.physics, PHYSICS),
        ("render", config.render, RENDER),
    ):
        if name not in table:
            raise ValueError(
                f"Unknown {kind} backend {name!r}; available: {sorted(table)}"
            )
        entry = table[name]
        cap = entry.capabilities
        if scene.kind not in cap.scenes:
            raise ValueError(f"{kind}/{name} does not support scene {scene.kind}")
        if cap.device != "cpu" or cap.transport != "host_snapshot":
            raise ValueError("This runtime implements CPU host_snapshot transport only")
        if kind == "physics" and not (cap.partial_reset and cap.active_mask):
            raise ValueError(
                f"{name} must support partial reset and frozen done environments"
            )
        missing_entities = cap.required_entities - set(scene.entity_ids)
        if missing_entities:
            raise ValueError(
                f"{kind}/{name} requires scene entities: {sorted(missing_entities)}"
            )
        if kind == "physics":
            expected = ControlSpec(task.action_dim, task.action_units)
            if cap.control != expected:
                raise ValueError(
                    f"physics/{name} control mismatch: task requires {expected}, "
                    f"backend declares {cap.control}"
                )
        if kind == "render":
            missing = set(config.channels) - cap.channels
            if missing:
                raise ValueError(f"{name} missing render channels: {sorted(missing)}")
        selected.append(entry)
    return tuple(selected)


def execution_plan(config: Config) -> dict:
    """Describe the selected task and adapters without allocating engine resources."""
    from embodiedforge.tasks import make_task

    task = make_task(config.task)
    physics, _ = resolve(config, task.scene, task.spec)
    return {
        "physics": config.physics,
        "physics_version": physics.sdk_version,
        "physics_source": physics.source,
        "render": config.render,
        "device": "cpu",
        "transport": "host_snapshot",
        "num_envs": config.num_envs,
        "substeps": config.physics_hz // config.control_hz,
        "channels": list(config.channels),
        "task": task.spec.id,
        "scene": task.scene.kind,
        "action_dim": task.spec.action_dim,
        "action_units": task.spec.action_units,
        "proprio_dim": task.spec.proprio_dim,
    }
