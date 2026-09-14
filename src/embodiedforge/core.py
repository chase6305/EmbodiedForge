"""CPU reference contracts; GPU adapters will add device-native state transport."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray
Observation = dict[str, Array]


@dataclass(frozen=True)
class Config:
    """Validated run configuration; names resolve only through explicit registries."""

    task: str = "reach"
    physics: str = "numpy"
    render: str = "null"
    num_envs: int = 16
    seed: int = 42
    control_hz: int = 50
    physics_hz: int = 200
    camera_hz: int = 25
    image_size: int = 64
    max_steps: int = 100
    channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("task", "physics", "render"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty registered name")
        if isinstance(self.channels, str) or not isinstance(
            self.channels, (tuple, list)
        ):
            raise ValueError("channels must be a sequence of names")
        object.__setattr__(self, "channels", tuple(self.channels))
        if any(not isinstance(c, str) or not c for c in self.channels):
            raise ValueError("channels must contain nonempty names")
        for key in (
            "num_envs",
            "control_hz",
            "physics_hz",
            "camera_hz",
            "image_size",
            "max_steps",
        ):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be divisible by control_hz")
        if self.control_hz % self.camera_hz:
            raise ValueError(
                "reference runtime requires camera_hz to divide control_hz"
            )
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("duplicate render channels")

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load strict JSON configuration; unknown keys fail rather than being ignored."""
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def to_dict(self) -> dict:
        """Return a serializable copy of all run configuration fields."""
        return asdict(self)


@dataclass(frozen=True)
class SceneSpec:
    """First portable scene: a unit-mass sphere constrained to the XY plane."""

    kind: str = "point_reach"
    entity_ids: tuple[str, ...] = ("agent", "target")
    radius: float = 0.06
    mass: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("Scene kind must be a nonempty name")
        if not isinstance(self.entity_ids, (tuple, list)) or any(
            not isinstance(name, str) or not name for name in self.entity_ids
        ):
            raise ValueError("Scene entity IDs must be a sequence of nonempty names")
        object.__setattr__(self, "entity_ids", tuple(self.entity_ids))
        if not (
            np.isfinite(self.mass)
            and self.mass > 0
            and np.isfinite(self.radius)
            and self.radius > 0
        ):
            raise ValueError("Scene mass and radius must be finite and positive")
        if len(set(self.entity_ids)) != len(self.entity_ids):
            raise ValueError("Scene entity IDs must be unique")


@dataclass(frozen=True)
class ControlSpec:
    """Physical control accepted by an adapter, before policy normalization.

    Units encode coordinate ordering as well as physical units (newtons_xy).
    Backends must declare this explicitly; renderers leave control unset.
    """

    dimension: int
    units: str

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("Control dimension must be a positive integer")
        if not isinstance(self.units, str) or not self.units:
            raise ValueError("Control units must be a nonempty name")


@dataclass(frozen=True)
class Capabilities:
    """Advertised adapter features, checked before constructing engine resources."""

    scenes: frozenset[str] = frozenset({"point_reach"})
    device: str = "cpu"
    partial_reset: bool = True
    active_mask: bool = True
    channels: frozenset[str] = frozenset()
    transport: str = "host_snapshot"
    control: ControlSpec | None = None
    required_entities: frozenset[str] = frozenset()


@dataclass
class StateSnapshot:
    """Owned CPU arrays; position/velocity [N,D], time/version [N].

    Callers may retain a snapshot across physics steps. Consumers must not mutate
    snapshots passed to them by the runtime. Version increases on step and reset.
    """

    position: np.ndarray
    velocity: np.ndarray
    time: np.ndarray
    version: np.ndarray


@dataclass(frozen=True)
class TaskSpec:
    """Stable policy-facing dimensions and scalar action bounds for one task."""

    id: str
    instruction: str
    action_dim: int
    proprio_dim: int
    action_units: str
    action_low: float = -1.0
    action_high: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (self.id, self.instruction, self.action_units)
        ):
            raise ValueError(
                "Task identity, instruction and action units must be nonempty"
            )
        if any(
            type(v) is not int or v <= 0 for v in (self.action_dim, self.proprio_dim)
        ):
            raise ValueError("Task dimensions must be positive integers")
        if not (
            np.isfinite(self.action_low)
            and np.isfinite(self.action_high)
            and self.action_low < self.action_high
        ):
            raise ValueError("Task action bounds must be finite and increasing")


@dataclass
class SceneUpdate:
    """Borrowed synchronous snapshot and entity-ID to [N,D] position mapping."""

    state: StateSnapshot
    entities: dict[str, Array]


class Task(Protocol):
    """Own task state; never access a concrete engine or advance simulation time.

    One instance belongs to one environment. build allocates per-environment task
    state; reset modifies only selected IDs. observe/evaluate are read-only.
    """

    scene: SceneSpec
    spec: TaskSpec

    def build(self, num_envs: int) -> None:
        """Allocate task state once, before the first reset."""
        ...

    def reset(self, ids: Array, rngs: list[np.random.Generator]) -> Array:
        """Reset selected rows; return initial positions [len(ids), D]."""
        ...

    def observe(self, state: StateSnapshot) -> Observation:
        """Return task features, including float32 proprio [N, proprio_dim]."""
        ...

    def scene_update(self, state: StateSnapshot) -> SceneUpdate:
        """Map named visual entities to positions for a synchronous renderer."""
        ...

    def evaluate(
        self, state: StateSnapshot, actions: Array, previous: StateSnapshot
    ) -> tuple[Array, Array, Array]:
        """Return reward float32, terminated bool, success bool, all [N]."""
        ...


class Policy(Protocol):
    """Stateless adapter interface; stateful model caches are managed by callers."""

    def act(self, observation: Observation) -> Array:
        """Return finite action chunks [N,H,A] in the environment's units."""
        ...


@dataclass
class RenderBatch:
    """Rendered rows in request order; arrays remain valid after render returns."""

    images: dict[str, np.ndarray]
    time: np.ndarray
    version: np.ndarray
    calibration: dict = field(default_factory=dict)


@dataclass
class StepResult:
    """Post-action, pre-reset observations and per-environment transition flags."""

    observation: dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict[str, np.ndarray]


class PhysicsBackend(Protocol):
    """Own engine resources; support selected reset and frozen inactive rows.

    Lifecycle: build once, reset/control/step/snapshot, close. close must be
    idempotent and safe after a partially failed build. No task/reward logic.
    """

    capabilities: Capabilities

    def build(self, scene: SceneSpec, config: Config) -> None:
        """Allocate resources for the selected scene and environment count once."""
        ...

    def reset(self, ids: np.ndarray, position: np.ndarray) -> None:
        """Initialize selected positions, clearing their velocity, control and time."""
        ...

    def apply_control(self, actions: np.ndarray) -> None:
        """Copy control input [N,A]; the current planar scene accepts XY forces."""
        ...

    def step(self, dt: float, substeps: int, active: np.ndarray) -> None:
        """Advance active [N] rows by total dt; each substep lasts dt/substeps."""
        ...

    def snapshot(self) -> StateSnapshot:
        """Return owned arrays; later steps must not change earlier snapshots."""
        ...

    def close(self) -> None:
        """Release resources once; tolerate a partially completed build."""
        ...


class RenderBackend(Protocol):
    """Consume synchronous scene updates; never advance physics or task clocks.

    sync borrows its update until render returns. Asynchronous implementations
    require a different transport contract and are rejected by this runtime.
    """

    capabilities: Capabilities

    def build(self, scene: SceneSpec, config: Config) -> None:
        """Allocate the visual scene and camera resources once."""
        ...

    def sync(self, update: SceneUpdate) -> None:
        """Borrow a current scene update until the following render returns."""
        ...

    def render(self, ids: np.ndarray) -> RenderBatch:
        """Render selected IDs in request order with matching state time/version."""
        ...

    def close(self) -> None:
        """Release resources once; tolerate a partially completed build."""
        ...
