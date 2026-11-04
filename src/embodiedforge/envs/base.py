"""Small CPU environment contract for reusable collection and evaluation."""

from typing import Protocol

from embodiedforge.core import Array, Observation, StepResult


class BatchEnvironment(Protocol):
    """Owned observations, explicit reset, and post-action/pre-reset transitions.

    spec supplies num_envs, action_shape, proprio_shape, control_dt, action_units
    and action_range (None for an unbounded policy-space action). Version 1 also
    declares schema_version and done_handling; unversioned layouts are legacy.
    reset returns
    only the selected rows in caller order; observe returns all rows without
    advancing time or sampling new task noise. Consumers must reset done rows
    before their next step: only some implementations can freeze them.

    This protocol borrows resources. The concrete environment's owner controls
    initialization, reseeding and shutdown; no universal asset or close API is
    implied. The core VectorEnv implements this contract directly.
    """

    @property
    def spec(self) -> dict: ...

    def reset(self, env_ids: Array | list[int] | None = None) -> Observation: ...

    def observe(self) -> Observation: ...

    def step(self, actions: Array) -> StepResult: ...


class TransitionWriter(Protocol):
    """Borrow a transition synchronously; copy any arrays retained after append."""

    def append(self, observation: Observation, result: StepResult) -> None: ...
