"""Action chunks are consumed outside the environment."""

import numpy as np

from .core import Array, Observation, Policy


class ActionExecutor:
    """Consume action chunks per environment, replanning only exhausted rows.

    Episode/step metadata automatically invalidates stale chunks. Without it,
    callers must reset selected rows explicitly. Neither path clears the policy's
    internal history/KV cache, which remains the caller's responsibility.
    """

    def __init__(
        self,
        num_envs: int,
        action_dim: int,
        horizon: int,
        *,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (num_envs, action_dim, horizon)
        ):
            raise ValueError("executor dimensions must be positive")
        self.chunks = np.zeros((num_envs, horizon, action_dim), dtype=np.float32)
        self._env_ids = np.arange(num_envs)
        self.cursor = np.zeros(num_envs, dtype=np.int64)
        self.length = np.zeros(num_envs, dtype=np.int64)
        if not (
            np.isfinite(action_low)
            and np.isfinite(action_high)
            and action_low < action_high
        ):
            raise ValueError("Invalid action bounds")
        self.action_low, self.action_high = action_low, action_high
        self._episode = np.full(num_envs, -1, dtype=np.int64)
        self._step = np.full(num_envs, -1, dtype=np.int64)

    def reset(self, ids: Array | list[int]) -> None:
        """Discard buffered actions for selected environments before the next act."""
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError("reset IDs must be a one-dimensional integer sequence")
        if ids.size == 0:
            return
        if (
            ids.dtype.kind not in "iu"
            or (ids < 0).any()
            or (ids >= len(self.cursor)).any()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("reset IDs must be unique integers in range")
        self._reset_rows(ids)

    def _reset_rows(self, ids: Array) -> None:
        """Clear validated IDs or the unique in-range rows found internally."""
        self.chunks[ids] = 0
        self.cursor[ids] = self.length[ids] = 0
        self._episode[ids] = self._step[ids] = -1

    def _changed_rows(self, observation: Observation) -> Array:
        """Detect reset, repeated observation or skipped control steps before acting."""
        count = len(self.cursor)
        if "env_id" in observation and not np.array_equal(
            observation["env_id"], self._env_ids
        ):
            raise ValueError("Executor requires canonical environment row order")
        tracked = {"episode_id", "step_id"} & observation.keys()
        if not tracked:
            return np.empty(0, dtype=np.int64)
        if len(tracked) != 2:
            raise ValueError("Episode tracking requires both episode_id and step_id")
        for name in tracked:
            value = observation[name]
            if (
                value.shape != (count,)
                or value.dtype.kind not in "iu"
                or (value < 0).any()
            ):
                raise ValueError(f"{name} must be nonnegative integer[N]")
        episode = observation["episode_id"]
        step = observation["step_id"]
        # Tracking buffers store int64, including wrapped uint64 bit patterns.
        # Match that signed encoding so large unsigned IDs compare correctly;
        # preserve byte order and borrow the input without copying its payload.
        if episode.dtype.kind == "u" and episode.dtype.itemsize == 8:
            episode = episode.view(episode.dtype.str.replace("u", "i"))
        if step.dtype.kind == "u" and step.dtype.itemsize == 8:
            step = step.view(step.dtype.str.replace("u", "i"))
        return np.flatnonzero((episode != self._episode) | (step != self._step + 1))

    def act(self, observation: Observation, policy: Policy) -> Array:
        """policy.act receives only rows requiring replanning, returns [B,H,A]."""
        if not observation or any(
            not isinstance(v, np.ndarray) or v.ndim < 1 or len(v) != len(self.cursor)
            for v in observation.values()
        ):
            raise ValueError("Observation batch must match executor environment count")
        changed = self._changed_rows(observation)
        if changed.size:
            self._reset_rows(changed)
        ids = np.flatnonzero(self.cursor >= self.length)
        if len(ids):
            chunks = np.asarray(
                policy.act({k: v[ids] for k, v in observation.items()}),
                dtype=np.float32,
            )
            if (
                chunks.ndim != 3
                or chunks.shape[0] != len(ids)
                or chunks.shape[2] != self.chunks.shape[2]
                or not 1 <= chunks.shape[1] <= self.chunks.shape[1]
                or not np.isfinite(chunks).all()
                or (chunks < self.action_low).any()
                or (chunks > self.action_high).any()
            ):
                raise ValueError(
                    "policy must return finite bounded [B,H,A] action chunks"
                )
            self.chunks[ids, : chunks.shape[1]] = chunks
            self.cursor[ids] = 0
            self.length[ids] = chunks.shape[1]
        # Integer indexing already allocates an independent action batch.
        action = self.chunks[self._env_ids, self.cursor]
        self.cursor += 1
        if "episode_id" in observation:
            self._episode[:] = observation["episode_id"]
            self._step[:] = observation["step_id"]
            step = observation["step_id"]
            if step.dtype.kind == "u" and step.dtype.itemsize == 8:
                # No valid integer step can follow uint64's maximum. Exhaust
                # these rows so a wrap to zero cannot reuse a stale chunk.
                self.length[step == np.iinfo(np.uint64).max] = 0
        return action
