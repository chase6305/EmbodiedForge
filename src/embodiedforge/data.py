"""Versioned complete episodes and windows that never cross episode boundaries."""

import json
import platform
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np

from .core import Observation, StepResult
from .env import VectorEnv

_REQUIRED_OBSERVATIONS = frozenset(
    {"env_id", "episode_id", "step_id", "time", "proprio"}
)


@dataclass(frozen=True)
class EpisodeReadOptions:
    """Select additional observations and one deterministic episode partition.

    None loads every field; an empty tuple loads only required state/identity
    fields. Required observations and transition fields are always validated.
    Unselected array payloads are neither decompressed nor validated. Partition
    membership is stable only for an unchanged directory of completed episodes.
    """

    observation_keys: tuple[str, ...] | None = None
    shard_index: int = 0
    num_shards: int = 1

    def __post_init__(self) -> None:
        if (
            type(self.num_shards) is not int
            or self.num_shards <= 0
            or type(self.shard_index) is not int
            or not 0 <= self.shard_index < self.num_shards
        ):
            raise ValueError("Require num_shards > 0 and 0 <= shard_index < num_shards")
        keys = self.observation_keys
        if keys is not None:
            if not isinstance(keys, (tuple, list)) or any(
                not isinstance(key, str) or not key or "/" in key for key in keys
            ):
                raise ValueError("observation_keys must be a sequence of field names")
            if len(set(keys)) != len(keys):
                raise ValueError("observation_keys must be unique")
            object.__setattr__(self, "observation_keys", tuple(keys))


def _json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _validate_observation(observation: Observation, count: int) -> None:
    """Validate the numeric batch and identifiers common to writer and reader."""
    if not _REQUIRED_OBSERVATIONS <= observation.keys():
        raise ValueError("Observation is missing required episode fields")
    for name, value in observation.items():
        if (
            not isinstance(value, np.ndarray)
            or value.ndim < 1
            or len(value) != count
            or value.dtype.kind not in "biuf"
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid numeric observation field: {name}")
    for name in ("env_id", "episode_id", "step_id"):
        value = observation[name]
        if value.shape != (count,) or value.dtype.kind not in "iu" or (value < 0).any():
            raise ValueError(f"{name} must be nonnegative integer[N]")
    if observation["time"].shape != (count,):
        raise ValueError("time must have shape [N]")


class EpisodeRecorder:
    """Synchronous, bounded recorder: slow storage applies explicit backpressure.

    One episode buffers at most max_steps per environment, within a shared byte
    budget for retained array payloads. Python objects, caller-owned batches and
    compression/stacking scratch space are additional memory. Overflow is an error;
    incomplete buffers are discarded on close and listed in the run manifest.
    Readers only accept shards with an adjacent completion metadata file.
    """

    def __init__(
        self,
        directory: str | Path,
        env: VectorEnv,
        *,
        max_buffer_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be a positive integer")
        self._max_buffer_bytes = max_buffer_bytes
        self.buffered_bytes = 0
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.pending = {}
        self.closed = False
        self.max_steps = env.config.max_steps
        self.num_envs = env.config.num_envs
        self.failure: BaseException | None = None
        self.close_error: Exception | None = None
        packages = {}
        sources = {}
        for name in (
            "numpy",
            "mujoco",
            "newton",
            "warp-lang",
            "torch",
            "embodiedforge",
        ):
            try:
                packages[name] = version(name)
                source = distribution(name).read_text("direct_url.json")
                if source is not None:
                    sources[name] = json.loads(source)
            except PackageNotFoundError:
                pass
        self.manifest = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "config": env.config.to_dict(),
            "env_spec": env.spec,
            "calibration": env.calibration,
            "python": platform.python_version(),
            "packages": packages,
            "dependency_sources": sources,
            "status": "recording",
            "completed_episodes": 0,
            "transport": "host_snapshot",
            "scene": f"{env.task.scene.kind}:v1",
            "scene_spec": asdict(env.task.scene),
            "task": env.task.spec.id,
            "seed": env.seed,
            "max_buffer_bytes": max_buffer_bytes,
        }
        _json_atomic(self.root / "run.json", self.manifest)

    @property
    def max_buffer_bytes(self) -> int:
        """Fixed payload budget for this run, also recorded in its manifest."""
        return self._max_buffer_bytes

    def _prepare(self, observation: Observation, result: StepResult) -> list[tuple]:
        """Validate and copy an entire batch without changing memory or disk state."""
        count = self.num_envs
        _validate_observation(observation, count)
        _validate_observation(result.observation, count)
        if observation.keys() != result.observation.keys():
            raise ValueError("Observation schema changed within a transition")
        for name, value in (
            ("active", result.info.get("active")),
            ("terminated", result.terminated),
            ("truncated", result.truncated),
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (count,)
                or value.dtype != bool
            ):
                raise ValueError(f"{name} must be bool[N]")
        action = result.info.get("executed_action")
        if (
            not isinstance(action, np.ndarray)
            or action.shape != (count, self.manifest["env_spec"]["action_shape"][0])
            or not np.isfinite(action).all()
        ):
            raise ValueError("Recorded action must be finite [N,A]")
        if (
            not isinstance(result.reward, np.ndarray)
            or result.reward.shape != (count,)
            or not np.isfinite(result.reward).all()
        ):
            raise ValueError("Recorded reward must be finite [N]")
        if (result.terminated & result.truncated).any():
            raise ValueError("A transition cannot be both terminated and truncated")
        expected_ids = np.arange(count)
        if not np.array_equal(
            observation["env_id"], expected_ids
        ) or not np.array_equal(result.observation["env_id"], expected_ids):
            raise ValueError("Recorder requires canonical environment row order")
        active_ids = np.flatnonzero(result.info["active"])
        payload = (
            *observation.values(),
            *result.observation.values(),
            action,
            result.reward,
            result.terminated,
            result.truncated,
        )
        incoming_bytes = len(active_ids) * sum(v.nbytes // count for v in payload)
        if self.buffered_bytes + incoming_bytes > self.max_buffer_bytes:
            raise BufferError(
                f"Recorder buffer budget exceeded: retained={self.buffered_bytes}, "
                f"incoming={incoming_bytes}, limit={self.max_buffer_bytes} bytes; "
                "reduce image size/environment count/episode length or explicitly "
                "increase max_buffer_bytes"
            )
        # One pending episode per environment; compute the lookup once per batch.
        pending_episodes = {key[0]: key[1] for key in self.pending}
        prepared = []
        for i in active_ids:
            episode = int(observation["episode_id"][i])
            key = (int(i), episode)
            step = int(observation["step_id"][i])
            if key[0] in pending_episodes and pending_episodes[key[0]] != episode:
                raise ValueError(
                    "An environment was reset before its recorded episode ended"
                )
            rows = self.pending.get(key, [])
            if step != len(rows):
                raise ValueError(
                    "Recording must start at reset and contain consecutive transitions"
                )
            if len(rows) >= self.max_steps:
                raise BufferError("Episode exceeds configured recorder capacity")
            if (self.root / f"env{key[0]:05d}-episode{key[1]:08d}.npz").exists():
                raise ValueError("Duplicate episode ID; do not reseed a recording run")
            if (
                result.observation["episode_id"][i] != episode
                or result.observation["step_id"][i] != step + 1
                or result.observation["time"][i] <= observation["time"][i]
            ):
                raise ValueError(
                    "next observation must precede reset and advance exactly one step"
                )
            for name, values in observation.items():
                next_values = result.observation[name]
                if (
                    values.shape != next_values.shape
                    or values.dtype != next_values.dtype
                ):
                    raise ValueError(f"Observation layout changed for {name}")
                if rows and (
                    f"next/{name}" not in rows[-1]
                    or not np.array_equal(values[i], rows[-1][f"next/{name}"])
                ):
                    raise ValueError(f"Observation continuity broken for {name}")
            row = {f"obs/{k}": v[i].copy() for k, v in observation.items()}
            row.update(
                {f"next/{k}": v[i].copy() for k, v in result.observation.items()}
            )
            row.update(
                action=action[i].copy(),
                reward=result.reward[i].copy(),
                terminated=result.terminated[i].copy(),
                truncated=result.truncated[i].copy(),
            )
            prepared.append(
                (key, row, bool(result.terminated[i] or result.truncated[i]))
            )
        return prepared

    def append(self, observation: Observation, result: StepResult) -> None:
        """Validate all rows before committing; storage failures disable further appends.

        Validation errors are retryable and make no changes. Disk publication is
        atomic per episode, not across multiple episodes in a batch.
        """
        if self.failure is not None:
            raise RuntimeError("Recorder failed; create a new run") from self.failure
        if self.closed:
            raise RuntimeError("Recorder is closed")
        prepared = self._prepare(observation, result)
        try:
            for key, row, complete in prepared:
                self.pending.setdefault(key, []).append(row)
                self.buffered_bytes += sum(value.nbytes for value in row.values())
                if complete:
                    self._finish(key)
        except BaseException as error:
            self.failure = error
            raise

    def _finish(self, key: tuple[int, int]) -> None:
        rows = self.pending[key]
        stem = f"env{key[0]:05d}-episode{key[1]:08d}"
        path = self.root / f"{stem}.npz"
        if path.exists():
            raise ValueError("Duplicate episode ID; do not reseed a recording run")
        temporary = self.root / f"{stem}.npz.tmp"
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, **{k: np.stack([row[k] for row in rows]) for k in rows[0]}
            )
        temporary.replace(path)
        _json_atomic(
            self.root / f"{stem}.json",
            {
                "schema_version": 1,
                "complete": True,
                "steps": len(rows),
                "env_id": key[0],
                "episode_id": key[1],
            },
        )
        self.buffered_bytes -= sum(
            value.nbytes for row in rows for value in row.values()
        )
        del self.pending[key]
        self.manifest["completed_episodes"] += 1

    def close(self) -> None:
        """Finalize the run manifest; list incomplete buffers without publishing them."""
        if self.closed:
            return
        self.manifest.update(
            status="failed" if self.failure is not None else "closed",
            incomplete_episodes=[
                {"env_id": k[0], "episode_id": k[1], "steps": len(v)}
                for k, v in self.pending.items()
            ],
        )
        _json_atomic(self.root / "run.json", self.manifest)
        self.pending.clear()
        self.buffered_bytes = 0
        self.closed = True

    def __enter__(self) -> "EpisodeRecorder":
        if self.closed or self.failure is not None:
            raise RuntimeError("Recorder is not available")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if exc_value is None:
            self.close()
        else:
            self.failure = self.failure or exc_value
            try:
                self.close()
            except Exception as error:
                self.close_error = error


def read_episodes(
    directory: str | Path,
    *,
    options: EpisodeReadOptions | None = None,
) -> Iterator[Observation]:
    """Yield owned, validated arrays; defaults preserve full episode validation.

    Partition completed episodes before opening archives. Projection checks the
    full observation key schema but validates only selected payloads plus the
    mandatory transition/state fields. Use the default for a full dataset audit.
    """
    options = options or EpisodeReadOptions()
    root = Path(directory)
    manifest = json.loads((root / "run.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported run schema")
    completed_index = 0
    for path in sorted(root.glob("*.npz")):
        marker = path.with_suffix(".json")
        if not marker.exists():
            continue
        metadata = json.loads(marker.read_text())
        if metadata.get("schema_version") != 1 or not metadata.get("complete"):
            continue
        selected = completed_index % options.num_shards == options.shard_index
        completed_index += 1
        if not selected:
            continue
        with np.load(path, allow_pickle=False) as archive:
            keys = archive.files
            if options.observation_keys is not None:
                obs_keys = {k[4:] for k in keys if k.startswith("obs/")}
                next_keys = {k[5:] for k in keys if k.startswith("next/")}
                if obs_keys != next_keys:
                    raise ValueError(f"Observation schema changed: {path}")
                requested = _REQUIRED_OBSERVATIONS | set(options.observation_keys)
                missing = requested - obs_keys
                if missing:
                    raise ValueError(
                        f"Missing requested observations {sorted(missing)}: {path}"
                    )
                keys = [
                    k
                    for k in keys
                    if (
                        k in {"action", "reward", "terminated", "truncated"}
                        or k.startswith("obs/")
                        and k[4:] in requested
                        or k.startswith("next/")
                        and k[5:] in requested
                    )
                ]
            data = {k: archive[k] for k in keys}
        n = metadata["steps"]
        if (
            type(n) is not int
            or n < 1
            or any(v.ndim < 1 or len(v) != n for v in data.values())
        ):
            raise ValueError(f"Corrupt episode lengths: {path}")
        required = {"action", "reward", "terminated", "truncated"}
        if not required <= data.keys():
            raise ValueError(f"Missing transition fields: {path}")
        observation = {k[4:]: v for k, v in data.items() if k.startswith("obs/")}
        following = {k[5:]: v for k, v in data.items() if k.startswith("next/")}
        _validate_observation(observation, n)
        _validate_observation(following, n)
        if observation.keys() != following.keys():
            raise ValueError(f"Observation schema changed: {path}")
        for name in ("terminated", "truncated"):
            if data[name].shape != (n,) or data[name].dtype != bool:
                raise ValueError(f"Invalid {name}: {path}")
        if (data["terminated"] & data["truncated"]).any():
            raise ValueError(f"Conflicting terminal flags: {path}")
        if (
            data["action"].ndim != 2
            or not np.isfinite(data["action"]).all()
            or data["reward"].shape != (n,)
            or not np.isfinite(data["reward"]).all()
        ):
            raise ValueError(f"Invalid action/reward: {path}")
        done = data["terminated"] | data["truncated"]
        if not done[-1] or done[:-1].any():
            raise ValueError(f"Invalid episode boundary: {path}")
        if not np.array_equal(data["obs/step_id"], np.arange(n)):
            raise ValueError(f"Nonconsecutive episode: {path}")
        if not np.array_equal(following["step_id"], np.arange(1, n + 1)):
            raise ValueError(f"Invalid next step IDs: {path}")
        for name in ("env_id", "episode_id"):
            if not (
                (observation[name] == metadata[name]).all()
                and (following[name] == metadata[name]).all()
            ):
                raise ValueError(f"Episode identity mismatch: {path}")
        if not (following["time"] > observation["time"]).all():
            raise ValueError(f"Time must advance: {path}")
        for name, value in observation.items():
            if (
                value.shape != following[name].shape
                or value.dtype != following[name].dtype
                or not np.array_equal(value[1:], following[name][:-1])
            ):
                raise ValueError(f"Observation continuity broken for {name}: {path}")
        yield data


def action_windows(
    directory: str | Path,
    history: int = 1,
    horizon: int = 1,
    *,
    options: EpisodeReadOptions | None = None,
) -> Iterator[dict]:
    """Yield full (unpadded) history/action windows, with language and images.

    Action at t aligns with the last observation in its history. Episodes too
    short for a full window are skipped; no windows span reset boundaries.
    """
    if any(type(value) is not int or value <= 0 for value in (history, horizon)):
        raise ValueError("history and horizon must be positive integers")
    manifest = json.loads((Path(directory) / "run.json").read_text())
    instruction = manifest["env_spec"]["instruction"]
    for episode in read_episodes(directory, options=options):
        for t in range(history - 1, len(episode["action"]) - horizon + 1):
            yield {
                "observation": {
                    k[4:]: v[t - history + 1 : t + 1].copy()
                    for k, v in episode.items()
                    if k.startswith("obs/")
                },
                "action": episode["action"][t : t + horizon].copy(),
                "instruction": instruction,
            }
