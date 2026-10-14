"""Versioned complete episodes and windows that never cross episode boundaries."""

import json
import platform
import uuid
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np

from .core import Array, Observation, StepResult
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


def _dataset_manifest(root: Path) -> dict:
    manifest = json.loads((root / "run.json").read_text())
    if manifest.get("kind") == "core-ppo":
        raise ValueError(
            "Expected an episode dataset, received a core PPO training run"
        )
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported run schema")
    return manifest


def _all_finite(value: Array) -> bool:
    """Avoid an episode-sized boolean temporary for large numeric fields."""
    if value.size <= 4 * 1024 * 1024:
        return bool(np.isfinite(value).all())
    with np.nditer(
        value,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        buffersize=max(1, 1024 * 1024 // value.dtype.itemsize),
    ) as chunks:
        return all(np.isfinite(chunk).all() for chunk in chunks)


def _array_equal(first: Array, second: Array) -> bool:
    """Compare numeric fields without flattening or copying strided inputs."""
    if first.shape != second.shape:
        return False
    if first.size <= 4 * 1024 * 1024:
        return np.array_equal(first, second)
    with np.nditer(
        [first, second],
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=[["readonly"], ["readonly"]],
        buffersize=max(
            1, 1024 * 1024 // max(first.dtype.itemsize, second.dtype.itemsize)
        ),
    ) as chunks:
        return all(np.array_equal(left, right) for left, right in chunks)


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
            or value.dtype.kind == "f"
            and not _all_finite(value)
        ):
            raise ValueError(f"Invalid numeric observation field: {name}")
    for name in ("env_id", "episode_id", "step_id"):
        value = observation[name]
        if value.shape != (count,) or value.dtype.kind not in "iu" or (value < 0).any():
            raise ValueError(f"{name} must be nonnegative integer[N]")
    if observation["time"].shape != (count,):
        raise ValueError("time must have shape [N]")


class EpisodeRecorder:
    """Bounded recorder: slow storage applies explicit backpressure.

    One episode buffers at most max_steps per environment, within a shared byte
    budget for retained array payloads. Python objects, caller-owned batches and
    compression/stacking scratch space are additional memory. Overflow is an error;
    write_workers=0 writes synchronously; positive values compress/write completed
    episodes in background threads. Queued payloads share the same byte budget;
    append waits for storage only when queued episodes can relieve that budget.
    Incomplete buffers are discarded on close and listed in the run manifest.
    Readers only accept shards with an adjacent completion metadata file.
    """

    def __init__(
        self,
        directory: str | Path,
        env: VectorEnv,
        *,
        max_buffer_bytes: int = 256 * 1024 * 1024,
        episode_format: str = "transitions",
        write_workers: int = 0,
    ) -> None:
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be a positive integer")
        if type(write_workers) is not int or write_workers < 0:
            raise ValueError("write_workers must be a nonnegative integer")
        if not isinstance(episode_format, str) or episode_format not in (
            "transitions",
            "compact",
        ):
            raise ValueError("episode_format must be transitions or compact")
        self._episode_format = episode_format
        self._max_buffer_bytes = max_buffer_bytes
        self.buffered_bytes = 0
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=False)
        self.pending = {}
        self.closed = False
        self.max_steps = env.config.max_steps
        self.num_envs = env.config.num_envs
        self.failure: BaseException | None = None
        self.close_error: BaseException | None = None
        self._writes: dict[tuple[int, int], tuple[Future, list[dict], int]] = {}
        self._executor: ThreadPoolExecutor | None = None
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
            "episode_format": episode_format,
            "write_workers": write_workers,
        }
        _json_atomic(self.root / "run.json", self.manifest)
        if write_workers:
            self._executor = ThreadPoolExecutor(
                max_workers=write_workers, thread_name_prefix="episode-writer"
            )

    @property
    def max_buffer_bytes(self) -> int:
        """Fixed payload budget for this run, also recorded in its manifest."""
        return self._max_buffer_bytes

    @property
    def episode_format(self) -> str:
        """Fixed storage layout; changing it mid-run would invalidate byte accounting."""
        return self._episode_format

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
        if self.episode_format == "compact":
            # Continuing obs_t is the already-owned next observation from t-1.
            continuing = sum(
                (int(i), int(observation["episode_id"][i])) in self.pending
                for i in active_ids
            )
            incoming_bytes -= continuing * sum(
                value.nbytes // count for value in observation.values()
            )
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
            if (
                key in self._writes
                or (self.root / f"env{key[0]:05d}-episode{key[1]:08d}.npz").exists()
            ):
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
            row = {
                f"obs/{k}": (
                    rows[-1][f"next/{k}"]
                    if rows and self.episode_format == "compact"
                    else v[i].copy()
                )
                for k, v in observation.items()
            }
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
        error = self._collect_writes()
        if error is not None:
            raise error
        while True:
            try:
                prepared = self._prepare(observation, result)
                break
            except BufferError:
                if not self._writes:
                    raise
                # Waiting can free queued payloads, never incomplete episodes.
                error = self._collect_writes(wait_for=next(iter(self._writes)))
                if error is not None:
                    raise error from None
        try:
            for key, row, complete in prepared:
                rows = self.pending.setdefault(key, [])
                self.buffered_bytes += self._row_bytes(row, continuing=bool(rows))
                rows.append(row)
                if complete:
                    self._finish(key)
        except BaseException as error:
            self.failure = error
            raise

    def _row_bytes(self, row: dict, *, continuing: bool) -> int:
        """Count only new owned payloads; compact rows share adjacent observations."""
        return sum(
            value.nbytes
            for name, value in row.items()
            if not (
                continuing
                and self.episode_format == "compact"
                and name.startswith("obs/")
            )
        )

    def _finish(self, key: tuple[int, int]) -> None:
        rows = self.pending[key]
        if self._executor is None:
            self._write_episode(key, rows)
            self._release_episode(key, rows)
        else:
            size = sum(
                self._row_bytes(row, continuing=index > 0)
                for index, row in enumerate(rows)
            )
            future = self._executor.submit(self._write_episode, key, rows)
            self._writes[key] = (future, rows, size)
            del self.pending[key]

    def _write_episode(self, key: tuple[int, int], rows: list[dict]) -> None:
        """Publish files only; worker threads never mutate recorder bookkeeping."""
        stem = f"env{key[0]:05d}-episode{key[1]:08d}"
        path = self.root / f"{stem}.npz"
        if path.exists():
            raise ValueError("Duplicate episode ID; do not reseed a recording run")
        temporary = self.root / f"{stem}.npz.tmp"
        if self.episode_format == "compact":
            arrays = {
                f"state/{name[4:]}": np.stack(
                    [row[name] for row in rows] + [rows[-1][f"next/{name[4:]}"]]
                )
                for name in rows[0]
                if name.startswith("obs/")
            }
            arrays.update(
                {
                    name: np.stack([row[name] for row in rows])
                    for name in ("action", "reward", "terminated", "truncated")
                }
            )
        else:
            arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        _json_atomic(
            self.root / f"{stem}.json",
            {
                "schema_version": 2 if self.episode_format == "compact" else 1,
                "complete": True,
                "steps": len(rows),
                "env_id": key[0],
                "episode_id": key[1],
            },
        )

    def _release_episode(self, key: tuple[int, int], rows: list[dict]) -> None:
        self.buffered_bytes -= sum(
            self._row_bytes(row, continuing=index > 0) for index, row in enumerate(rows)
        )
        del self.pending[key]
        self.manifest["completed_episodes"] += 1

    def _collect_writes(
        self, *, wait_for: tuple[int, int] | None = None
    ) -> BaseException | None:
        """Reap finished work on the caller thread, retaining failed rows for close."""
        error = None
        for key, (future, rows, size) in list(self._writes.items()):
            if key != wait_for and not future.done():
                continue
            try:
                future.result()
            except BaseException as caught:
                if not future.done():
                    # A signal interrupted the caller's wait, not the write.
                    raise
                if future.cancelled():
                    if not isinstance(caught, CancelledError):
                        raise
                elif future.exception() is not caught:
                    # The write may finish just as the caller receives a signal.
                    # Do not mistake that signal for a worker/storage failure.
                    raise
                self.failure = self.failure or caught
                error = error or caught
                self.pending[key] = rows
            else:
                self.buffered_bytes -= size
                self.manifest["completed_episodes"] += 1
            del self._writes[key]
        return error

    def close(self) -> None:
        """Wait for queued writes, finalize the manifest and discard incomplete rows."""
        if self.closed:
            return
        error = None
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            error = self._collect_writes()
        self.manifest.update(
            status="failed" if self.failure is not None else "closed",
            incomplete_episodes=[
                {"env_id": k[0], "episode_id": k[1], "steps": len(v)}
                for k, v in self.pending.items()
            ],
        )
        try:
            _json_atomic(self.root / "run.json", self.manifest)
        except BaseException as caught:
            self.failure = self.failure or caught
            if error is not None:
                self.close_error = caught
                raise error from caught
            raise
        self.pending.clear()
        self.buffered_bytes = 0
        self.closed = True
        if error is not None:
            raise error

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
            except BaseException as error:
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
    yield from _read_episodes(directory, options=options, owned=True)


def _read_episodes(
    directory: str | Path,
    *,
    options: EpisodeReadOptions | None,
    owned: bool,
) -> Iterator[Observation]:
    """Internal compact views stay private; window results copy their slices."""
    options = options or EpisodeReadOptions()
    root = Path(directory)
    _dataset_manifest(root)
    completed_index = 0
    for path in sorted(root.glob("*.npz")):
        marker = path.with_suffix(".json")
        if not marker.exists():
            continue
        metadata = json.loads(marker.read_text())
        schema = metadata.get("schema_version")
        if not metadata.get("complete"):
            continue
        if type(schema) is not int or schema not in (1, 2):
            raise ValueError(f"Unsupported episode schema: {path}")
        selected = completed_index % options.num_shards == options.shard_index
        completed_index += 1
        if not selected:
            continue
        yield _load_episode(path, metadata, options, owned=owned)


def _load_episode(
    path: Path, metadata: dict, options: EpisodeReadOptions, *, owned: bool
) -> Observation:
    """Decode and validate in a short-lived scope so scratch arrays are released."""
    schema = metadata["schema_version"]
    n = metadata["steps"]
    if type(n) is not int or n < 1:
        raise ValueError(f"Corrupt episode lengths: {path}")
    with np.load(path, allow_pickle=False) as archive:
        keys = archive.files
        prefix = "state/" if schema == 2 else "obs/"
        obs_keys = {k[len(prefix) :] for k in keys if k.startswith(prefix)}
        if schema == 2:
            if any(
                not k.startswith("state/")
                and k not in {"action", "reward", "terminated", "truncated"}
                for k in keys
            ):
                raise ValueError(f"Invalid compact episode fields: {path}")
        elif options.observation_keys is not None:
            next_keys = {k[5:] for k in keys if k.startswith("next/")}
            if obs_keys != next_keys:
                raise ValueError(f"Observation schema changed: {path}")
        if options.observation_keys is not None:
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
                    or k.startswith(prefix)
                    and k[len(prefix) :] in requested
                    or k.startswith("next/")
                    and k[5:] in requested
                )
            ]
        data = {k: archive[k] for k in keys}
    if schema == 2:
        states = {
            name[6:]: value for name, value in data.items() if name.startswith("state/")
        }
        if any(value.ndim < 1 or len(value) != n + 1 for value in states.values()):
            raise ValueError(f"Corrupt compact observation lengths: {path}")
        data = {
            name: value for name, value in data.items() if not name.startswith("state/")
        }
        # Validate compact states before allocating independent public arrays.
        for name, values in states.items():
            data[f"obs/{name}"] = values[:-1]
            data[f"next/{name}"] = values[1:]
    if any(v.ndim < 1 or len(v) != n for v in data.values()):
        raise ValueError(f"Corrupt episode lengths: {path}")
    required = {"action", "reward", "terminated", "truncated"}
    if not required <= data.keys():
        raise ValueError(f"Missing transition fields: {path}")
    observation = {k[4:]: v for k, v in data.items() if k.startswith("obs/")}
    following = {k[5:]: v for k, v in data.items() if k.startswith("next/")}
    if schema == 2:
        _validate_observation(states, n + 1)
    else:
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
    for name, value in observation.items() if schema == 1 else ():
        if (
            value.shape != following[name].shape
            or value.dtype != following[name].dtype
            or not _array_equal(value[1:], following[name][:-1])
        ):
            raise ValueError(f"Observation continuity broken for {name}: {path}")
    if schema == 2 and owned:
        for name in states:
            data[f"obs/{name}"] = observation[name].copy()
            data[f"next/{name}"] = following[name].copy()
    return data


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
    for observation, action, instruction in _episode_windows(
        directory, history, horizon, options=options
    ):
        for index in range(len(action)):
            yield {
                "observation": {
                    key: value[index].copy() for key, value in observation.items()
                },
                "action": action[index].copy(),
                "instruction": instruction,
            }
        del observation, action


def _episode_windows(
    directory: str | Path,
    history: int,
    horizon: int,
    *,
    options: EpisodeReadOptions | None,
) -> Iterator[tuple[Observation, Array, str]]:
    """Private read-only window views; consumers must copy before advancing."""
    if any(type(value) is not int or value <= 0 for value in (history, horizon)):
        raise ValueError("history and horizon must be positive integers")
    manifest = _dataset_manifest(Path(directory))
    instruction = manifest["env_spec"]["instruction"]
    for episode in _read_episodes(directory, options=options, owned=False):
        count = max(0, len(episode["action"]) - history - horizon + 2)
        if count:
            observation = {
                key[4:]: np.moveaxis(
                    np.lib.stride_tricks.sliding_window_view(value, history, axis=0),
                    -1,
                    1,
                )[:count]
                for key, value in episode.items()
                if key.startswith("obs/")
            }
            action = np.moveaxis(
                np.lib.stride_tricks.sliding_window_view(
                    episode["action"], horizon, axis=0
                ),
                -1,
                1,
            )[history - 1 : history - 1 + count]
            # Following/reward/terminal arrays were validated, but no longer
            # need to stay resident while consumers copy observation windows.
            del episode
            yield observation, action, instruction
            del observation, action
            continue
        del episode
