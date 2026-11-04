"""Versioned complete episodes and windows that never cross episode boundaries."""

import fnmatch
import json
import logging
import os
import platform
import uuid
import zipfile
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

import numpy as np

from .core import Array, Observation, StepResult
from .envs.base import BatchEnvironment
from .envs.specs import validate_environment_spec as _validate_environment_spec
from .runtime.artifacts import write_json as _json_atomic
from .runtime.hashing import file_sha256, stream_sha256
from .runtime.records import run_identity, validate_run_identity

_REQUIRED_OBSERVATIONS = frozenset(
    {"env_id", "episode_id", "step_id", "time", "proprio"}
)
# NumPy 1.x can override ndarray equality globally; scalar equality bypasses it.
_ALLOW_SCALAR_EQUALITY = not hasattr(np, "set_numeric_ops")
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RecordingMetadata:
    """Explicit provenance for recording a non-core CPU environment.

    config and scene_spec describe the caller's environment; the recorder does
    not infer assets, randomization or reconstruction settings from array shapes.
    max_steps bounds one buffered episode and must cover the task's episode limit.
    """

    family: str
    max_steps: int
    seed: int
    instruction: str
    scene: str
    scene_spec: dict
    config: dict
    calibration: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.max_steps) is not int or self.max_steps <= 0:
            raise ValueError("Recording max_steps must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("Recording seed must be a nonnegative integer")
        for name in ("family", "instruction", "scene"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Recording {name} must be a nonempty string")
        for name in ("scene_spec", "config", "calibration", "source"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"Recording {name} must be a JSON object")
        json.dumps(asdict(self), allow_nan=False)


def _recording_description(env: BatchEnvironment, metadata: RecordingMetadata | None):
    spec = deepcopy(env.spec)
    _validate_environment_spec(spec)
    if metadata is None:
        try:
            metadata = RecordingMetadata(
                family="core",
                max_steps=env.config.max_steps,
                seed=env.seed,
                instruction=env.task.spec.instruction,
                scene=f"{env.task.scene.kind}:v1",
                scene_spec=asdict(env.task.scene),
                config=env.config.to_dict(),
                calibration=env.calibration,
            )
        except AttributeError as error:
            raise ValueError(
                "Non-core recording requires explicit RecordingMetadata"
            ) from error
    elif not isinstance(metadata, RecordingMetadata):
        raise ValueError("metadata must be RecordingMetadata")
    metadata = RecordingMetadata(**asdict(metadata))
    spec["instruction"] = metadata.instruction
    json.dumps(spec, allow_nan=False)
    return spec, metadata


@dataclass(frozen=True)
class EpisodeReadOptions:
    """Select additional observations and one deterministic episode partition.

    None loads every field; an empty tuple loads only required state/identity
    fields. Required observations and transition fields are always validated.
    Unselected array payloads are not decompressed. verify_archive additionally
    checks the complete encoded file against its recorded SHA256; it requires a
    dataset written with archive verification enabled. Partition membership is
    stable only for an unchanged directory of completed episodes.
    """

    observation_keys: tuple[str, ...] | None = None
    shard_index: int = 0
    num_shards: int = 1
    verify_archive: bool = False

    def __post_init__(self) -> None:
        if type(self.verify_archive) is not bool:
            raise ValueError("verify_archive must be a boolean")
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


def _dataset_manifest(root: Path) -> dict:
    manifest = json.loads((root / "run.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Dataset manifest must be a JSON object")
    if manifest.get("kind") == "core-ppo":
        raise ValueError(
            "Expected an episode dataset, received a core PPO training run"
        )
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
    ):
        raise ValueError("Unsupported run schema")
    if manifest.keys() == {"schema_version"}:
        # Original standalone episode archives supplied no run-level layout.
        # Keep their low-level reader compatible; windows still need language.
        return manifest
    if not {"config", "env_spec", "completed_episodes", "run_id"} <= manifest.keys():
        raise ValueError("Expected an episode dataset manifest")
    if (
        not isinstance(manifest["config"], dict)
        or not isinstance(manifest["run_id"], str)
        or not manifest["run_id"]
        or type(manifest["completed_episodes"]) is not int
        or manifest["completed_episodes"] < 0
    ):
        raise ValueError("Invalid episode dataset metadata")
    if "run_contract" in manifest:
        identity = validate_run_identity(manifest["run_contract"])
        if (
            identity["workflow"] not in ("core-data", "episode-data")
            or identity["operation"] != "record"
        ):
            raise ValueError("Expected an episode recording run identity")
    _validate_environment_spec(manifest["env_spec"])
    instruction = manifest["env_spec"].get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Dataset instruction must be a nonempty string")
    return manifest


def _all_finite(value: Array) -> bool:
    """Avoid an episode-sized boolean temporary for large numeric fields."""
    if type(value) is not np.ndarray or value.size <= 4 * 1024 * 1024:
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
    if (
        _ALLOW_SCALAR_EQUALITY
        and type(first) is type(second) is first.dtype.type
        and first.dtype.kind in "biuf"
    ):
        # Identity/time fields become NumPy scalars when selecting one row.
        # Keep array and scalar subclasses on NumPy's comparison path.
        return bool(first == second)
    if (
        type(first) is not np.ndarray
        or type(second) is not np.ndarray
        or first.size <= 4 * 1024 * 1024
    ):
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
    verify_archive adds a bounded SHA256 pass and ZIP CRC read-back before
    publication, recording the hash in each completion marker. It costs extra
    I/O/decompression and is disabled by default.
    """

    def __init__(
        self,
        directory: str | Path,
        env: BatchEnvironment,
        *,
        metadata: RecordingMetadata | None = None,
        max_buffer_bytes: int = 256 * 1024 * 1024,
        episode_format: str = "transitions",
        write_workers: int = 0,
        verify_archive: bool = False,
    ) -> None:
        if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be a positive integer")
        if type(write_workers) is not int or write_workers < 0:
            raise ValueError("write_workers must be a nonnegative integer")
        if type(verify_archive) is not bool:
            raise ValueError("verify_archive must be a boolean")
        if not isinstance(episode_format, str) or episode_format not in (
            "transitions",
            "compact",
        ):
            raise ValueError("episode_format must be transitions or compact")
        spec, metadata = _recording_description(env, metadata)
        self._episode_format = episode_format
        self._verify_archive = verify_archive
        self._max_buffer_bytes = max_buffer_bytes
        self.buffered_bytes = 0
        self.root = Path(directory)
        self.pending = {}
        self.closed = False
        self._closing = False
        self.max_steps = metadata.max_steps
        self.num_envs = spec["num_envs"]
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
            "run_contract": run_identity(
                "core-data" if metadata.family == "core" else "episode-data",
                "record",
                spec["task"],
            ),
            "environment_family": metadata.family,
            "run_id": uuid.uuid4().hex,
            "config": metadata.config,
            "env_spec": spec,
            "calibration": metadata.calibration,
            "python": platform.python_version(),
            "packages": packages,
            "dependency_sources": sources,
            "status": "recording",
            "completed_episodes": 0,
            "transport": "host_snapshot",
            "scene": metadata.scene,
            "scene_spec": metadata.scene_spec,
            "task": spec["task"],
            "seed": metadata.seed,
            "max_buffer_bytes": max_buffer_bytes,
            "episode_format": episode_format,
            "write_workers": write_workers,
            "verify_archive": verify_archive,
        }
        if metadata.source:
            self.manifest["source"] = metadata.source
        # Resolve metadata before claiming a new directory. Once claimed, the
        # constructor owns failure finalization even before __enter__ can run.
        self.root.mkdir(parents=True, exist_ok=False)
        try:
            _json_atomic(self.root / "run.json", self.manifest)
            if write_workers:
                self._executor = ThreadPoolExecutor(
                    max_workers=write_workers, thread_name_prefix="episode-writer"
                )
        except BaseException as error:
            self.failure = error
            try:
                self.close()
            except BaseException:
                LOGGER.exception(
                    "Could not finalize recorder startup %s; preserving its original error",
                    self.root,
                )
            raise

    @property
    def max_buffer_bytes(self) -> int:
        """Fixed payload budget for this run, also recorded in its manifest."""
        return self._max_buffer_bytes

    @property
    def episode_format(self) -> str:
        """Fixed storage layout; changing it mid-run would invalidate byte accounting."""
        return self._episode_format

    @property
    def verify_archive(self) -> bool:
        """Fixed integrity mode: validate ZIP CRCs and record its byte hash."""
        return self._verify_archive

    def _validate_transition(
        self, observation: Observation, result: StepResult
    ) -> tuple[Array, Array]:
        """Validate batch layouts and return executed actions and active row IDs."""
        count = self.num_envs
        _validate_observation(observation, count)
        _validate_observation(result.observation, count)
        expected_proprio = (count, self.manifest["env_spec"]["proprio_shape"][0])
        if any(
            batch["proprio"].shape != expected_proprio
            for batch in (observation, result.observation)
        ):
            raise ValueError("Recorded proprio layout must match environment spec")
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
            or action.dtype.kind not in "fiu"
            or not _all_finite(action)
        ):
            raise ValueError("Recorded action must be finite [N,A]")
        if (
            not isinstance(result.reward, np.ndarray)
            or result.reward.shape != (count,)
            or result.reward.dtype.kind not in "fiu"
            or not _all_finite(result.reward)
        ):
            raise ValueError("Recorded reward must be finite [N]")
        if (result.terminated & result.truncated).any():
            raise ValueError("A transition cannot be both terminated and truncated")
        expected_ids = np.arange(count)
        if not np.array_equal(
            observation["env_id"], expected_ids
        ) or not np.array_equal(result.observation["env_id"], expected_ids):
            raise ValueError("Recorder requires canonical environment row order")
        return action, np.flatnonzero(result.info["active"])

    def _prepare(self, observation: Observation, result: StepResult) -> list[tuple]:
        """Validate and copy an entire batch without changing recorder state."""
        action, active_ids = self._validate_transition(observation, result)
        count = self.num_envs
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
                (int(env_id), int(observation["episode_id"][env_id])) in self.pending
                for env_id in active_ids
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
        fields = [
            (name, f"obs/{name}", f"next/{name}", values)
            for name, values in observation.items()
        ]
        next_fields = [
            (f"next/{name}", values) for name, values in result.observation.items()
        ]
        for name, _, _, values in fields:
            next_values = result.observation[name]
            if values.shape != next_values.shape or values.dtype != next_values.dtype:
                raise ValueError(f"Observation layout changed for {name}")
        row_fields = {"action", "reward", "terminated", "truncated"}
        row_fields.update(obs for _, obs, _, _ in fields)
        row_fields.update(name for name, _ in next_fields)
        compact = self.episode_format == "compact"
        prepared = []
        for env_id in active_ids:
            episode = int(observation["episode_id"][env_id])
            key = (int(env_id), episode)
            step_id = int(observation["step_id"][env_id])
            if key[0] in pending_episodes and pending_episodes[key[0]] != episode:
                raise ValueError(
                    "An environment was reset before its recorded episode ended"
                )
            rows = self.pending.get(key, [])
            if rows and rows[-1].keys() != row_fields:
                raise ValueError("Observation schema changed within an episode")
            if step_id != len(rows):
                raise ValueError(
                    "Recording must start at reset and contain consecutive transitions"
                )
            if len(rows) >= self.max_steps:
                raise BufferError("Episode exceeds configured recorder capacity")
            # A pending episode passed this check at its first transition.
            # The run directory belongs to this writer; publication rechecks
            # the path in case another file appeared while it was buffered.
            if (
                key in self._writes
                or not rows
                and os.path.exists(
                    os.path.join(self.root, f"env{key[0]:05d}-episode{key[1]:08d}.npz")
                )
            ):
                raise ValueError("Duplicate episode ID; do not reseed a recording run")
            if (
                int(result.observation["episode_id"][env_id]) != episode
                or int(result.observation["step_id"][env_id]) != step_id + 1
                or result.observation["time"][env_id] <= observation["time"][env_id]
            ):
                raise ValueError(
                    "next observation must precede reset and advance exactly one step"
                )
            for name, _, next_name, values in fields:
                if rows and (
                    next_name not in rows[-1]
                    or values.shape[1:] != rows[-1][next_name].shape
                    or values.dtype != rows[-1][next_name].dtype
                    or not _array_equal(values[env_id], rows[-1][next_name])
                ):
                    raise ValueError(
                        f"Observation continuity broken or layout changed for {name}"
                    )
            if rows and any(
                value.dtype != rows[-1][name].dtype
                for name, value in (("action", action), ("reward", result.reward))
            ):
                raise ValueError("Action/reward layout changed within an episode")
            row = {
                obs_name: (
                    rows[-1][next_name] if rows and compact else values[env_id].copy()
                )
                for _, obs_name, next_name, values in fields
            }
            row.update({name: values[env_id].copy() for name, values in next_fields})
            row.update(
                action=action[env_id].copy(),
                reward=result.reward[env_id].copy(),
                terminated=result.terminated[env_id].copy(),
                truncated=result.truncated[env_id].copy(),
            )
            prepared.append(
                (key, row, bool(result.terminated[env_id] or result.truncated[env_id]))
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
        if self._closing:
            raise RuntimeError("Recorder is closing; retry close")
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
        if continuing and self.episode_format == "compact":
            return sum(
                value.nbytes
                for name, value in row.items()
                if not name.startswith("obs/")
            )
        return sum(value.nbytes for value in row.values())

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
        # Do not parse unique episode filenames through pathlib: on CPython
        # 3.12 its interned path components live until interpreter shutdown.
        path = os.path.join(self.root, f"{stem}.npz")
        if os.path.exists(path):
            raise ValueError("Duplicate episode ID; do not reseed a recording run")
        temporary = os.path.join(self.root, f"{stem}.npz.tmp")
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
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        checksum = None
        if self.verify_archive:
            # Hash before CRC validation: subsequent changes are detected by
            # strict readers even if they happen after this validation pass.
            checksum = file_sha256(temporary)
            try:
                with zipfile.ZipFile(temporary) as archive:
                    bad_entry = archive.testzip()
                if bad_entry is not None:
                    raise zipfile.BadZipFile(f"Bad CRC for {bad_entry}")
            except zipfile.BadZipFile as error:
                raise zipfile.BadZipFile(
                    f"Episode archive failed verification: {path}: {error}"
                ) from error
        os.replace(temporary, path)
        marker = {
            "schema_version": 2 if self.episode_format == "compact" else 1,
            "complete": True,
            "steps": len(rows),
            "env_id": key[0],
            "episode_id": key[1],
        }
        if checksum is not None:
            marker["archive_sha256"] = checksum
        _json_atomic(
            os.path.join(self.root, f"{stem}.json"),
            marker,
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
                buffered = self.buffered_bytes
                completed = self.manifest["completed_episodes"]
                try:
                    self.buffered_bytes = buffered - size
                    self.manifest["completed_episodes"] = completed + 1
                    del self._writes[key]
                except BaseException:
                    # A caller interruption must not count the same completed
                    # future again when append/close retries its collection.
                    self.buffered_bytes = buffered
                    self.manifest["completed_episodes"] = completed
                    self._writes[key] = (future, rows, size)
                    raise
                continue
            del self._writes[key]
        return error

    def close(self) -> None:
        """Wait for queued writes, finalize the manifest and discard incomplete rows."""
        if self.closed:
            return
        self._closing = True
        error = None

        def drain(operation):
            nonlocal error
            try:
                return operation()
            except BaseException as caught:
                error = error or caught
                try:
                    return operation()
                except BaseException as repeated:
                    # Retain the undrained handles/rows for an explicit retry,
                    # and prevent appends to an already shut-down executor.
                    self.failure = self.failure or error
                    self.close_error = repeated
                    raise error from repeated

        if self._executor is not None:
            drain(lambda: self._executor.shutdown(wait=True))
            self._executor = None
        # A prior interrupted close may have released the executor already.
        # Explicitly wait for each future, even after an interrupted thread join.
        for key in tuple(self._writes):
            if key in self._writes:
                worker_error = drain(lambda key=key: self._collect_writes(wait_for=key))
                error = error or worker_error
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
        if self._closing:
            raise RuntimeError("Recorder is closing; retry close")
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
    if options is not None and not isinstance(options, EpisodeReadOptions):
        raise ValueError("options must be EpisodeReadOptions")
    options = options or EpisodeReadOptions()
    root = Path(directory)
    manifest = _dataset_manifest(root)
    completed_index = 0
    with os.scandir(root) as entries:
        paths = sorted(
            (entry.path for entry in entries if fnmatch.fnmatch(entry.name, "*.npz")),
            key=os.path.normcase,
        )
    for path in paths:
        marker = os.path.splitext(path)[0] + ".json"
        if not os.path.exists(marker):
            continue
        with open(marker, encoding="utf-8") as stream:
            metadata = json.load(stream)
        if not isinstance(metadata, dict):
            raise ValueError(f"Episode marker must be a JSON object: {marker}")
        schema = metadata.get("schema_version")
        if metadata.get("complete") is False:
            continue
        if metadata.get("complete") is not True:
            raise ValueError(f"Episode completion marker must be boolean: {marker}")
        if type(schema) is not int or schema not in (1, 2):
            raise ValueError(f"Unsupported episode schema: {path}")
        selected = completed_index % options.num_shards == options.shard_index
        completed_index += 1
        if not selected:
            continue
        if options.verify_archive:
            expected = metadata.get("archive_sha256")
            if (
                not isinstance(expected, str)
                or len(expected) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in expected.lower()
                )
            ):
                raise ValueError(f"Archive SHA256 is missing or invalid: {path}")
        try:
            source = open(path, "rb") if options.verify_archive else nullcontext()
            with source as stream:
                if options.verify_archive:
                    if stream_sha256(stream) != expected.lower():
                        raise ValueError(f"Episode archive SHA256 mismatch: {path}")
                    stream.seek(0)
                episode = _load_episode(
                    path,
                    metadata,
                    options,
                    owned=owned,
                    spec=manifest.get("env_spec"),
                    source=stream,
                )
        except zipfile.BadZipFile as error:
            raise zipfile.BadZipFile(
                f"Corrupt episode archive: {path}: {error}"
            ) from error
        yield episode
        del episode


def _load_episode(
    path: str | Path,
    metadata: dict,
    options: EpisodeReadOptions,
    *,
    owned: bool,
    spec: dict | None,
    source=None,
) -> Observation:
    """Decode and validate in a short-lived scope so scratch arrays are released."""
    schema = metadata["schema_version"]
    for field_name in ("env_id", "episode_id"):
        if type(metadata.get(field_name)) is not int or metadata[field_name] < 0:
            raise ValueError(f"Invalid episode identity: {path}")
    step_count = metadata["steps"]
    if type(step_count) is not int or step_count < 1:
        raise ValueError(f"Corrupt episode lengths: {path}")
    # Read selected payloads once, after checking the archive field names.
    with np.load(path if source is None else source, allow_pickle=False) as archive:
        archive_keys = archive.files
        observation_prefix = "state/" if schema == 2 else "obs/"
        observation_keys = {
            field_key[len(observation_prefix) :]
            for field_key in archive_keys
            if field_key.startswith(observation_prefix)
        }
        if schema == 2:
            if any(
                not field_key.startswith("state/")
                and field_key not in {"action", "reward", "terminated", "truncated"}
                for field_key in archive_keys
            ):
                raise ValueError(f"Invalid compact episode fields: {path}")
        elif options.observation_keys is not None:
            next_observation_keys = {
                field_key[5:]
                for field_key in archive_keys
                if field_key.startswith("next/")
            }
            if observation_keys != next_observation_keys:
                raise ValueError(f"Observation schema changed: {path}")
        if options.observation_keys is not None:
            requested_keys = _REQUIRED_OBSERVATIONS | set(options.observation_keys)
            missing_keys = requested_keys - observation_keys
            if missing_keys:
                raise ValueError(
                    f"Missing requested observations {sorted(missing_keys)}: {path}"
                )
            archive_keys = [
                field_key
                for field_key in archive_keys
                if field_key in {"action", "reward", "terminated", "truncated"}
                or (
                    field_key.startswith(observation_prefix)
                    and field_key[len(observation_prefix) :] in requested_keys
                )
                or (field_key.startswith("next/") and field_key[5:] in requested_keys)
            ]
        episode_fields = {field_key: archive[field_key] for field_key in archive_keys}
    if schema == 2:
        compact_states = {
            field_name[6:]: field_array
            for (field_name, field_array) in episode_fields.items()
            if field_name.startswith("state/")
        }
        if any(
            field_array.ndim < 1 or len(field_array) != step_count + 1
            for field_array in compact_states.values()
        ):
            raise ValueError(f"Corrupt compact observation lengths: {path}")
        episode_fields = {
            field_name: field_array
            for (field_name, field_array) in episode_fields.items()
            if not field_name.startswith("state/")
        }
        for field_name, field_values in compact_states.items():
            episode_fields[f"obs/{field_name}"] = field_values[:-1]
            episode_fields[f"next/{field_name}"] = field_values[1:]
    if any(
        field_value.ndim < 1 or len(field_value) != step_count
        for field_value in episode_fields.values()
    ):
        raise ValueError(f"Corrupt episode lengths: {path}")
    required_transition_fields = {"action", "reward", "terminated", "truncated"}
    if not required_transition_fields <= episode_fields.keys():
        raise ValueError(f"Missing transition fields: {path}")
    observations = {
        field_key[4:]: field_value
        for (field_key, field_value) in episode_fields.items()
        if field_key.startswith("obs/")
    }
    next_observations = {
        field_key[5:]: field_value
        for (field_key, field_value) in episode_fields.items()
        if field_key.startswith("next/")
    }
    if schema == 2:
        _validate_observation(compact_states, step_count + 1)
    else:
        _validate_observation(observations, step_count)
        _validate_observation(next_observations, step_count)
    if observations.keys() != next_observations.keys():
        raise ValueError(f"Observation schema changed: {path}")
    if spec is not None and (
        episode_fields["action"].shape != (step_count, spec["action_shape"][0])
        or observations["proprio"].shape != (step_count, spec["proprio_shape"][0])
        or next_observations["proprio"].shape != (step_count, spec["proprio_shape"][0])
    ):
        raise ValueError(
            f"Episode action/proprio layout differs from environment spec: {path}"
        )
    for field_name in ("terminated", "truncated"):
        if (
            episode_fields[field_name].shape != (step_count,)
            or episode_fields[field_name].dtype != bool
        ):
            raise ValueError(f"Invalid {field_name}: {path}")
    if (episode_fields["terminated"] & episode_fields["truncated"]).any():
        raise ValueError(f"Conflicting terminal flags: {path}")
    if (
        episode_fields["action"].ndim != 2
        or episode_fields["action"].dtype.kind not in "fiu"
        or (not _all_finite(episode_fields["action"]))
        or (episode_fields["reward"].shape != (step_count,))
        or (episode_fields["reward"].dtype.kind not in "fiu")
        or (not _all_finite(episode_fields["reward"]))
    ):
        raise ValueError(f"Invalid action/reward: {path}")
    completed_steps = episode_fields["terminated"] | episode_fields["truncated"]
    if not completed_steps[-1] or completed_steps[:-1].any():
        raise ValueError(f"Invalid episode boundary: {path}")
    # Share the reference range and release it before allocating public copies.
    expected_step_ids = np.arange(step_count + 1)
    if not np.array_equal(episode_fields["obs/step_id"], expected_step_ids[:-1]):
        raise ValueError(f"Nonconsecutive episode: {path}")
    if not np.array_equal(next_observations["step_id"], expected_step_ids[1:]):
        raise ValueError(f"Invalid next step IDs: {path}")
    del expected_step_ids
    for field_name in ("env_id", "episode_id"):
        if not (
            metadata[field_name] <= np.iinfo(observations[field_name].dtype).max
            and metadata[field_name]
            <= np.iinfo(next_observations[field_name].dtype).max
            and (observations[field_name] == metadata[field_name]).all()
            and (next_observations[field_name] == metadata[field_name]).all()
        ):
            raise ValueError(f"Episode identity mismatch: {path}")
    if not (next_observations["time"] > observations["time"]).all():
        raise ValueError(f"Time must advance: {path}")
    # Compact observations already share adjacent slices of the same state array.
    if schema == 1:
        for field_name, field_array in observations.items():
            if (
                field_array.shape != next_observations[field_name].shape
                or field_array.dtype != next_observations[field_name].dtype
                or not _array_equal(field_array[1:], next_observations[field_name][:-1])
            ):
                raise ValueError(
                    f"Observation continuity broken for {field_name}: {path}"
                )
    # Compact state views stay private; public observations own separate arrays.
    if schema == 2 and owned:
        for field_name in tuple(compact_states):
            episode_fields[f"obs/{field_name}"] = observations.pop(field_name).copy()
            episode_fields[f"next/{field_name}"] = next_observations.pop(
                field_name
            ).copy()
            del compact_states[field_name]
    return episode_fields


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
    instruction = manifest.get("env_spec", {}).get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Action windows require a recorded dataset instruction")
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
