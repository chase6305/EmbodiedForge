"""Stateless episode archive decoding and shared numeric schema validation.

The data API owns directory traversal, projection/shard options, hash checking,
recorder state and publication. This module interprets one archive and its
manifest without importing those APIs. Private compact views are permitted only
when the caller explicitly takes responsibility for copying window results.
"""

import json
from pathlib import Path

import numpy as np

from .core import Array, Observation
from .envs.specs import validate_environment_spec as _validate_environment_spec
from .runtime.records import validate_run_identity

_REQUIRED_OBSERVATIONS = frozenset(
    {"env_id", "episode_id", "step_id", "time", "proprio"}
)
# NumPy 1.x can override ndarray equality globally; scalar equality bypasses it.
_ALLOW_SCALAR_EQUALITY = not hasattr(np, "set_numeric_ops")


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


def _load_episode(
    path: str | Path,
    metadata: dict,
    observation_keys: tuple[str, ...] | None,
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
        available_observation_keys = {
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
        elif observation_keys is not None:
            next_observation_keys = {
                field_key[5:]
                for field_key in archive_keys
                if field_key.startswith("next/")
            }
            if available_observation_keys != next_observation_keys:
                raise ValueError(f"Observation schema changed: {path}")
        if observation_keys is not None:
            requested_keys = _REQUIRED_OBSERVATIONS | set(observation_keys)
            missing_keys = requested_keys - available_observation_keys
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
