"""Episode boundaries and step IDs retain validation order and owned output."""

import json
import tracemalloc

import numpy as np
import pytest

from embodiedforge._episode_archive import _validate_step_ids
from embodiedforge.data import read_episodes


def episode_archive(root, schema, dtype, steps, damage=None):
    root.mkdir()
    (root / "run.json").write_text(json.dumps({"schema_version": 1}))
    path = root / "episode.npz"
    states = {
        "env_id": np.full(steps + 1, 7, dtype=dtype),
        "episode_id": np.full(steps + 1, 9, dtype=dtype),
        "step_id": np.arange(steps + 1, dtype=dtype),
        "time": np.arange(steps + 1) * 0.02,
        "proprio": np.zeros((steps + 1, 2), dtype=np.float32),
    }
    arrays = {
        "action": np.zeros((steps, 2), dtype=np.float32),
        "reward": np.zeros(steps, dtype=np.float32),
        "terminated": np.arange(steps) == steps - 1,
        "truncated": np.zeros(steps, dtype=bool),
    }
    if schema == 2:
        arrays.update({"state/" + key: value for key, value in states.items()})
        if damage in ("obs", "both"):
            arrays["state/step_id"][0] = steps + 3
        if damage in ("next", "both"):
            arrays["state/step_id"][-1] = steps + 3
    else:
        for key, value in states.items():
            arrays["obs/" + key] = value[:-1].copy()
            arrays["next/" + key] = value[1:].copy()
        if damage in ("obs", "both"):
            arrays["obs/step_id"][0] = steps + 3
        if damage in ("next", "both"):
            arrays["next/step_id"][-1] = steps + 3
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema_version": schema,
                "complete": True,
                "steps": steps,
                "env_id": 7,
                "episode_id": 9,
            }
        )
    )
    return path


INTEGER_DTYPES = [
    np.dtype(name) for name in ("int8", "int16", "int64", "uint8", "uint16", "uint64")
] + [np.dtype(name).newbyteorder("S") for name in ("int64", "uint64")]


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("steps", [1, 5])
@pytest.mark.parametrize("dtype", INTEGER_DTYPES, ids=str)
def test_episode_step_ranges_keep_dtype_values_and_independent_arrays(
    tmp_path, schema, steps, dtype
):
    root = tmp_path / "episode"
    episode_archive(root, schema, dtype, steps)
    (episode,) = read_episodes(root)
    observed, following = episode["obs/step_id"], episode["next/step_id"]
    assert observed.dtype == following.dtype == dtype
    if schema == 2:
        assert observed.flags.owndata and following.flags.owndata
    assert not np.shares_memory(observed, following)
    np.testing.assert_array_equal(observed, np.arange(steps))
    np.testing.assert_array_equal(following, np.arange(1, steps + 1))
    observed.fill(99)
    np.testing.assert_array_equal(following, np.arange(1, steps + 1))
    (reread,) = read_episodes(root)
    np.testing.assert_array_equal(reread["obs/step_id"], np.arange(steps))


@pytest.mark.parametrize("schema", [1, 2])
@pytest.mark.parametrize("damage", ["obs", "next", "both"])
@pytest.mark.parametrize(
    "dtype",
    [np.dtype("int16"), np.dtype("uint64"), *INTEGER_DTYPES[-2:]],
    ids=str,
)
def test_invalid_step_ranges_keep_observation_before_next_error(
    tmp_path, schema, damage, dtype
):
    root = tmp_path / "episode"
    path = episode_archive(root, schema, dtype, 5, damage)
    expected = "Invalid next step IDs" if damage == "next" else "Nonconsecutive episode"
    with pytest.raises(ValueError) as raised:
        list(read_episodes(root))
    assert str(raised.value) == f"{expected}: {path}"


def test_long_episode_step_ranges_bound_scratch_and_keep_error_order():
    count = 1024 * 1024 + 1
    observed = np.arange(count, dtype=np.uint64)
    following = np.arange(1, count + 1, dtype=np.uint64)
    tracemalloc.start()
    try:
        _validate_step_ids(observed, following, "long.npz")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024

    following[0] = 3
    observed[-1] = 0
    with pytest.raises(ValueError, match="Nonconsecutive episode"):
        _validate_step_ids(observed, following, "long.npz")
    observed[-1] = count - 1
    with pytest.raises(ValueError, match="Invalid next step IDs"):
        _validate_step_ids(observed, following, "long.npz")


@pytest.mark.parametrize(
    "terminated,truncated,error",
    [
        ([True, False], [True, True], "Conflicting terminal flags"),
        ([False, True], [False, True], "Conflicting terminal flags"),
        ([True, False], [False, True], "Invalid episode boundary"),
        ([False, False], [False, False], "Invalid episode boundary"),
    ],
)
def test_invalid_episode_boundaries(tmp_path, terminated, truncated, error):
    root = tmp_path / "episode"
    path = episode_archive(root, 1, np.dtype("int32"), 2)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["terminated"] = np.array(terminated)
    arrays["truncated"] = np.array(truncated)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=error):
        list(read_episodes(root))
