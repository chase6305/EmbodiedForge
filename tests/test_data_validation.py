"""Large and strided episode fields preserve NumPy validation semantics."""

import tracemalloc

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import (
    EpisodeRecorder,
    _all_finite,
    _array_equal,
    read_episodes,
)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("layout", ["C", "F", "reverse", "non_native"])
def test_large_validation_matches_numpy_across_layouts(dtype, layout):
    source = np.arange(2051 * 2048, dtype=dtype).reshape(2051, 2048)
    if layout == "F":
        source = np.asfortranarray(source)
    elif layout == "reverse":
        source = source[:, ::-1]
    elif layout == "non_native":
        source = source.astype(source.dtype.newbyteorder("S"))
    first, second = source[1:], source[1:].copy(order="K")
    assert _array_equal(first, second) == np.array_equal(first, second)
    assert _all_finite(first) == np.isfinite(first).all()
    assert not _array_equal(first, second[:-1])
    # Differences at both ends must not be missed, including a short final chunk.
    for index in ((0, 0), (-1, -1)):
        saved = second[index].copy()
        second[index] = int(saved) ^ 1 if second.dtype.kind == "u" else saved + 1
        assert _array_equal(first, second) == np.array_equal(first, second) is False
        second[index] = saved
    if source.dtype.kind == "f":
        for index in ((0, 0), (512, 0), (-1, -1)):
            saved = first[index].copy()
            for value in (np.nan, np.inf, -np.inf):
                first[index] = value
                assert not _all_finite(first)
                assert not np.isfinite(first).all()
            first[index] = saved
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("shape", [(0,), (1,), (2, 3), (0, 2, 3)])
def test_small_and_empty_validation_matches_numpy(shape):
    value = np.zeros(shape, dtype=np.float32)
    assert _all_finite(value) == np.isfinite(value).all()
    assert _array_equal(value, value) == np.array_equal(value, value)


def test_large_validation_has_bounded_boolean_scratch():
    first = np.ones(8 * 1024 * 1024, dtype=np.uint8)
    second = first.copy()
    floats = np.ones(4 * 1024 * 1024 + 17, dtype=np.float32)
    for operation in (lambda: _array_equal(first, second), lambda: _all_finite(floats)):
        tracemalloc.start()
        try:
            assert operation()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 2 * 1024 * 1024


def test_large_masked_numeric_fields_keep_numpy_finite_semantics():
    value = np.ma.array(np.ones(4 * 1024 * 1024 + 17, dtype=np.float32), mask=False)
    value[-1] = np.nan
    value.mask[-1] = True
    # nditer operates on underlying data, so subclasses keep NumPy's dispatch
    # rather than losing their mask or custom ufunc behavior in chunked checks.
    assert _all_finite(value) == bool(np.isfinite(value).all()) is True
    value.mask[-1] = False
    assert _all_finite(value) == bool(np.isfinite(value).all()) is False


@pytest.mark.parametrize("damage", ["first", "last", "nonfinite"])
def test_reader_rejects_corrupt_large_numeric_fields(tmp_path, damage):
    directory = tmp_path / "dataset"
    with VectorEnv(Config(num_envs=1, max_steps=3, seed=42)) as env:
        with EpisodeRecorder(directory, env) as writer:
            observation = env.observe()
            for _ in range(3):
                result = env.step(np.zeros((1, 2)))
                writer.append(observation, result)
                observation = result.observation
    path = next(directory.glob("*.npz"))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["obs/large"] = np.zeros((3, 1536, 1536), dtype=np.float32)
    arrays["next/large"] = arrays["obs/large"].copy()
    if damage == "nonfinite":
        arrays["obs/large"][-1, -1, -1] = np.inf
    else:
        index = (0, 0, 0) if damage == "first" else (1, -1, -1)
        arrays["next/large"][index] = 1
    np.savez_compressed(path, **arrays)
    error = "numeric observation" if damage == "nonfinite" else "continuity broken"
    with pytest.raises(ValueError, match=error):
        list(read_episodes(directory))
