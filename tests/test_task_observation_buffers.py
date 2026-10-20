"""Task features retain conversion, dispatch and ownership across layouts."""

import warnings

import numpy as np
import pytest

from embodiedforge.core import StateSnapshot
from embodiedforge.tasks import HoldTask, ReachTask


class LegacyConcatenate(np.ndarray):
    def __array_function__(self, func, types, args, kwargs):
        if func is np.concatenate:
            assert "dtype" not in kwargs and "casting" not in kwargs
        return super().__array_function__(func, types, args, kwargs)


class NoSizeInspection(LegacyConcatenate):
    @property
    def size(self):
        raise RuntimeError("Feature assembly must not inspect subclass size")


@pytest.mark.parametrize("task_class", [ReachTask, HoldTask])
def test_task_observation_does_not_inspect_subclass_size(task_class):
    position = np.zeros((8192, 2)).view(NoSizeInspection)
    velocity = np.zeros((8192, 2))
    task = task_class()
    task.build(8192)
    state = StateSnapshot(position, velocity, np.zeros(8192), np.zeros(8192, np.int64))
    parts = (position, velocity)
    if task_class is ReachTask:
        parts += (task.target - position,)
    expected = np.concatenate(parts, axis=1).astype(np.float32)
    assert task.observe(state)["proprio"].tobytes() == expected.tobytes()


@pytest.mark.parametrize("task_class", [ReachTask, HoldTask])
@pytest.mark.parametrize("count", [0, 1, 8191, 8192, 16385])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
@pytest.mark.parametrize("kind", ["float64", "float32", ">f8", "subclass"])
def test_task_observation_conversion_and_owned_storage(task_class, count, layout, kind):
    rng = np.random.default_rng(812)
    values = []
    for _ in range(3):
        value = rng.normal(size=(count, 2)).astype(
            np.float64 if kind == "subclass" else kind
        )
        if layout == "F":
            value = np.asfortranarray(value)
        elif layout == "strided":
            value = np.repeat(value, 2, axis=0)[::2]
        if kind == "subclass":
            value = value.view(LegacyConcatenate)
        value.flags.writeable = False
        values.append(value)
    position, velocity, target = values
    before = [value.tobytes() for value in values]
    state = StateSnapshot(
        position, velocity, np.zeros(count), np.zeros(count, np.int64)
    )
    task = task_class()
    task.build(count)
    task.target = target
    parts = (position, velocity)
    if task_class is ReachTask:
        parts += (target - position,)
    expected = np.concatenate(parts, axis=1).astype(np.float32)
    actual = task.observe(state)["proprio"]
    assert type(actual) is type(expected)
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.owndata
    assert actual.flags.c_contiguous == expected.flags.c_contiguous
    assert actual.flags.f_contiguous == expected.flags.f_contiguous
    for value in values:
        assert not np.shares_memory(actual, value)
    actual[:] = 99
    assert task.observe(state)["proprio"].tobytes() == expected.tobytes()
    assert before == [value.tobytes() for value in values]


@pytest.mark.parametrize("task_class", [ReachTask, HoldTask])
def test_mixed_integer_promotion_is_preserved(task_class):
    count = 8192
    # The integer lies just above a float32 midpoint. Promoting it to float64
    # before the final cast can round differently from a direct integer cast.
    midpoint = 2**63 + 2**39 + 1
    position = np.full((count, 2), midpoint, np.uint64)
    velocity = np.zeros((count, 2), np.int64)
    task = task_class()
    task.build(count)
    task.target = position.copy()
    state = StateSnapshot(
        position, velocity, np.zeros(count), np.zeros(count, np.int64)
    )
    parts = (position, velocity)
    if task_class is ReachTask:
        parts += (task.target - position,)
    expected = np.concatenate(parts, axis=1).astype(np.float32)
    assert position.astype(np.float32).tobytes() != expected[:, :2].tobytes()
    assert task.observe(state)["proprio"].tobytes() == expected.tobytes()


@pytest.mark.parametrize("task_class", [ReachTask, HoldTask])
def test_float64_cast_preserves_special_value_bits(task_class):
    values = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, np.finfo(float).tiny])
    position = np.resize(values, (8192, 2))
    velocity = np.resize(values[::-1], (8192, 2))
    task = task_class()
    task.build(8192)
    state = StateSnapshot(position, velocity, np.zeros(8192), np.zeros(8192, np.int64))
    with np.errstate(all="ignore"):
        parts = (position, velocity)
        if task_class is ReachTask:
            parts += (task.target - position,)
        expected = np.concatenate(parts, axis=1).astype(np.float32)
        assert task.observe(state)["proprio"].tobytes() == expected.tobytes()


@pytest.mark.parametrize("task_class", [ReachTask, HoldTask])
@pytest.mark.parametrize(
    "mode", ["ignore", "warn", "raise", "warn_no_underflow", "raise_no_underflow"]
)
@pytest.mark.parametrize("profile", ["finite", "extreme", "signaling_nan"])
def test_task_observation_preserves_floating_point_diagnostics(
    task_class, mode, profile
):
    count = 8192
    position = np.full((count, 2), 0.5)
    velocity = np.zeros((count, 2))
    if profile == "extreme":
        position[:] = np.nextafter(0.0, 1.0)
        velocity[:] = np.finfo(np.float64).max
    elif profile == "signaling_nan":
        velocity.view(np.uint64)[:] = 0x7FF0000000000081
    task = task_class()
    task.build(count)
    state = StateSnapshot(
        position, velocity, np.zeros(count), np.zeros(count, np.int64)
    )

    def evaluate(current):
        settings = {"all": mode.split("_")[0]}
        if mode.endswith("no_underflow"):
            settings["under"] = "ignore"
        with warnings.catch_warnings(record=True) as records, np.errstate(**settings):
            warnings.simplefilter("always")
            try:
                if current:
                    output = task.observe(state)["proprio"]
                else:
                    parts = (position, velocity)
                    if task_class is ReachTask:
                        parts += (task.target - position,)
                    output = np.concatenate(parts, axis=1).astype(np.float32)
                result = ("passed", output.dtype.str, output.shape, output.tobytes())
            except FloatingPointError as error:
                result = ("error", str(error))
            return result, [(w.category, str(w.message)) for w in records]

    assert evaluate(True) == evaluate(False)
