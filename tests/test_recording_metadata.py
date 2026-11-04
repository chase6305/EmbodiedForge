"""Recording provenance is explicit, copied, and checked before directory creation."""

from dataclasses import replace

import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, RecordingMetadata


def metadata():
    return RecordingMetadata(
        family="custom",
        max_steps=5,
        seed=0,
        instruction="Follow the command",
        scene="custom:v1",
        scene_spec={},
        config={},
    )


@pytest.mark.parametrize(
    "change",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"seed": -1},
        {"seed": True},
        {"family": ""},
        {"instruction": " "},
        {"scene": None},
        {"scene_spec": []},
        {"config": None},
        {"calibration": []},
        {"source": {"bad": float("nan")}},
    ],
)
def test_invalid_recording_metadata_is_rejected(change):
    with pytest.raises((ValueError, TypeError)):
        replace(metadata(), **change)


def test_mutated_nested_metadata_is_revalidated_before_creating_output(tmp_path):
    description = metadata()
    description.config["bad"] = float("inf")
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(ValueError):
            EpisodeRecorder(tmp_path / "episodes", env, metadata=description)
    assert not (tmp_path / "episodes").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", "cuda"),
        ("num_envs", True),
        ("action_shape", [0]),
        ("proprio_shape", [2, 3]),
        ("task", None),
        ("control_dt", float("nan")),
        ("action_units", None),
        ("action_range", [2, 1]),
        ("action_range", [True, 2]),
    ],
)
def test_invalid_environment_spec_does_not_create_a_recording(tmp_path, field, value):
    class Environment:
        spec = {
            "device": "cpu",
            "num_envs": 2,
            "action_shape": [2],
            "proprio_shape": [4],
            "task": "custom",
            "control_dt": 0.02,
            "action_units": "newtons_xy",
            "action_range": [-1, 1],
            field: value,
        }

    with pytest.raises(ValueError):
        EpisodeRecorder(tmp_path / "episodes", Environment(), metadata=metadata())
    assert not (tmp_path / "episodes").exists()
