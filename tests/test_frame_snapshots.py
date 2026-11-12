"""Frame transport selects owned display state rather than training batches."""

import pickle
from dataclasses import dataclass

import numpy as np
import pytest

from embodiedforge.core import SceneUpdate, StateSnapshot
from embodiedforge.viewers._frame_snapshot import frame_snapshot
from embodiedforge.viewers.robot import RobotSceneUpdate


class UnserializableImage(np.ndarray):
    def __reduce_ex__(self, protocol):
        raise AssertionError("An unused training image reached frame transport")


def point_scene():
    position = np.arange(24, dtype=">f8").reshape(6, 4)[:, ::2]
    velocity = np.asfortranarray(np.arange(12, dtype=np.float32).reshape(6, 2))
    state = StateSnapshot(
        position,
        velocity,
        np.arange(6, dtype=np.float64),
        np.arange(6, dtype=">i8"),
    )
    target = np.empty((6, 2), dtype=np.dtype("f4", metadata={"units": "meters"}))
    target[:] = -position
    # The source agent aliases state.position; each projected field still owns
    # its row so it cannot keep an entire batch alive through a view.
    return SceneUpdate(state, {"agent": position, "target": target})


def assert_owned_row(actual, source, env_id):
    np.testing.assert_array_equal(actual[0], source[env_id])
    assert actual.shape == (1, *source.shape[1:])
    assert actual.dtype == source.dtype
    assert actual.dtype.str == source.dtype.str
    assert actual.dtype.metadata == source.dtype.metadata
    assert actual.flags.owndata and actual.flags.c_contiguous
    assert not np.shares_memory(actual, source)


@pytest.mark.parametrize("backend", ["raster", "mujoco", "gl", "rtx"])
def test_point_snapshot_selects_owned_row_and_only_needed_observations(backend):
    scene = point_scene()
    image = np.zeros((6, 1, 2, 2, 3), dtype=np.uint8).view(UnserializableImage)
    observation = {"time": scene.state.time + 100, "rgb": image}
    scene.entities["unused"] = image
    selected, selected_observation, selected_id = frame_snapshot(
        scene, observation, 4, backend
    )
    assert selected_id == 0
    assert selected.entities.keys() == {"agent", "target"}
    for name in ("position", "velocity", "time", "version"):
        assert_owned_row(getattr(selected.state, name), getattr(scene.state, name), 4)
    for name in ("agent", "target"):
        assert_owned_row(selected.entities[name], scene.entities[name], 4)
    assert not np.shares_memory(selected.entities["agent"], selected.state.position)
    if backend in ("gl", "rtx"):
        assert selected_observation.keys() == {"time"}
        assert_owned_row(selected_observation["time"], observation["time"], 4)
        assert selected_observation["time"][0] != selected.state.time[0]
    else:
        assert selected_observation == {}
    with pytest.raises(AssertionError, match="unused training image"):
        pickle.dumps((scene, observation))
    pickle.dumps((selected, selected_observation, selected_id))
    saved_agent = selected.entities["agent"].copy()
    scene.state.position.fill(99)
    np.testing.assert_array_equal(selected.entities["agent"], saved_agent)
    selected.state.position.fill(-123)
    np.testing.assert_array_equal(selected.entities["agent"], saved_agent)


@pytest.mark.parametrize("backend", ["raster", "mujoco"])
def test_point_renderers_that_ignore_observations_do_not_require_time(backend):
    scene, observation = point_scene(), {}
    selected, selected_observation, selected_id = frame_snapshot(
        scene, observation, 2, backend
    )
    assert selected_id == 0 and selected_observation == {}
    assert_owned_row(selected.entities["agent"], scene.entities["agent"], 2)


@pytest.mark.parametrize("backend", ["mujoco", "gl", "rtx"])
def test_robot_snapshot_uses_only_selected_qpos_and_scene_time(backend):
    scene = RobotSceneUpdate(
        np.arange(42, dtype=">f4").reshape(6, 7)[:, ::-1],
        np.arange(6, dtype=np.float64),
    )
    observation = {
        "rgb": np.zeros((6, 1, 2, 2, 3), dtype=np.uint8).view(UnserializableImage)
    }
    selected, selected_observation, selected_id = frame_snapshot(
        scene, observation, 3, backend
    )
    assert selected_id == 0 and selected_observation == {}
    assert_owned_row(selected.qpos, scene.qpos, 3)
    assert_owned_row(selected.time, scene.time, 3)
    pickle.dumps((selected, selected_observation, selected_id))
    before = scene.qpos.copy()
    selected.qpos.fill(-9)
    np.testing.assert_array_equal(scene.qpos, before)


@pytest.mark.parametrize(
    "kind",
    ["unknown", "point_subclass", "state_subclass", "robot_subclass", "array_subclass"],
)
def test_custom_snapshots_keep_all_fields_original_identity_and_global_id(kind):
    @dataclass
    class UnknownScene:
        payload: object

    class CustomPoint(SceneUpdate):
        pass

    class CustomState(StateSnapshot):
        pass

    class CustomRobot(RobotSceneUpdate):
        pass

    class CustomArray(np.ndarray):
        pass

    scene = point_scene()
    if kind == "unknown":
        scene = UnknownScene({"custom": object()})
    elif kind == "point_subclass":
        scene = CustomPoint(scene.state, scene.entities)
    elif kind == "state_subclass":
        scene.state = CustomState(**vars(scene.state))
    elif kind == "robot_subclass":
        scene = CustomRobot(np.zeros((6, 7)), np.zeros(6))
    else:
        scene.entities["agent"] = scene.entities["agent"].view(CustomArray)
    observation = {"time": np.arange(6), "custom": object()}
    selected, selected_observation, selected_id = frame_snapshot(
        scene, observation, 5, "gl"
    )
    assert selected is scene and selected_observation is observation
    assert selected_id == 5
