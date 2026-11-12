"""Select the state used by built-in frame renderers before process transport."""

import numpy as np

from embodiedforge.core import SceneUpdate, StateSnapshot

from .robot import RobotSceneUpdate


def _row(value: np.ndarray, env_id: int) -> np.ndarray:
    """Own one batch row while retaining its dtype, byte order and metadata."""
    selected = np.empty((1, *value.shape[1:]), dtype=value.dtype)
    selected[0] = value[env_id]
    return selected


def frame_snapshot(scene, observation, env_id, backend):
    """Project known snapshots to local row zero, leaving public IDs unchanged.

    Built-in renderers need point poses or robot qpos, not training observations.
    Only the native point renderer also uses observation time; it can differ
    from scene time. Retained arrays are independent of the caller's batch.

    Unknown snapshot types, array subclasses and non-array fields keep the
    original scene, observations and ID. No fields are discarded on that path.
    The renderer's configuration can retain its original environment count:
    built-in adapters index these snapshots by the returned local ID.
    """
    if type(scene) is SceneUpdate and type(scene.state) is StateSnapshot:
        state = scene.state
        fields = (state.position, state.velocity, state.time, state.version)
        entities = (scene.entities["agent"], scene.entities["target"])
        native = backend in ("gl", "rtx")
        inputs = (*fields, *entities)
        if native:
            inputs += (observation["time"],)
        if not all(type(value) is np.ndarray for value in inputs):
            return scene, observation, env_id
        selected_state = StateSnapshot(*(_row(value, env_id) for value in fields))
        selected_scene = SceneUpdate(
            selected_state,
            {
                name: _row(value, env_id)
                for name, value in zip(("agent", "target"), entities, strict=True)
            },
        )
        selected_observation = (
            {"time": _row(observation["time"], env_id)} if native else {}
        )
        return selected_scene, selected_observation, 0
    if type(scene) is RobotSceneUpdate:
        if type(scene.qpos) is not np.ndarray or type(scene.time) is not np.ndarray:
            return scene, observation, env_id
        return (
            RobotSceneUpdate(_row(scene.qpos, env_id), _row(scene.time, env_id)),
            {},
            0,
        )
    return scene, observation, env_id
