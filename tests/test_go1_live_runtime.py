"""Exercise live controls using the actual mjbatch task and PyTorch policy."""

import numpy as np
import pytest

from embodiedforge._go1_live_worker import Go1LiveRuntime
from embodiedforge.recipes import sha256

pytest.importorskip("mjbatch")
pytest.importorskip("mujoco_menagerie")
torch = pytest.importorskip("torch")


@pytest.fixture
def runtime(tmp_path):
    from embodiedforge.locomotion.go1_ppo import ActorCritic

    torch.manual_seed(17)
    policy = ActorCritic()
    checkpoint = tmp_path / "model.pt"
    contract = {
        "checkpoint_iteration": 0,
        "task_semantics": "transition-v2",
        "command_profile": "original",
        "reward_profile": "original",
    }
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "iteration": 0,
            **{k: v for k, v in contract.items() if k != "checkpoint_iteration"},
        },
        checkpoint,
    )
    request = {
        "checkpoint": str(checkpoint),
        "sha256": sha256(checkpoint),
        "contract": contract,
        "threads": 1,
        "num_envs": 2,
        "seed": 21,
        "episode_steps": 3,
        "model_path": str(tmp_path / "model.mjb"),
    }
    return Go1LiveRuntime(request), request


def test_commands_reach_policy_observation_and_stats_stay_frozen(runtime):
    owner, _ = runtime
    before = {k: v.clone() for k, v in owner.policy.state_dict().items()}
    owner.execute({"action": "velocity", "env_id": 1, "value": [0.5, 0.2, -0.3]})
    np.testing.assert_array_equal(owner.env.obs()[0, 45:48], [0, 0, 0])
    np.testing.assert_allclose(owner.env.obs()[1, 45:48], [0.5, 0.2, -0.3])
    for _ in range(2):
        state = owner.execute({"action": "step"})
        assert np.isfinite(state["qpos"]).all() and np.isfinite(state["reward"]).all()
        assert state["command"] == [[0, 0, 0], [0.5, 0.2, -0.3]]
    assert state["step"] == [2, 2]
    for name, value in owner.policy.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_terminal_freeze_and_selected_reset(runtime):
    owner, _ = runtime
    for _ in range(3):
        last = owner.execute({"action": "step"})
    assert last["done"] == [True, True]
    assert owner.execute({"action": "step"}) == last
    state = owner.execute({"action": "reset", "env_id": 0})
    assert state["step"] == [0, 3] and state["episode"] == [1, 0]
    assert state["qpos"][1] == last["qpos"][1]
    assert owner.execute({"action": "step"}) == state
    owner.execute({"action": "reset", "env_id": 1})
    assert owner.execute({"action": "step"})["step"] == [1, 1]


@pytest.mark.parametrize(
    "message",
    [
        {"action": "velocity", "env_id": 0, "value": [100, 0, 0]},
        {"action": "velocity", "env_id": True, "value": [0, 0, 0]},
        {"action": "velocity", "env_id": 0, "value": [float("nan"), 0, 0]},
        {"action": "velocity", "env_id": 0, "value": [True, 0, 0]},
        {"action": "reset", "env_id": 2},
    ],
)
def test_invalid_controls_do_not_mutate_simulation(runtime, message):
    owner, _ = runtime
    before = owner.snapshot()
    with pytest.raises(ValueError):
        owner.execute(message)
    assert owner.snapshot() == before


def test_compiled_render_model_matches_live_model(runtime):
    import mujoco

    owner, request = runtime
    model = mujoco.MjModel.from_binary_path(request["model_path"])
    source = owner.env.batch.model
    assert model.nq == source.nq and model.ngeom == source.ngeom
    np.testing.assert_array_equal(model.mesh_vert, source.mesh_vert)
    assert (
        sha256(__import__("pathlib").Path(request["model_path"]))
        == owner.metadata["model_sha256"]
    )


def test_worker_rejects_checksum_and_contract_changes(runtime):
    _, request = runtime
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        Go1LiveRuntime({**request, "sha256": "wrong"})
    changed = {
        **request,
        "contract": {**request["contract"], "command_profile": "lateral-v1"},
    }
    with pytest.raises(ValueError, match="command_profile differs"):
        Go1LiveRuntime(changed)
    changed = {
        **request,
        "contract": {
            **request["contract"],
            "runtime": {"versions": {"mujoco": "0.0.0"}},
        },
    }
    with pytest.raises(ValueError, match="SDK version mismatch"):
        Go1LiveRuntime(changed)


def test_worker_rejects_negative_normalization_variance(runtime):
    from pathlib import Path

    _, request = runtime
    checkpoint = Path(request["checkpoint"])
    state = torch.load(checkpoint, weights_only=True)
    state["model_state_dict"]["var"][0] = -1
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="normalization statistics"):
        Go1LiveRuntime({**request, "sha256": sha256(checkpoint)})
