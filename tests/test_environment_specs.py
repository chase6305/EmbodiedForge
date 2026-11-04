"""Environment versions fail before reset, storage creation or model loading."""

import json
from copy import deepcopy

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, RecordingMetadata, read_episodes
from embodiedforge.envs.evaluation import evaluate_policy
from embodiedforge.envs.specs import validate_environment_spec


def description():
    return {
        "schema_version": 1,
        "task": "custom",
        "device": "cpu",
        "done_handling": "reset_before_next_step",
        "num_envs": 2,
        "action_shape": [2],
        "proprio_shape": [4],
        "control_dt": 0.02,
        "action_units": "newtons_xy",
        "action_range": [-1, 1],
    }


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 0},
        {"schema_version": 2},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": None},
        {"done_handling": "autoreset"},
        {"done_handling": []},
        {"control_dt": 10**400},
        {"action_range": [-1, 10**400]},
        {"task": " "},
        {"action_units": " "},
    ],
)
def test_invalid_spec_is_rejected_before_environment_or_storage_side_effects(
    tmp_path, change
):
    class Environment:
        spec = {**description(), **change}

        def reset(self):
            pytest.fail("Invalid environment must not reset")

    with pytest.raises(ValueError):
        evaluate_policy(Environment(), lambda obs: None, steps=1)
    with pytest.raises(ValueError):
        EpisodeRecorder(
            tmp_path / "episodes",
            Environment(),
            metadata=RecordingMetadata(
                family="custom",
                max_steps=1,
                seed=0,
                instruction="Hold",
                scene="custom:v1",
                scene_spec={},
                config={},
            ),
        )
    assert not (tmp_path / "episodes").exists()


def test_versioned_specs_require_explicit_done_handling_and_validation_is_pure():
    spec = description()
    del spec["done_handling"]
    with pytest.raises(ValueError, match="done_handling"):
        validate_environment_spec(spec)
    del spec["schema_version"]
    before = deepcopy(spec)
    validate_environment_spec(spec)
    assert spec == before
    assert "schema_version" not in spec


@pytest.mark.parametrize("versioned", [False, True])
def test_readers_support_legacy_specs_and_reject_explicit_unknown_versions(
    tmp_path, monkeypatch, versioned
):
    root = tmp_path / "episodes"
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        assert env.spec["schema_version"] == 1
        assert env.spec["done_handling"] == "freeze"
        with EpisodeRecorder(root, env) as writer:
            writer.append(env.observe(), env.step(np.zeros((1, 2))))
    path = root / "run.json"
    manifest = json.loads(path.read_text())
    if not versioned:
        del manifest["env_spec"]["schema_version"]
        del manifest["env_spec"]["done_handling"]
        path.write_text(json.dumps(manifest))
    assert len(list(read_episodes(root))) == 1
    manifest["env_spec"]["schema_version"] = 2
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: pytest.fail("Unknown version opened an archive")
    )
    with pytest.raises(ValueError, match="schema_version"):
        list(read_episodes(root))


def test_unknown_checkpoint_environment_version_fails_before_policy_creation(
    monkeypatch,
):
    pytest.importorskip("torch")
    from embodiedforge import training

    monkeypatch.setattr(
        training,
        "ActorCritic",
        lambda *a, **kw: pytest.fail("Unknown environment constructed a model"),
    )
    with pytest.raises(ValueError, match="schema_version"):
        training._policy_checkpoint(
            {"schema_version": 2, "env_spec": {"schema_version": 2}}, "verified"
        )
