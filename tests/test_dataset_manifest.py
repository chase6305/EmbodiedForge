"""Dataset readers reject another workflow or a contradictory stored layout."""

import json

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.batching import window_batches
from embodiedforge.data import (
    EpisodeReadOptions,
    EpisodeRecorder,
    action_windows,
    read_episodes,
)
from embodiedforge.runtime.records import run_identity


@pytest.fixture(params=["transitions", "compact"])
def dataset(tmp_path, request):
    root = tmp_path / "episodes"
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        with EpisodeRecorder(root, env, episode_format=request.param) as writer:
            writer.append(env.observe(), env.step(np.zeros((1, 2))))
    return root


@pytest.mark.parametrize("reader", [read_episodes, action_windows, window_batches])
@pytest.mark.parametrize(
    "replacement", [[], None, {"schema_version": 1, "status": "complete"}]
)
def test_wrong_run_kind_is_rejected_before_archive_access(
    dataset, reader, replacement, monkeypatch
):
    (dataset / "run.json").write_text(json.dumps(replacement))
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: pytest.fail("wrong manifest opened an archive")
    )
    with pytest.raises(ValueError, match="manifest"):
        list(reader(dataset))


@pytest.mark.parametrize(
    "identity",
    [
        run_identity("h1-native", "evaluate"),
        run_identity("core-data", "train"),
        {"schema_version": 2},
    ],
)
def test_explicit_identity_cannot_be_reinterpreted_as_a_dataset(dataset, identity):
    path = dataset / "run.json"
    manifest = json.loads(path.read_text())
    manifest["run_contract"] = identity
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity|schema"):
        list(read_episodes(dataset))


@pytest.mark.parametrize(
    "field,value",
    [("run_id", ""), ("completed_episodes", True), ("config", []), ("env_spec", None)],
)
def test_invalid_manifest_fields_are_rejected(dataset, field, value):
    path = dataset / "run.json"
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        list(read_episodes(dataset))


@pytest.mark.parametrize("field", ["action_shape", "proprio_shape"])
@pytest.mark.parametrize("projection", [None, ()])
def test_required_payload_layout_matches_run_spec_even_with_projection(
    dataset, field, projection
):
    path = dataset / "run.json"
    manifest = json.loads(path.read_text())
    manifest["env_spec"][field][0] += 1
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="layout differs"):
        list(
            read_episodes(
                dataset, options=EpisodeReadOptions(observation_keys=projection)
            )
        )


@pytest.mark.parametrize(
    "marker",
    [
        [],
        None,
        {"schema_version": 1, "complete": "false"},
        {"schema_version": 1, "complete": 1},
    ],
)
def test_episode_marker_requires_an_object_and_boolean_completion(dataset, marker):
    path = next(dataset.glob("env*.json"))
    path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="marker"):
        list(read_episodes(dataset))


def test_legacy_dataset_without_run_contract_remains_readable(dataset):
    path = dataset / "run.json"
    manifest = json.loads(path.read_text())
    del manifest["run_contract"]
    manifest.pop("environment_family")
    path.write_text(json.dumps(manifest))
    assert len(list(read_episodes(dataset))) == 1
    assert len(list(action_windows(dataset))) == 1


@pytest.mark.parametrize("options", [0, False, {}, []])
def test_options_are_checked_before_opening_any_run(tmp_path, options):
    with pytest.raises(ValueError, match="EpisodeReadOptions"):
        list(read_episodes(tmp_path / "missing", options=options))
