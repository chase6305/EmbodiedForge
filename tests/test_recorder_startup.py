"""A recorder that fails to construct must not leave an active run behind."""

import json

import pytest

from embodiedforge import Config, VectorEnv, data


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("after_commit", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_initial_publication_failure_is_finalized_without_entering_body(
    tmp_path, monkeypatch, error_type, after_commit, workers
):
    publish = data._json_atomic
    primary = error_type("initial publication interrupted")
    calls = 0

    def fail_once(path, record):
        nonlocal calls
        calls += 1
        if calls == 1:
            if after_commit:
                publish(path, record)
            raise primary
        publish(path, record)

    monkeypatch.setattr(data, "_json_atomic", fail_once)
    root = tmp_path / "recording"
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(error_type) as caught:
            with data.EpisodeRecorder(root, env, write_workers=workers):
                pytest.fail("A failed constructor entered its body")
    assert caught.value is primary
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["completed_episodes"] == 0
    assert not list(root.glob("*.npz"))


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_executor_creation_failure_cannot_leave_recording_status(
    tmp_path, monkeypatch, error_type
):
    primary = error_type("executor unavailable")

    def fail(**kwargs):
        raise primary

    monkeypatch.setattr(data, "ThreadPoolExecutor", fail)
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(error_type) as caught:
            data.EpisodeRecorder(tmp_path / "run", env, write_workers=1)
    assert caught.value is primary
    assert json.loads((tmp_path / "run/run.json").read_text())["status"] == "failed"


def test_secondary_startup_publication_error_does_not_replace_original(
    tmp_path, monkeypatch, caplog
):
    publish = data._json_atomic
    primary, secondary = KeyboardInterrupt("first"), OSError("disk unavailable")
    calls = 0

    def fail(path, record):
        nonlocal calls
        calls += 1
        if calls == 1:
            publish(path, record)
            raise primary
        raise secondary

    monkeypatch.setattr(data, "_json_atomic", fail)
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(KeyboardInterrupt) as caught:
            data.EpisodeRecorder(tmp_path / "run", env, write_workers=1)
    assert caught.value is primary
    assert calls == 2
    assert "preserving" in caplog.text


def test_dependency_metadata_failure_precedes_directory_claim(tmp_path, monkeypatch):
    primary = OSError("unreadable installed metadata")

    def fail(name):
        raise primary

    monkeypatch.setattr(data, "version", fail)
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(OSError) as caught:
            data.EpisodeRecorder(tmp_path / "run", env)
    assert caught.value is primary
    assert not (tmp_path / "run").exists()
