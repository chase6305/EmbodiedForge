"""Interrupted recorder drains preserve owned writes and the first exception."""

import json
import threading

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


def once(operation, error):
    first = True

    def run(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise error
        return operation(*args, **kwargs)

    return run


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, OSError])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("workers", [1, 2])
def test_interrupted_shutdown_still_drains_real_writes_and_releases_threads(
    tmp_path, monkeypatch, error_type, episode_format, workers
):
    gate, entered = threading.Event(), threading.Event()
    primary = error_type("shutdown interrupted")
    with VectorEnv(Config(num_envs=2, max_steps=1)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=workers, episode_format=episode_format
        )
        write = writer._write_episode

        def blocked_write(*args):
            entered.set()
            assert gate.wait(5)
            return write(*args)

        monkeypatch.setattr(writer, "_write_episode", blocked_write)
        writer.append(env.observe(), env.step(np.zeros((2, 2))))
        assert entered.wait(5)
        executor = writer._executor
        shutdown = executor.shutdown
        threads = tuple(executor._threads)

        def finish(*args, **kwargs):
            gate.set()
            return shutdown(*args, **kwargs)

        monkeypatch.setattr(executor, "shutdown", once(finish, primary))
        try:
            with pytest.raises(error_type) as caught:
                writer.close()
            assert caught.value is primary
            assert writer.closed and writer._executor is None
            assert not writer._writes and not writer.pending
            assert writer.buffered_bytes == 0
            assert all(not thread.is_alive() for thread in threads)
            assert writer.failure is None
            manifest = json.loads((writer.root / "run.json").read_text())
            assert manifest["status"] == "closed"
            assert manifest["completed_episodes"] == 2
            assert len(list(read_episodes(writer.root))) == 2
            writer.close()
        finally:
            gate.set()
            monkeypatch.setattr(executor, "shutdown", shutdown)
            writer.close()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, OSError])
def test_close_retries_caller_interruption_while_collecting_a_finished_future(
    tmp_path, monkeypatch, error_type
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)
        writer.append(env.observe(), env.step(np.zeros((1, 2))))
        future = writer._writes[(0, 0)][0]
        result = future.result
        result(timeout=5)
        primary = error_type("caller interrupted collection")
        monkeypatch.setattr(future, "result", once(result, primary))
        try:
            with pytest.raises(error_type) as caught:
                writer.close()
            assert caught.value is primary
            assert writer.closed and not writer._writes
            assert writer.failure is None
            assert writer.buffered_bytes == 0
            manifest = json.loads((writer.root / "run.json").read_text())
            assert manifest["completed_episodes"] == 1
            assert manifest["status"] == "closed"
            assert len(list(read_episodes(writer.root))) == 1
        finally:
            monkeypatch.setattr(future, "result", result)
            writer.close()


@pytest.mark.parametrize("stage", ["shutdown", "collection"])
def test_persistent_drain_fault_preserves_retry_state_and_blocks_append(
    tmp_path, monkeypatch, stage
):
    primary, secondary = KeyboardInterrupt("first interruption"), OSError("again")
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)
        observation = env.observe()
        result = env.step(np.zeros((1, 2)))
        writer.append(observation, result)
        future = writer._writes[(0, 0)][0]
        future.result(timeout=5)
        owner = writer._executor if stage == "shutdown" else future
        name = "shutdown" if stage == "shutdown" else "result"
        operation = getattr(owner, name)
        errors = iter([primary, secondary])

        def fail(*args, **kwargs):
            raise next(errors)

        monkeypatch.setattr(owner, name, fail)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                writer.close()
            assert caught.value is writer.failure is primary
            assert caught.value.__cause__ is writer.close_error is secondary
            assert not writer.closed
            assert writer._writes and writer.buffered_bytes > 0
            with pytest.raises(RuntimeError, match="Recorder failed"):
                writer.append(observation, result)
        finally:
            monkeypatch.setattr(owner, name, operation)
            writer.close()
        assert writer.closed and not writer._writes
        assert writer.buffered_bytes == 0
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["completed_episodes"] == 1


def test_shutdown_interrupt_does_not_hide_worker_failure(tmp_path, monkeypatch):
    with VectorEnv(Config(num_envs=2, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=2)
        primary = KeyboardInterrupt("shutdown interrupted")
        worker_error = OSError("archive failed")
        write = writer._write_episode

        def mixed(key, rows):
            if key[0] == 0:
                raise worker_error
            return write(key, rows)

        monkeypatch.setattr(writer, "_write_episode", mixed)
        writer.append(env.observe(), env.step(np.zeros((2, 2))))
        executor = writer._executor
        shutdown = executor.shutdown
        monkeypatch.setattr(executor, "shutdown", once(shutdown, primary))
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                writer.close()
            assert caught.value is primary
            assert writer.failure is worker_error
            assert writer.closed and not writer._writes
            manifest = json.loads((writer.root / "run.json").read_text())
            assert manifest["status"] == "failed"
            assert manifest["completed_episodes"] == 1
            assert manifest["incomplete_episodes"] == [
                {"env_id": 0, "episode_id": 0, "steps": 1}
            ]
            assert len(list(read_episodes(writer.root))) == 1
        finally:
            monkeypatch.setattr(executor, "shutdown", shutdown)
            writer.close()


def test_body_failure_survives_shutdown_interrupt_after_complete_drain(
    tmp_path, monkeypatch
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)
        primary = ValueError("body failed")
        secondary = KeyboardInterrupt("shutdown interrupted")
        shutdown = writer._executor.shutdown
        monkeypatch.setattr(writer._executor, "shutdown", once(shutdown, secondary))
        try:
            with pytest.raises(ValueError) as caught:
                with writer:
                    writer.append(env.observe(), env.step(np.zeros((1, 2))))
                    raise primary
            assert caught.value is writer.failure is primary
            assert writer.close_error is secondary
            assert writer.closed and not writer._writes
            assert len(list(read_episodes(writer.root))) == 1
        finally:
            writer.close()


def test_shutdown_interrupt_survives_secondary_manifest_failure(tmp_path, monkeypatch):
    from embodiedforge import data

    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)
        writer.append(env.observe(), env.step(np.zeros((1, 2))))
        writer._writes[(0, 0)][0].result(timeout=5)
        primary, secondary = (
            KeyboardInterrupt("shutdown interrupted"),
            OSError("manifest full"),
        )
        executor = writer._executor
        monkeypatch.setattr(executor, "shutdown", once(executor.shutdown, primary))
        publish = data._json_atomic

        def fail(*args):
            raise secondary

        monkeypatch.setattr(data, "_json_atomic", fail)
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                writer.close()
            assert caught.value is primary
            assert caught.value.__cause__ is writer.close_error is secondary
            assert writer.failure is secondary
            assert not writer.closed and not writer._writes
            assert writer._executor is None
            assert len(list(read_episodes(writer.root))) == 1
        finally:
            monkeypatch.setattr(data, "_json_atomic", publish)
            writer.close()
        assert writer.closed
        assert json.loads((writer.root / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
def test_shutdown_interrupt_finalizes_incomplete_episode(
    tmp_path, monkeypatch, episode_format
):
    with VectorEnv(Config(num_envs=1, max_steps=3)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=1, episode_format=episode_format
        )
        writer.append(env.observe(), env.step(np.zeros((1, 2))))
        primary = KeyboardInterrupt("shutdown interrupted")
        monkeypatch.setattr(
            writer._executor, "shutdown", once(writer._executor.shutdown, primary)
        )
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                writer.close()
            assert caught.value is primary
            assert writer.closed and not writer._writes and not writer.pending
            assert writer.buffered_bytes == 0
            manifest = json.loads((writer.root / "run.json").read_text())
            assert manifest["status"] == "closed"
            assert manifest["completed_episodes"] == 0
            assert manifest["incomplete_episodes"] == [
                {"env_id": 0, "episode_id": 0, "steps": 1}
            ]
            assert not list(read_episodes(writer.root))
        finally:
            writer.close()


@pytest.mark.parametrize("stage", ["collect_boundary", "release_boundary"])
@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
def test_interrupted_close_never_reopens_append_before_explicit_retry(
    tmp_path, stage, workers, episode_format
):
    primary = KeyboardInterrupt("between cleanup operations")

    class InterruptedRows(dict):
        interrupted = False

        def __iter__(self):
            if stage == "collect_boundary" and not self.interrupted:
                self.interrupted = True
                raise primary
            return super().__iter__()

        def clear(self):
            if stage == "release_boundary" and not self.interrupted:
                self.interrupted = True
                raise primary
            return super().clear()

    with VectorEnv(Config(num_envs=1, max_steps=3)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=workers, episode_format=episode_format
        )
        writer.append(env.observe(), env.step(np.zeros((1, 2))))
        if stage == "collect_boundary":
            writer._writes = InterruptedRows(writer._writes)
        else:
            writer.pending = InterruptedRows(writer.pending)
        retained = writer.buffered_bytes
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                writer.close()
            assert caught.value is primary
            assert not writer.closed
            with pytest.raises(RuntimeError, match="closing|closed|failed"):
                writer.append(env.observe(), env.step(np.zeros((1, 2))))
            assert writer.buffered_bytes == retained
            assert len(writer.pending[(0, 0)]) == 1
            with pytest.raises(RuntimeError, match="closing|closed|failed"):
                writer.__enter__()
        finally:
            writer.close()
        assert writer.closed and writer.buffered_bytes == 0
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["incomplete_episodes"] == [
            {"env_id": 0, "episode_id": 0, "steps": 1}
        ]
