"""Background publication retains ownership, bounded backpressure and failures."""

import json
import sys
import threading

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


def transition_bytes(observation, result):
    return sum(
        value.nbytes
        for value in (
            *observation.values(),
            *result.observation.values(),
            result.info["executed_action"],
            result.reward,
            result.terminated,
            result.truncated,
        )
    )


@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("workers", [1, 2])
def test_async_matches_sync_and_close_drains_all_episodes(
    tmp_path, episode_format, workers
):
    roots = [tmp_path / "sync", tmp_path / "async"]
    for root, count in zip(roots, (0, workers), strict=True):
        with VectorEnv(
            Config(
                num_envs=3,
                max_steps=5,
                render="raster",
                channels=("rgb", "depth"),
                image_size=16,
            )
        ) as env:
            with EpisodeRecorder(
                root, env, episode_format=episode_format, write_workers=count
            ) as writer:
                for _ in range(3):
                    for _ in range(5):
                        observation = env.observe()
                        writer.append(observation, env.step(np.zeros((3, 2))))
                    env.reset()
            assert writer.buffered_bytes == 0
            assert not writer._writes
            assert writer.closed
        manifest = json.loads((root / "run.json").read_text())
        assert manifest["status"] == "closed"
        assert manifest["completed_episodes"] == 9
        assert manifest["write_workers"] == count
    for expected, actual in zip(
        read_episodes(roots[0]), read_episodes(roots[1]), strict=True
    ):
        assert expected.keys() == actual.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])


def test_queued_payloads_own_arrays_and_exert_budget_backpressure(
    tmp_path, monkeypatch
):
    gate, entered, waiting, appended = (threading.Event() for _ in range(4))
    errors = []
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        observation = env.observe()
        result = env.step(np.zeros((1, 2)))
        expected = observation["proprio"].copy()
        budget = transition_bytes(observation, result)
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=1, max_buffer_bytes=budget
        )
        write = writer._write_episode

        def blocked_write(key, rows):
            entered.set()
            assert gate.wait(5)
            write(key, rows)

        monkeypatch.setattr(writer, "_write_episode", blocked_write)
        collect = writer._collect_writes

        def collect_writes(*, wait_for=None):
            if wait_for is not None:
                waiting.set()
            return collect(wait_for=wait_for)

        monkeypatch.setattr(writer, "_collect_writes", collect_writes)
        thread = None
        try:
            writer.append(observation, result)
            assert entered.wait(5)
            assert writer.buffered_bytes == budget
            assert not writer.pending
            assert len(writer._writes) == 1
            observation["proprio"][:] = 99
            result.observation["proprio"][:] = 99
            env.reset()
            observation = env.observe()
            result = env.step(np.zeros((1, 2)))

            def append():
                try:
                    writer.append(observation, result)
                except BaseException as error:
                    errors.append(error)
                finally:
                    appended.set()

            thread = threading.Thread(target=append)
            thread.start()
            assert waiting.wait(5)
            assert not appended.is_set()
            assert writer.buffered_bytes == budget
            gate.set()
            thread.join(5)
            assert not thread.is_alive()
            assert not errors
            assert writer.buffered_bytes <= budget
        finally:
            gate.set()
            if thread is not None:
                thread.join(5)
            writer.close()
        episodes = list(read_episodes(writer.root))
        assert len(episodes) == 2
        np.testing.assert_array_equal(episodes[0]["obs/proprio"], expected)


@pytest.mark.parametrize("report_at", ["append", "close"])
def test_async_storage_failure_is_reported_and_never_published(
    tmp_path, monkeypatch, report_at
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)

        def fail(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(np, "savez_compressed", fail)
        observation = env.observe()
        result = env.step(np.zeros((1, 2)))
        writer.append(observation, result)
        if report_at == "append":
            future = next(iter(writer._writes.values()))[0]
            with pytest.raises(OSError, match="disk full"):
                future.result(timeout=5)
            with pytest.raises(OSError, match="disk full"):
                writer.append(observation, result)
            with pytest.raises(RuntimeError, match="Recorder failed"):
                writer.append(observation, result)
            writer.close()
        else:
            with pytest.raises(OSError, match="disk full"):
                writer.close()
        writer.close()
        assert writer.closed
        assert isinstance(writer.failure, OSError)
        assert not writer._writes
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["completed_episodes"] == 0
        assert manifest["incomplete_episodes"] == [
            {"env_id": 0, "episode_id": 0, "steps": 1}
        ]
        assert not list(read_episodes(writer.root))


def test_close_counts_successful_writes_even_when_another_worker_fails(
    tmp_path, monkeypatch
):
    with VectorEnv(Config(num_envs=2, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=2)
        write = writer._write_episode

        def mixed(key, rows):
            if key[0] == 0:
                raise OSError("first episode failed")
            write(key, rows)

        monkeypatch.setattr(writer, "_write_episode", mixed)
        writer.append(env.observe(), env.step(np.zeros((2, 2))))
        with pytest.raises(OSError, match="first episode failed"):
            writer.close()
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["status"] == "failed"
        assert manifest["completed_episodes"] == 1
        assert manifest["incomplete_episodes"][0]["env_id"] == 0
        assert len(list(read_episodes(writer.root))) == 1


def test_background_close_failure_does_not_mask_original_exception(
    tmp_path, monkeypatch
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)

        def fail(*args):
            raise OSError("background failure")

        monkeypatch.setattr(writer, "_write_episode", fail)
        with pytest.raises(KeyError, match="original"):
            with writer:
                writer.append(env.observe(), env.step(np.zeros((1, 2))))
                raise KeyError("original")
        assert isinstance(writer.failure, KeyError)
        assert isinstance(writer.close_error, OSError)
        assert writer.closed


@pytest.mark.parametrize("cleanup_error", [OSError, KeyboardInterrupt])
def test_worker_cleanup_never_masks_primary_exception(
    tmp_path, monkeypatch, cleanup_error
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)

        def fail(*args):
            raise cleanup_error("secondary worker error")

        monkeypatch.setattr(writer, "_write_episode", fail)
        with pytest.raises(KeyError, match="primary error"):
            with writer:
                writer.append(env.observe(), env.step(np.zeros((1, 2))))
                raise KeyError("primary error")
        assert isinstance(writer.failure, KeyError)
        assert isinstance(writer.close_error, cleanup_error)
        assert writer.closed


@pytest.mark.parametrize("workers", [0, 2])
def test_manifest_close_failure_disables_append_but_allows_close_retry(
    tmp_path, monkeypatch, workers
):
    from embodiedforge import data

    with VectorEnv(Config(num_envs=1, max_steps=3)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=workers)
        observation = env.observe()
        result = env.step(np.zeros((1, 2)))
        writer.append(observation, result)
        before = writer.buffered_bytes
        original = data._json_atomic

        def fail(path, value):
            raise OSError("final manifest full")

        monkeypatch.setattr(data, "_json_atomic", fail)
        with pytest.raises(OSError, match="final manifest full"):
            writer.close()
        assert not writer.closed and writer.buffered_bytes == before
        with pytest.raises(RuntimeError, match="Recorder failed"):
            writer.append(observation, result)
        monkeypatch.setattr(data, "_json_atomic", original)
        writer.close()
        assert writer.closed and writer.buffered_bytes == 0
        assert json.loads((writer.root / "run.json").read_text())["status"] == "failed"


def test_worker_error_survives_a_secondary_final_manifest_error(tmp_path, monkeypatch):
    from embodiedforge import data

    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)
        primary, secondary = OSError("archive full"), OSError("manifest full")
        original = data._json_atomic

        def fail_archive(*args):
            raise primary

        def fail_manifest(*args):
            raise secondary

        monkeypatch.setattr(writer, "_write_episode", fail_archive)
        monkeypatch.setattr(data, "_json_atomic", fail_manifest)
        writer.append(env.observe(), env.step(np.zeros((1, 2))))
        with pytest.raises(OSError, match="archive full") as error:
            writer.close()
        assert error.value is writer.failure is primary
        assert error.value.__cause__ is writer.close_error is secondary
        monkeypatch.setattr(data, "_json_atomic", original)
        writer.close()
        assert writer.closed


def test_caller_signal_is_not_misclassified_as_a_completed_worker_failure(
    tmp_path, monkeypatch
):
    with VectorEnv(Config(num_envs=1, max_steps=1)) as env:
        writer = EpisodeRecorder(tmp_path / "run", env, write_workers=1)
        observation = env.observe()
        result = env.step(np.zeros((1, 2)))
        writer.append(observation, result)
        future = next(iter(writer._writes.values()))[0]
        original = future.result
        original(timeout=5)
        interrupted = False

        def signal(*args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt("caller interrupted")
            return original(*args, **kwargs)

        monkeypatch.setattr(future, "result", signal)
        try:
            with pytest.raises(KeyboardInterrupt, match="caller interrupted"):
                writer.append(observation, result)
            assert writer.failure is None
            assert len(writer._writes) == 1
            assert not writer.pending
        finally:
            writer.close()
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["status"] == "closed" and manifest["completed_episodes"] == 1
        assert len(list(read_episodes(writer.root))) == 1


@pytest.mark.parametrize("workers", [-1, True, 1.5, None])
def test_invalid_worker_count_does_not_create_output(tmp_path, workers):
    with VectorEnv(Config(num_envs=1)) as env:
        with pytest.raises(ValueError, match="write_workers"):
            EpisodeRecorder(tmp_path / "invalid", env, write_workers=workers)
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        ["rollout", "--record-workers", "-1"],
        ["rollout", "--record-workers", "1"],
        ["plan", "--record-workers", "1", "--output", "unused"],
    ],
)
def test_cli_rejects_invalid_background_options_before_building_env(
    monkeypatch, arguments
):
    from embodiedforge import cli

    def unexpected_env(*args):
        pytest.fail("Invalid CLI input must not construct an environment")

    monkeypatch.setattr(cli, "VectorEnv", unexpected_env)
    monkeypatch.setattr(sys, "argv", ["embodiedforge", *arguments])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


def test_cli_compact_background_recording_is_closed_and_readable(
    tmp_path, monkeypatch, capsys
):
    from embodiedforge import cli

    config = tmp_path / "config.json"
    config.write_text(json.dumps(Config(num_envs=2, max_steps=2).to_dict()))
    root = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "embodiedforge",
            "rollout",
            "--config",
            str(config),
            "--steps",
            "6",
            "--record-format",
            "compact",
            "--record-workers",
            "2",
            "--output",
            str(root),
        ],
    )
    cli.main()
    json.loads(capsys.readouterr().out)
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "closed"
    assert manifest["write_workers"] == 2
    assert manifest["episode_format"] == "compact"
    episodes = list(read_episodes(root))
    assert len(episodes) == manifest["completed_episodes"] == 6
    assert sum(len(episode["action"]) for episode in episodes) == 12
