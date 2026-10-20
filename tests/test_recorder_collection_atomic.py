"""Completed-write accounting rolls back a single caller interruption."""

import inspect
import json
import sys

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.data import EpisodeRecorder, read_episodes


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, OSError])
@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("entry", ["close", "append"])
@pytest.mark.parametrize("boundary", ["count", "release"])
@pytest.mark.parametrize("interrupt_env", [0, 1])
def test_close_interrupted_after_completion_accounting_is_not_counted_twice(
    tmp_path, error_type, episode_format, entry, boundary, interrupt_env
):
    with VectorEnv(Config(num_envs=2, max_steps=1)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=2, episode_format=episode_format
        )
        writer.append(env.observe(), env.step(np.zeros((2, 2))))
        primary = error_type("interrupt after accounting")
        code = EpisodeRecorder._collect_writes.__code__
        lines, start = inspect.getsourcelines(EpisodeRecorder._collect_writes)
        target_text = (
            'self.manifest["completed_episodes"] = completed + 1'
            if boundary == "count"
            else "del self._writes[key]"
        )
        target = start + next(
            i for i, line in enumerate(lines) if line.strip() == target_text
        )
        injected = False

        def interrupt(frame, event, arg):
            nonlocal injected
            if (
                frame.f_code is code
                and event == "line"
                and frame.f_lineno == target
                and frame.f_locals["key"][0] == interrupt_env
                and not injected
            ):
                injected = True
                raise primary
            return interrupt

        previous = sys.gettrace()
        try:
            for future, _, _ in writer._writes.values():
                future.result(timeout=5)
            retained = writer.buffered_bytes
            first_size = writer._writes[(0, 0)][2]
            if entry == "append":
                observation = env.reset()
                result = env.step(np.zeros((2, 2)))
            sys.settrace(interrupt)
            with pytest.raises(error_type) as caught:
                if entry == "close":
                    writer.close()
                else:
                    writer.append(observation, result)
            assert caught.value is primary
            sys.settrace(previous)
            if entry == "append":
                assert writer.buffered_bytes == retained - interrupt_env * first_size
                assert writer.manifest["completed_episodes"] == interrupt_env
                assert len(writer._writes) == 2 - interrupt_env
                writer.append(observation, result)
        finally:
            sys.settrace(previous)
            writer.close()
        assert injected
        assert writer.closed and not writer._writes
        assert writer.buffered_bytes == 0
        manifest = json.loads((writer.root / "run.json").read_text())
        expected = 2 if entry == "close" else 4
        assert manifest["completed_episodes"] == expected
        assert len(list(read_episodes(writer.root))) == expected


@pytest.mark.parametrize("episode_format", ["transitions", "compact"])
@pytest.mark.parametrize("interruptions", [1, 2])
def test_interruption_after_handle_removal_restores_retry_state(
    tmp_path, episode_format, interruptions
):
    primary, secondary = KeyboardInterrupt("first removal"), OSError("second removal")
    faults = [primary, secondary][:interruptions]

    class InterruptedWrites(dict):
        def __delitem__(self, key):
            super().__delitem__(key)
            if faults:
                raise faults.pop(0)

    with VectorEnv(Config(num_envs=2, max_steps=1)) as env:
        writer = EpisodeRecorder(
            tmp_path / "run", env, write_workers=2, episode_format=episode_format
        )
        writer.append(env.observe(), env.step(np.zeros((2, 2))))
        writer._writes = InterruptedWrites(writer._writes)
        retained = writer.buffered_bytes
        try:
            with pytest.raises(KeyboardInterrupt) as caught:
                writer.close()
            assert caught.value is primary
            if interruptions == 2:
                assert caught.value.__cause__ is secondary
                assert not writer.closed
                assert len(writer._writes) == 2
                assert writer.buffered_bytes == retained
                assert writer.manifest["completed_episodes"] == 0
                assert writer.failure is primary
            else:
                assert writer.closed and writer.failure is None
        finally:
            writer.close()
        assert writer.closed and not writer._writes
        assert writer.buffered_bytes == 0
        manifest = json.loads((writer.root / "run.json").read_text())
        assert manifest["completed_episodes"] == 2
        assert manifest["status"] == ("closed" if interruptions == 1 else "failed")
        assert len(list(read_episodes(writer.root))) == 2
