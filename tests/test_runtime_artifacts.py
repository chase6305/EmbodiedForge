"""Publication must preserve committed records and independent concurrent writers."""

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from embodiedforge.runtime import artifacts


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, object()])
def test_invalid_payload_preserves_existing_record(tmp_path, invalid):
    path = tmp_path / "run.json"
    artifacts.write_json(path, {"status": "running"})
    before = path.read_bytes()
    with pytest.raises((ValueError, TypeError)):
        artifacts.write_json(path, {"invalid": invalid})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_failed_rename_preserves_record_and_cleans_owned_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "run.json"
    artifacts.write_json(path, {"status": "running"})
    before = path.read_bytes()

    def fail_replace(source, target):
        raise OSError("publication failed")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failed"):
        artifacts.write_json(path, {"status": "complete"})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "cleanup_error",
    [KeyboardInterrupt(), SystemExit(3), PermissionError("cleanup denied")],
)
def test_cleanup_cannot_replace_the_publication_failure(
    tmp_path, monkeypatch, cleanup_error, caplog
):
    path = tmp_path / "run.json"
    artifacts.write_json(path, {"status": "running"})
    before = path.read_bytes()
    primary = OSError("publication failed")

    def fail_replace(*args):
        raise primary

    def fail_cleanup(*args):
        raise cleanup_error

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    monkeypatch.setattr(artifacts.os, "unlink", fail_cleanup)
    with pytest.raises(OSError) as caught:
        artifacts.write_json(path, {"status": "complete"})
    assert caught.value is primary
    assert path.read_bytes() == before
    assert "Could not remove partial JSON" in caplog.text


def test_colliding_temporary_is_never_overwritten_or_removed(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    collision = tmp_path / "run.json.existing.tmp"
    collision.write_text("another writer")
    monkeypatch.setattr(artifacts, "uuid4", lambda: SimpleNamespace(hex="existing"))
    with pytest.raises(FileExistsError):
        artifacts.write_json(path, {"status": "complete"})
    assert collision.read_text() == "another writer"
    assert not path.exists()


def test_concurrent_writers_publish_only_complete_independent_payloads(
    tmp_path, monkeypatch
):
    path = tmp_path / "run.json"
    barrier = threading.Barrier(4)
    replace = artifacts.os.replace
    published = []
    lock = threading.Lock()

    def publish(source, target):
        # All four serialized records must coexist before any one is published.
        barrier.wait(timeout=5)
        with lock:
            result = replace(source, target)
            published.append(json.loads(path.read_text()))
            return result

    monkeypatch.setattr(artifacts.os, "replace", publish)
    payloads = [{"writer": i, "payload": "记录" * (1000 + i)} for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda value: artifacts.write_json(path, value), payloads))
    assert sorted(published, key=lambda value: value["writer"]) == payloads
    assert json.loads(path.read_text()) in payloads
    assert list(tmp_path.iterdir()) == [path]


def test_unique_artifact_names_do_not_accumulate_python_allocations(tmp_path):
    import gc
    import os
    import tracemalloc

    def publish(index):
        # Keep generated names as strings: the writer owns filesystem conversion.
        path = os.path.join(tmp_path, f"artifact-memory-unique-{index:08d}.json")
        artifacts.write_json(path, {"index": index})
        os.unlink(path)

    for index in range(64):
        publish(index)
    gc.collect()
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for index in range(64, 4064):
            publish(index)
        gc.collect()
        retained = tracemalloc.get_traced_memory()[0] - before
        assert retained < 256 * 1024, retained
    finally:
        if started:
            tracemalloc.stop()
    assert not list(tmp_path.iterdir())
