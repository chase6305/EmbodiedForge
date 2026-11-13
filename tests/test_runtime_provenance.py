"""Source manifests cannot make validation read outside the snapshot tree."""

from pathlib import Path

import pytest

from embodiedforge.runtime.hashing import file_sha256
from embodiedforge.runtime.provenance import (
    snapshot_implementation,
    validate_snapshot,
)


@pytest.mark.parametrize(
    "relative",
    [
        "",
        ".",
        "..",
        "../file.py",
        "/tmp/file.py",
        "a/../file.py",
        "a//b",
        "a/./b",
        "a\\b",
        "C:/file.py",
        "C:file.py",
        1,
    ],
)
def test_invalid_manifest_path_is_rejected_before_hashing(
    tmp_path, relative, monkeypatch
):
    def no_read(path):
        pytest.fail(f"invalid manifest tried to hash {path}")

    monkeypatch.setattr(
        "embodiedforge.runtime.provenance.snapshot_file_hashes", no_read
    )
    with pytest.raises(ValueError, match="invalid relative path"):
        validate_snapshot(tmp_path, {"files": {relative: "ignored"}})


def test_linked_directory_cannot_substitute_snapshot_contents(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file = source / "module.py"
    file.write_text("VALUE = 1\n")
    project = tmp_path / "snapshot"
    project.mkdir()
    (project / "nested").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="Cached source differs"):
        validate_snapshot(project, {"files": {"nested/module.py": file_sha256(file)}})


def test_missing_source_does_not_publish_an_empty_snapshot(tmp_path):
    target = tmp_path / "snapshot"
    with pytest.raises(ValueError, match="not a directory"):
        snapshot_implementation(tmp_path / "missing", target)
    assert not target.exists()


def test_snapshot_contains_the_new_runtime_and_runs_without_launcher_modules(tmp_path):
    import os
    import subprocess
    import sys

    package = Path(__file__).resolve().parents[1] / "src/embodiedforge"
    destination = tmp_path / "embodiedforge"
    record = snapshot_implementation(package, destination)
    validate_snapshot(destination, record)
    assert "runtime/process.py" in record["files"]
    assert "runtime/artifacts.py" in record["files"]
    script = (
        "import sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "from embodiedforge.runtime import artifacts, process, provenance, environment; "
        "assert not {'embodiedforge.recipes', 'embodiedforge.h1', 'torch', 'mujoco', 'warp'} & sys.modules.keys(); "
        "artifacts.write_json('result.json', {'snapshot': True})"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result.json").is_file()


def test_snapshot_rechecks_earlier_sources_after_copy(tmp_path, monkeypatch):
    from embodiedforge.runtime import provenance

    source = tmp_path / "source"
    source.mkdir()
    first = source / "a.py"
    first.write_bytes(b"first")
    (source / "z.py").write_bytes(b"last")
    copy = provenance.copy_file_snapshot

    def mutate_after_copy(path, target):
        result = copy(path, target)
        if path.name == "z.py":
            first.write_bytes(b"other")
        return result

    monkeypatch.setattr(provenance, "copy_file_snapshot", mutate_after_copy)
    with pytest.raises(ValueError, match="Implementation source differs"):
        snapshot_implementation(source, tmp_path / "snapshot")


def test_snapshot_independently_checks_written_destination(tmp_path, monkeypatch):
    from embodiedforge.runtime import provenance

    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_bytes(b"original")
    copy = provenance.copy_file_snapshot

    def corrupt_after_copy(path, target):
        result = copy(path, target)
        target.write_bytes(b"corrupted")
        return result

    monkeypatch.setattr(provenance, "copy_file_snapshot", corrupt_after_copy)
    with pytest.raises(ValueError):
        snapshot_implementation(source, tmp_path / "snapshot")
