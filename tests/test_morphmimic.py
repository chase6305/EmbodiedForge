"""Boundary and provenance checks for the isolated author-code workflow."""

import io
import json
import os
import subprocess
import sys
import tarfile

import pytest

from embodiedforge import morphmimic
from embodiedforge.runtime.hashing import file_sha256


def test_help_does_not_import_learning_or_simulator_sdks():
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "from embodiedforge.morphmimic import parser; parser().format_help(); "
            "assert not {'torch', 'numpy', 'mujoco', 'mjlab'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "--robots", "g1", "g1"],
        ["prepare", "--seconds", "nan"],
        ["train", "--iterations", "500"],
        ["train", "--batch", "0"],
    ],
)
def test_invalid_request_creates_no_output(tmp_path, arguments):
    with pytest.raises(SystemExit) as error:
        morphmimic.main([*arguments, "--workspace", str(tmp_path)])
    assert error.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_source_verification_rejects_tampering_and_archive_replacement(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    content = b"pinned author algorithm\n"
    (source / "algorithm.py").write_bytes(content)
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("commit/algorithm.py")
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(morphmimic, "ARCHIVE_SHA256", file_sha256(archive))
    assert morphmimic.verify_source(source, archive)["verified_files"] == 1
    (source / "algorithm.py").write_bytes(b"modified algorithm")
    with pytest.raises(ValueError, match="source changed"):
        morphmimic.verify_source(source, archive)
    archive.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="archive hash"):
        morphmimic.verify_source(source, archive)


def test_source_verification_rejects_parent_traversal(tmp_path, monkeypatch):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("commit/../escape")
        member.size = 0
        bundle.addfile(member, io.BytesIO(b""))
    monkeypatch.setattr(morphmimic, "ARCHIVE_SHA256", file_sha256(archive))
    with pytest.raises(ValueError, match="Unexpected source archive entry"):
        morphmimic.verify_source(tmp_path, archive)


def test_source_verification_pins_the_archive_it_actually_parses(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "algorithm.py").write_bytes(b"replacement source")
    archive = tmp_path / "source.tar"
    replacement = tmp_path / "replacement.tar"
    for path, content in (
        (archive, b"pinned source"),
        (replacement, b"replacement source"),
    ):
        with tarfile.open(path, "w") as bundle:
            member = tarfile.TarInfo("commit/algorithm.py")
            member.size = len(content)
            bundle.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(morphmimic, "ARCHIVE_SHA256", file_sha256(archive))
    file_hash, stream_hash = morphmimic.file_sha256, morphmimic.stream_sha256
    replaced = False

    def publish_once():
        nonlocal replaced
        if not replaced:
            replacement.replace(archive)
            replaced = True

    def file_then_publish(path):
        result = file_hash(path)
        publish_once()
        return result

    def stream_then_publish(stream):
        result = stream_hash(stream)
        publish_once()
        return result

    monkeypatch.setattr(morphmimic, "file_sha256", file_then_publish)
    monkeypatch.setattr(morphmimic, "stream_sha256", stream_then_publish)
    with pytest.raises(ValueError, match="source changed"):
        morphmimic.verify_source(source, archive)


@pytest.mark.parametrize("operation", ["verify", "setup"])
@pytest.mark.parametrize(
    "name,kind",
    [
        ("commit/../escape", tarfile.REGTYPE),
        ("commit/link", tarfile.SYMTYPE),
        ("commit/link", tarfile.LNKTYPE),
        ("commit/pipe", tarfile.FIFOTYPE),
    ],
)
def test_setup_and_verification_reject_the_same_archive_entries(
    tmp_path, monkeypatch, operation, name, kind
):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "algorithm.py"
        bundle.addfile(member, io.BytesIO(b""))
    monkeypatch.setattr(morphmimic, "ARCHIVE_SHA256", file_sha256(archive))
    monkeypatch.setattr(morphmimic.shutil, "which", lambda _: "uv")
    source = tmp_path / "source"
    with pytest.raises(ValueError, match="Unexpected source archive entry"):
        if operation == "verify":
            morphmimic.verify_source(source, archive)
        else:
            morphmimic.setup(source, tmp_path / "environment", archive)
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "environment").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires POSIX FIFOs")
def test_source_verification_rejects_a_fifo_without_waiting_for_a_writer(tmp_path):
    from pathlib import Path

    source = tmp_path / "source"
    source.mkdir()
    os.mkfifo(source / "algorithm.py")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("commit/algorithm.py")
        bundle.addfile(member, io.BytesIO(b""))
    script = (
        "import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
        "from embodiedforge import morphmimic; "
        "morphmimic.ARCHIVE_SHA256=morphmimic.file_sha256(sys.argv[3]); "
        "morphmimic.verify_source(Path(sys.argv[2]), Path(sys.argv[3]))"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            str(Path(morphmimic.__file__).resolve().parents[1]),
            str(source),
            str(archive),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert "Expected a regular file" in result.stderr


def test_worker_failure_is_recorded_with_observed_exit(tmp_path, monkeypatch):
    from embodiedforge.runtime.records import read_run

    source = tmp_path / ".cache/external/morphmimic-source"
    source.mkdir(parents=True)
    python = tmp_path / ".cache/morphmimic-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setattr(
        morphmimic, "verify_source", lambda *args: {"revision": "fixture"}
    )

    def fail(command, *, env, **kwargs):
        assert command[1] == "-I"
        assert "PYTHONPATH" not in env
        raise subprocess.CalledProcessError(-11, command)

    monkeypatch.setattr(morphmimic, "run_process", fail)
    output = tmp_path / "failed-run"
    with pytest.raises(subprocess.CalledProcessError):
        morphmimic.main(
            ["doctor", "--workspace", str(tmp_path), "--output", str(output)]
        )
    record = json.loads((output / "run.json").read_text())
    assert record["worker_exit_observed"] is True
    assert record["worker_returncode"] == -11
    assert read_run(output)["status"] == "failed"


def test_setup_download_preserves_other_partial_and_includes_export_dependencies(
    tmp_path, monkeypatch
):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        member = tarfile.TarInfo("commit/algorithm.py")
        member.size = 4
        bundle.addfile(member, io.BytesIO(b"pass"))
    import hashlib

    monkeypatch.setattr(
        morphmimic, "ARCHIVE_SHA256", hashlib.sha256(buffer.getvalue()).hexdigest()
    )
    monkeypatch.setattr(morphmimic.shutil, "which", lambda _: "uv")
    monkeypatch.setattr(
        morphmimic.urllib.request,
        "urlopen",
        lambda *a, **kw: io.BytesIO(buffer.getvalue()),
    )
    archive = tmp_path / "source.tar.gz"
    foreign = archive.with_suffix(".partial")
    foreign.write_bytes(b"another download owns this")
    environment = tmp_path / "sdk"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if "export" in command:
            (tmp_path / "pylock.runtime.toml").write_text("# frozen fixture\n")

    monkeypatch.setattr(morphmimic.subprocess, "run", run)
    source = tmp_path / "upstream"
    morphmimic.setup(source, environment, archive)
    assert foreign.read_bytes() == b"another download owns this"
    assert (source / "algorithm.py").read_bytes() == b"pass"
    assert morphmimic.verify_source(source, archive)["verified_files"] == 1
    export = next(command for command in commands if "export" in command)
    assert export[export.index("--group") + 1] == "export"
    assert not list(tmp_path.glob("morphmimic-*"))


@pytest.mark.parametrize("feasibility", [None, {"feas_reject": "floating_feet"}])
def test_training_rejects_unchecked_or_floating_transferred_references(
    tmp_path, feasibility
):
    source = tmp_path / ".cache/external/morphmimic-source"
    source.mkdir(parents=True)
    python = tmp_path / ".cache/morphmimic-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    data = tmp_path / "data"
    data.mkdir()
    (data / "g1.yaml").write_text("root: unused\nclips: []\n")
    clip = {"robot": "g1"}
    if feasibility is not None:
        clip["feasibility"] = feasibility
    (data / "result.json").write_text(
        json.dumps({"dataset": "existing LAFAN1 G1", "clips": [clip]})
    )
    output = tmp_path / "rejected-training"
    with pytest.raises(SystemExit) as error:
        morphmimic.main(
            [
                "train",
                "--workspace",
                str(tmp_path),
                "--data",
                str(data),
                "--robots",
                "g1",
                "--output",
                str(output),
            ]
        )
    assert error.value.code == 2
    assert not output.exists()
