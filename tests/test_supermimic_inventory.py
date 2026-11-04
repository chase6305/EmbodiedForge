"""Pinned sources and bounded motion discovery use actual filesystem evidence."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from embodiedforge import supermimic as sm


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "source"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    git("init", "-q")
    (root / "required.py").write_text("original source\n")
    (root / "other.py").write_text("original helper\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return root, git("rev-parse", "HEAD"), git


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
@pytest.mark.parametrize("name", ["required.py", "other.py"])
def test_hidden_tracked_changes_cannot_pass_source_preflight(checkout, flag, name):
    root, revision, git = checkout
    git("update-index", flag, name)
    (root / name).write_text("changed locally\n")
    assert not git("status", "--porcelain", "--untracked-files=no")
    with pytest.raises(ValueError, match="differs from pinned commit"):
        sm.source_identity(root, revision, ["required.py"])
    # Inspection must not alter the caller's index flags or source.
    assert not git("status", "--porcelain", "--untracked-files=no")
    assert (root / name).read_text() == "changed locally\n"


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_clean_hidden_files_are_verified_without_mutating_index(checkout, flag):
    root, revision, git = checkout
    git("update-index", flag, "other.py")
    before = git("ls-files", "-v")
    result = sm.source_identity(root, revision, ["required.py"])
    assert (
        result["files"]["other.py"]
        == hashlib.sha256((root / "other.py").read_bytes()).hexdigest()
    )
    assert git("ls-files", "-v") == before


def test_missing_sparse_file_can_remain_outside_required_source_set(checkout):
    root, revision, git = checkout
    git("update-index", "--skip-worktree", "other.py")
    (root / "other.py").unlink()
    assert sm.source_identity(root, revision, ["required.py"])["revision"] == revision
    with pytest.raises(ValueError, match="Missing regular source file"):
        sm.source_identity(root, revision, ["other.py"])


def test_combined_sparse_and_assume_flags_allow_missing_optional_file(checkout):
    root, revision, git = checkout
    git("update-index", "--skip-worktree", "other.py")
    git("update-index", "--assume-unchanged", "other.py")
    (root / "other.py").unlink()
    assert sm.source_identity(root, revision, ["required.py"])["revision"] == revision
    with pytest.raises(ValueError, match="Missing regular source file"):
        sm.source_identity(root, revision, ["other.py"])


@pytest.mark.parametrize("assume_unchanged", [False, True])
def test_dangling_sparse_symlink_is_present_and_rejected(
    checkout, tmp_path, assume_unchanged
):
    root, revision, git = checkout
    git("update-index", "--skip-worktree", "other.py")
    if assume_unchanged:
        git("update-index", "--assume-unchanged", "other.py")
    (root / "other.py").unlink()
    (root / "other.py").symlink_to(tmp_path / "absent-target")
    assert (root / "other.py").is_symlink() and not (root / "other.py").exists()
    assert not git("status", "--porcelain", "--untracked-files=no")
    before = git("ls-files", "-v")
    with pytest.raises(ValueError, match="Missing regular source file"):
        sm.source_identity(root, revision, ["required.py"])
    assert git("ls-files", "-v") == before
    assert not (tmp_path / "absent-target").exists()


def commit_fixture(git):
    git("add", "--all")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture files",
    )
    return git("rev-parse", "HEAD")


def test_many_files_use_batched_git_queries(checkout, monkeypatch):
    root, _, git = checkout
    names = [f"file-{i:03}.py" for i in range(256)]
    for name in names:
        (root / name).write_text(name)
    revision = commit_fixture(git)
    original = subprocess.check_output
    commands = []

    def counting(*args, **kwargs):
        commands.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", counting)
    result = sm.source_identity(root, revision, names)
    assert result["files"] == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in names
    }
    assert len(commands) <= 10, "Git process count must not grow twice per file"


def test_batch_argument_bytes_are_bounded(checkout, monkeypatch):
    root, _, git = checkout
    prefix = "/".join(["a" * 240, "b" * 240])
    (root / prefix).mkdir(parents=True)
    names = [prefix + f"/file-{i:03}-" + "c" * 50 for i in range(128)]
    for name in names:
        (root / name).write_text("source")
    revision = commit_fixture(git)
    original = subprocess.check_output
    batches = []

    def checking(command, **kwargs):
        if "ls-tree" in command:
            paths = command[command.index("--") + 1 :]
            batches.append(paths)
            assert sum(len(os.fsencode(p)) + 1 for p in paths) <= 65536
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", checking)
    assert len(sm.source_identity(root, revision, names)["files"]) == 128
    assert len(batches) == 2


def test_literal_and_non_ascii_git_paths_keep_exact_names(checkout):
    root, _, git = checkout
    names = ["a[1].py", "-leading.py", "colon:name.py", "tab\tline\n.py", "路径.py"]
    for name in names:
        (root / name).write_bytes(name.encode())
    revision = commit_fixture(git)
    result = sm.source_identity(root, revision, names)
    assert result["files"] == {
        name: hashlib.sha256(name.encode()).hexdigest() for name in names
    }


def test_non_utf8_unix_filename_can_be_verified(checkout):
    root, _, git = checkout
    name = os.fsdecode(b"source-\xff.py")
    (root / name).write_bytes(b"source")
    revision = commit_fixture(git)
    result = sm.source_identity(root, revision, [name])
    assert result["files"][name] == hashlib.sha256(b"source").hexdigest()


@pytest.mark.parametrize("name", ["", ".", "../outside", "/absolute"])
def test_invalid_source_paths_are_rejected_before_tree_lookup(
    checkout, monkeypatch, name
):
    root, revision, git = checkout
    original = subprocess.check_output

    def checking(command, **kwargs):
        assert "ls-tree" not in command
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", checking)
    with pytest.raises(ValueError, match="Missing regular source file"):
        sm.source_identity(root, revision, [name])


def test_oversized_source_path_never_reaches_git_tree_query(checkout, monkeypatch):
    root, revision, git = checkout
    original = subprocess.check_output

    def checking(command, **kwargs):
        assert "ls-tree" not in command
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", checking)
    with pytest.raises(ValueError, match="argument budget"):
        sm.source_identity(root, revision, ["x" * 65536])


@pytest.mark.parametrize("size", [0, 17, 2 * 1024 * 1024 + 31])
def test_streamed_sha256_and_git_blob_hash_match_git(checkout, size):
    root, _, git = checkout
    path = root / "binary"
    payload = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    path.write_bytes(payload)
    digest, blob, info = sm._file_hashes(path, "sha1")
    assert digest == hashlib.sha256(payload).hexdigest()
    assert blob == git("hash-object", "--no-filters", "--", "binary")
    assert info.st_size == size


def test_sha256_git_repository_is_supported(tmp_path):
    root = tmp_path / "sha256"
    root.mkdir()
    initialized = subprocess.run(
        ["git", "init", "-q", "--object-format=sha256", str(root)], capture_output=True
    )
    if initialized.returncode:
        pytest.skip("Local Git does not support SHA256 repositories")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    (root / "source.py").write_text("sha256 repository\n")
    revision = commit_fixture(git)
    assert len(revision) == 64
    result = sm.source_identity(root, revision, ["source.py"])
    assert (
        result["files"]["source.py"]
        == hashlib.sha256((root / "source.py").read_bytes()).hexdigest()
    )


def test_git_environment_cannot_redirect_checkout_inspection(
    checkout, tmp_path, monkeypatch
):
    root, revision, git = checkout
    fake = tmp_path / "unrelated.git"
    monkeypatch.setenv("GIT_DIR", str(fake))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "unrelated-index"))
    assert sm.source_identity(root, revision, ["required.py"])["revision"] == revision
    assert not fake.exists() and not (tmp_path / "unrelated-index").exists()


def test_nested_checkout_path_is_not_the_framework_root(checkout):
    root, revision, git = checkout
    nested = root / "nested"
    nested.mkdir()
    with pytest.raises(ValueError, match="checkout root"):
        sm.source_identity(nested, revision)


@pytest.mark.parametrize("change", ["rewrite", "replace"])
def test_open_hash_snapshot_detects_changes_and_closes_descriptor(
    tmp_path, monkeypatch, change
):
    path = tmp_path / "payload"
    path.write_bytes(b"a" * (2 * 1024 * 1024))
    original = os.fdopen
    streams = []

    class ChangingReader:
        changed = False

        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            block = self.stream.read(count)
            if block and not self.changed:
                self.changed = True
                previous = path.stat()
                time.sleep(0.02)
                if change == "replace":
                    replacement = tmp_path / "replacement"
                    replacement.write_bytes(b"b" * previous.st_size)
                    os.replace(replacement, path)
                else:
                    path.write_bytes(b"b" * previous.st_size)
                os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
            return block

    def opening(descriptor, mode="r", *args, **kwargs):
        stream = original(descriptor, mode, *args, **kwargs)
        streams.append(stream)
        return ChangingReader(stream)

    monkeypatch.setattr(os, "fdopen", opening)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(ValueError, match="File changed while hashing"):
        sm._file_hashes(path, "sha1")
    assert streams and all(stream.closed for stream in streams)
    assert len(os.listdir("/proc/self/fd")) == before


def test_source_change_during_hashing_is_rejected(checkout, monkeypatch):
    root, revision, git = checkout
    git("update-index", "--assume-unchanged", "required.py")
    original = sm._file_hashes

    def changing_hash(path, *args):
        digest = original(path, *args)
        previous = path.stat()
        time.sleep(0.02)
        path.write_text("changed source!\n")
        os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        return digest

    monkeypatch.setattr(sm, "_file_hashes", changing_hash)
    with pytest.raises(ValueError, match="Source changed during inspection"):
        sm.source_identity(root, revision, ["required.py"])


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo"])
def test_hash_rejects_special_paths_before_opening(tmp_path, monkeypatch, kind):
    path = tmp_path / "input"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "target"
        target.write_bytes(b"must not read")
        path.symlink_to(target)
    else:
        os.mkfifo(path)

    def unexpected_open(*args, **kwargs):
        pytest.fail("Nonregular inputs must be rejected before opening")

    monkeypatch.setattr(os, "open", unexpected_open)
    with pytest.raises(ValueError, match="Expected a regular file"):
        sm._file_hashes(path)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "regular"])
def test_hash_open_race_is_bounded_and_does_not_read_replacement(tmp_path, kind):
    # Run in a separate process so a regression cannot hang the test suite on
    # a FIFO with no writer. The actual replacement happens inside os.open.
    path = tmp_path / "input"
    path.write_bytes(b"original")
    script = r"""
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from embodiedforge import supermimic as sm
path = Path(sys.argv[2]); kind = sys.argv[3]
replacement = path.with_name('replacement'); replacement.write_bytes(b'replaced')
original_open, original_fdopen = os.open, os.fdopen
original_path_open = Path.open
opened = []
swapped = False
def replace_input():
    global swapped
    if swapped: return
    swapped = True
    path.unlink()
    if kind == 'fifo': os.mkfifo(path)
    elif kind == 'symlink': path.symlink_to(replacement)
    else: os.replace(replacement, path)
def swapping(name, flags, *args, **kwargs):
    replace_input()
    return original_open(name, flags, *args, **kwargs)
def path_open(self, *args, **kwargs):
    if self == path: replace_input()
    return original_path_open(self, *args, **kwargs)
def wrapping(fd, *args, **kwargs):
    stream = original_fdopen(fd, *args, **kwargs)
    opened.append(stream)
    return stream
os.open, os.fdopen = swapping, wrapping
Path.open = path_open
try:
    sm._file_hashes(path)
except (OSError, ValueError):
    assert all(stream.closed for stream in opened)
else:
    raise AssertionError('Replacement was accepted')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            str(Path(sm.__file__).resolve().parents[1]),
            str(path),
            kind,
        ],
        capture_output=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_hash_closes_raw_descriptor_when_stream_construction_fails(
    tmp_path, monkeypatch
):
    path = tmp_path / "input"
    path.write_bytes(b"regular")
    before = len(os.listdir("/proc/self/fd"))

    def fail_stream(*args, **kwargs):
        raise RuntimeError("stream construction failed")

    monkeypatch.setattr(os, "fdopen", fail_stream)
    with pytest.raises(RuntimeError, match="stream construction failed"):
        sm._file_hashes(path)
    assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize("change", ["grow", "truncate"])
def test_hash_read_budget_is_bounded_by_opened_size(tmp_path, monkeypatch, change):
    path = tmp_path / "input"
    path.write_bytes(b"original bytes")
    original = os.fdopen
    readers = []

    class ChangingReader:
        calls = 0
        requested = 0

        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            self.calls += 1
            assert self.calls <= 3, "Hash chased a continually growing input"
            self.requested += count
            if change == "truncate":
                path.write_bytes(b"")
            block = self.stream.read(count)
            if change == "grow":
                with path.open("ab") as writer:
                    writer.write(b"more bytes")
            return block

    def opening(fd, *args, **kwargs):
        reader = ChangingReader(original(fd, *args, **kwargs))
        readers.append(reader)
        return reader

    monkeypatch.setattr(os, "fdopen", opening)
    with pytest.raises(ValueError, match="File changed while hashing"):
        sm._file_hashes(path)
    assert readers[0].requested <= len(b"original bytes") + 1
    assert readers[0].stream.closed


def test_motion_entry_budget_counts_nonmatching_files(tmp_path):
    for i in range(100):
        (tmp_path / f"unrelated-{i}.txt").touch()
    result = sm.inspect_motions(tmp_path, ".npz", entry_limit=17)
    assert result["entries_scanned"] == 17
    assert result["matched_files_at_least"] == 0
    assert result["enumeration_capped"] and not result["content_verified"]


def test_motion_match_budget_and_descriptor_cleanup(tmp_path):
    for i in range(20):
        (tmp_path / f"clip-{i}.npz").write_bytes(b"not an array")
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(50):
        result = sm.inspect_motions(tmp_path, ".npz", match_limit=3)
        assert result["matched_files_at_least"] == 3
        assert result["enumeration_capped"]
    assert len(os.listdir("/proc/self/fd")) == before


def test_motion_scan_does_not_follow_symlink_files_or_directory_loops(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "clip.npz").touch()
    motions = tmp_path / "motions"
    motions.mkdir()
    (motions / "loop").symlink_to(motions, target_is_directory=True)
    (motions / "elsewhere").symlink_to(outside, target_is_directory=True)
    (motions / "clip.npz").symlink_to(outside / "clip.npz")
    result = sm.inspect_motions(motions, ".npz")
    assert result["entries_scanned"] == 3
    assert result["matched_files_at_least"] == 0
    assert not result["enumeration_capped"]


def test_motion_depth_limit_is_explicit(tmp_path):
    nested = tmp_path / "one" / "two"
    nested.mkdir(parents=True)
    (nested / "clip.csv").touch()
    result = sm.inspect_motions(tmp_path, ".csv", depth_limit=2)
    assert result["depth_limited"] and result["enumeration_capped"]
    assert result["matched_files_at_least"] == 0
    result = sm.inspect_motions(tmp_path, ".csv", depth_limit=3)
    assert result["matched_files_at_least"] == 1
    assert not result["enumeration_capped"]


def test_motion_scan_closes_parent_descriptor_on_child_error(tmp_path, monkeypatch):
    (tmp_path / "child").mkdir()
    original = os.scandir

    def fail_child(path):
        if str(path).endswith("/child"):
            raise PermissionError("directory became unreadable")
        return original(path)

    monkeypatch.setattr(os, "scandir", fail_child)
    before = len(os.listdir("/proc/self/fd"))
    with pytest.raises(PermissionError):
        sm.inspect_motions(tmp_path, ".npz")
    assert len(os.listdir("/proc/self/fd")) == before


@pytest.mark.parametrize("path", [None, "missing"])
def test_missing_motion_root_is_an_empty_inventory(tmp_path, path):
    result = sm.inspect_motions(None if path is None else tmp_path / path, ".npz")
    assert result["entries_scanned"] == result["matched_files_at_least"] == 0
    assert not result["enumeration_capped"]
