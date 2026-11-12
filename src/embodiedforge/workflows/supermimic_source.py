"""Pinned source and dependency metadata shared by SuperMimic entry points.

Keep these checks independent of the preflight CLI so SDK workers and reference
adapters can validate their inputs without importing a host command launcher.
"""

import os
import re
import subprocess
from pathlib import Path

from embodiedforge.runtime.hashing import file_identity as _file_identity
from embodiedforge.runtime.hashing import snapshot_file_hashes as _file_hashes


def source_identity(root, expected, files=()):
    if root is None:
        raise ValueError("A local checkout path is required")
    root = root.expanduser().resolve(strict=True)
    # A caller's Git environment must not redirect inspection to another tree.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
            stderr=subprocess.PIPE,
            text=True,
            errors="surrogateescape",
            timeout=5,
            env=environment,
        ).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Source path must be the checkout root")
    revision = git("rev-parse", "HEAD")
    if revision != expected:
        raise ValueError(f"Expected revision {expected}, found {revision}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked source changes prevent a reproducible preflight")
    # Git status trusts assume-unchanged and skip-worktree flags. Verify present
    # flagged files directly against committed blobs, including sparse checkouts.
    hidden = []
    for entry in git("ls-files", "-v", "-z").split("\0"):
        if entry and (entry[0].islower() or entry[0] == "S"):
            relative = entry[2:]
            if (
                entry[0].upper() != "S"
                or (root / relative).exists()
                or (root / relative).is_symlink()
            ):
                hidden.append(relative)

    paths = list(dict.fromkeys([*(os.fspath(path) for path in files), *hidden]))
    # Bound argument bytes as well as path count, and use literal pathspecs so
    # brackets, newlines and leading dashes retain their filename meaning.
    objects = {}
    batch = []
    batch_bytes = 0

    def load_objects():
        if not batch:
            return
        for entry in git("ls-tree", "-r", "-z", revision, "--", *batch).split("\0"):
            if entry:
                header, _, name = entry.partition("\t")
                mode, kind, object_id = header.split()
                if kind == "blob" and mode in ("100644", "100755"):
                    objects[name] = object_id

    for relative in paths:
        if (
            not Path(relative).parts
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"Missing regular source file: {relative}")
        argument = ":(literal)" + relative
        size = len(os.fsencode(argument)) + 1
        if size > 65536:
            raise ValueError("Source path exceeds Git argument budget")
        if batch and (len(batch) >= 128 or batch_bytes + size > 65536):
            load_objects()
            batch.clear()
            batch_bytes = 0
        batch.append(argument)
        batch_bytes += size
    load_objects()
    git_format = {40: "sha1", 64: "sha256"}.get(len(revision))
    if git_format is None:
        raise ValueError("Unsupported Git object format")
    hashes = {}
    for relative in paths:
        path = root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(f"Missing regular source file: {relative}")
        before = path.stat()
        actual, object_id, opened = _file_hashes(path, git_format)
        committed = objects.get(relative)
        after = path.stat()
        if _file_identity(before) != _file_identity(after) or _file_identity(
            opened
        ) != _file_identity(after):
            raise ValueError(f"Source changed during inspection: {relative}")
        if committed != object_id:
            raise ValueError(f"Source file differs from pinned commit: {relative}")
        hashes[relative] = actual
    if git("rev-parse", "HEAD") != revision or git(
        "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("Source changed during inspection")
    return {"root": str(root), "revision": revision, "files": hashes}


def matches_version(value, constraint):
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+[A-Za-z0-9.]+)?", value)
    if match is None:
        return not constraint  # Unconstrained packages need only be installed.
    version = tuple(int(part or 0) for part in match.groups())
    return (
        "minimum" not in constraint or version >= tuple(constraint["minimum"])
    ) and ("maximum" not in constraint or version < tuple(constraint["maximum"]))
