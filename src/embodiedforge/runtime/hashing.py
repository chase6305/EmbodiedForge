"""File identity checks that do not retain the complete artifact in memory."""

import hashlib
import os
import stat
from pathlib import Path
from typing import BinaryIO


def file_identity(info: os.stat_result) -> tuple:
    """Metadata observable across one regular-file read; this is not a lock."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _write_block(stream: BinaryIO, block: bytes) -> None:
    remaining = memoryview(block)
    while remaining:
        written = stream.write(remaining)
        if (
            not isinstance(written, int)
            or isinstance(written, bool)
            or written <= 0
            or written > len(remaining)
        ):
            raise OSError("Snapshot copy did not make valid write progress")
        remaining = remaining[written:]


def snapshot_file_hashes(
    path: str | Path,
    git_format: str | None = None,
    *,
    copy_to: BinaryIO | None = None,
) -> tuple[str, str | None, os.stat_result]:
    """Hash one regular opened snapshot, optionally copying to a borrowed sink.

    Read at most the initial size plus one probe byte. Where supported, reject
    leaf symlink replacement and avoid blocking on a substituted FIFO. Metadata
    checks cannot detect changes invisible to those fields; keep inputs immutable.
    The caller owns ``copy_to`` and any partially written output after failure.
    """
    path = Path(path)
    expected = path.lstat()
    if not stat.S_ISREG(expected.st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    flags = os.O_RDONLY
    for option in ("O_NONBLOCK", "O_NOFOLLOW", "O_CLOEXEC"):
        flags |= getattr(os, option, 0)
    descriptor = os.open(path, flags)
    try:
        stream = os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Expected a regular file: {path}")
        if file_identity(expected) != file_identity(before):
            raise ValueError(f"File changed while opening: {path}")
        digest = hashlib.sha256()
        blob = hashlib.new(git_format, usedforsecurity=False) if git_format else None
        if blob is not None:
            blob.update(b"blob " + str(before.st_size).encode("ascii") + b"\0")
        remaining = before.st_size
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"File changed while hashing: {path}")
            digest.update(block)
            if blob is not None:
                blob.update(block)
            if copy_to is not None:
                _write_block(copy_to, block)
            remaining -= len(block)
        if stream.read(1):
            raise ValueError(f"File changed while hashing: {path}")
        after = os.fstat(stream.fileno())
        if file_identity(before) != file_identity(after) or file_identity(
            after
        ) != file_identity(path.lstat()):
            raise ValueError(f"File changed while hashing: {path}")
    return digest.hexdigest(), blob.hexdigest() if blob is not None else None, after


def copy_file_snapshot(source: str | Path, destination: str | Path) -> str:
    """Create a new destination while hashing the same source read.

    An existing destination is never overwritten. As with a failed ordinary
    copy, the caller owns any partial output if reading, writing or validation
    fails. No source or destination locking is provided.
    """
    with open(os.fspath(destination), "xb") as stream:
        digest, _, _ = snapshot_file_hashes(source, copy_to=stream)
    return digest


def file_sha256(path: str | Path) -> str:
    """Hash file bytes in bounded blocks; callers own snapshot/locking semantics."""
    with open(os.fspath(path), "rb") as stream:
        return stream_sha256(stream)


def stream_sha256(stream: BinaryIO) -> str:
    """Hash from current position to EOF; borrow the stream without closing it."""
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()
