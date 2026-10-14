"""File identity checks that do not retain the complete artifact in memory."""

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Hash file bytes in bounded blocks; callers own snapshot/locking semantics."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
