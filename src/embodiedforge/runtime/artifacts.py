"""Atomic JSON publication shared across workflows.

Publication uses a unique sibling file and an atomic rename. It does not promise
fsync durability or serialize concurrent writers; the last completed writer wins.
"""

import json
import logging
import os
from pathlib import Path
from uuid import uuid4

LOGGER = logging.getLogger(__name__)


def discard_partial(path: str | Path, *, label="artifact") -> None:
    """Best-effort cleanup after a failure; never replace its active exception.

    Only call for a temporary file owned by the failed operation. This does not
    acquire ownership, coordinate writers, or retry an interrupted deletion.
    """
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except BaseException:
        LOGGER.warning("Could not remove partial %s %s", label, path, exc_info=True)


def write_json(path: str | Path, value) -> None:
    """Publish finite JSON without exposing partial output or reusing a temp file.

    The caller owns directory creation, schema validation and writer coordination.
    Existing output survives serialization/write/rename errors. Cleanup errors are
    logged without replacing the original failure.
    """
    # Unique path components are retained by CPython 3.12's pathlib interning.
    # Keep high-volume episode names as filesystem strings throughout this path.
    path = os.fspath(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    parent, name = os.path.split(path)
    temporary = os.path.join(parent, f"{name}.{uuid4().hex}.tmp")
    created = False
    try:
        with open(temporary, "x", encoding="utf-8") as stream:
            created = True
            stream.write(payload)
        os.replace(temporary, path)
    except BaseException:
        if created:
            discard_partial(temporary, label="JSON")
        raise
