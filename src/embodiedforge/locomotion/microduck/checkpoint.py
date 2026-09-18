"""Publish complete local checkpoints while retaining the previous saved model."""

from pathlib import Path
from uuid import uuid4


def atomic_save(save, path, infos=None):
    target = Path(path)
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    try:
        save(str(temporary), infos)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
