"""Publish complete local checkpoints while retaining the previous saved model."""

import logging
from pathlib import Path
from uuid import uuid4


def atomic_save(save, path, infos=None):
    target = Path(path)
    temporary = target.with_name(f"{target.name}.{uuid4().hex}.tmp")
    try:
        save(str(temporary), infos)
        temporary.replace(target)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "Could not remove partial checkpoint %s: %s", temporary, exc
            )
        raise
