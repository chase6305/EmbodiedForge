"""Content identity for local ACT datasets, without loading the training SDK."""

import hashlib
import json
from pathlib import Path

_SCOPE = ["embodiedforge.json", "meta", "data", "images", "videos"]


def _files(root):
    paths = []

    def visit(path):
        if path.is_symlink():
            raise ValueError(f"Dataset fingerprint does not support symlinks: {path}")
        if path.is_dir():
            for child in sorted(path.iterdir()):
                visit(child)
        elif path.is_file():
            paths.append(path)
        else:
            raise ValueError(f"Dataset file is missing or not regular: {path}")

    for name in _SCOPE:
        path = root / name
        if name in ("images", "videos") and not path.exists() and not path.is_symlink():
            continue
        visit(path)
    return sorted(paths)


def fingerprint_dataset(root):
    """Hash managed data/metadata/media files; unrelated root files are excluded.

    This is a preflight identity check, not a snapshot or a lock against writes
    during training. Read in blocks so large data files do not fill memory.
    """
    root = Path(root).resolve()
    paths = _files(root)
    files = []
    for path in paths:
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Dataset changed while computing fingerprint: {path}")
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": after.st_size,
                "sha256": digest.hexdigest(),
            }
        )
    if paths != _files(root):
        raise ValueError("Dataset file list changed while computing fingerprint")
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": _SCOPE.copy(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def verify_dataset_fingerprint(expected, current):
    """Compare to the source run; old runs cannot retroactively prove identity."""
    if expected is None:
        return {"status": "unverified_legacy", "current_sha256": current["sha256"]}
    if (
        expected.get("schema_version") != 1
        or expected.get("algorithm") != "sha256"
        or expected.get("scope") != _SCOPE
    ):
        raise ValueError("Unsupported ACT dataset fingerprint")
    if expected != current:
        previous = {item["path"]: item for item in expected["files"]}
        present = {item["path"]: item for item in current["files"]}
        changed = sorted(
            name
            for name in previous.keys() | present.keys()
            if previous.get(name) != present.get(name)
        )
        raise ValueError(
            "Dataset content changed since the source training run: "
            + ", ".join(changed[:10])
            + (f" (+{len(changed) - 10} more)" if len(changed) > 10 else "")
        )
    return {"status": "verified", "sha256": current["sha256"]}
