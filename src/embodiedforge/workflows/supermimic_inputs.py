"""Private, verified NPZ inputs for independent SuperMimic runs; no SDK imports."""

import json
import zipfile
import zlib
from pathlib import Path

from embodiedforge.runtime.hashing import copy_file_snapshot, snapshot_file_hashes


def smoke_training_clips(files):
    """Keep the walking smoke default, while accepting other single-skill data."""
    names = sorted(Path(name).stem for name in files)
    if not names:
        raise ValueError("Smoke training requires at least one motion")
    return ["walk1_subject1" if "walk1_subject1" in names else names[0]]


def motion_manifest(path):
    value = json.loads(Path(path).read_text())
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("Motion manifest requires a nonempty files mapping")
    for name, digest in files.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".npz")
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"Invalid motion manifest entry: {name!r}")
    return files


def check_npz(path):
    """Check ZIP members without deserializing arrays or loading the SDK."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                not members
                or len(members) > 64
                or len({x.filename for x in members}) != len(members)
                or sum(x.file_size for x in members) > 512 * 1024 * 1024
            ):
                raise ValueError("Unsupported member count or uncompressed size")
            bad = archive.testzip()
            if bad is not None:
                raise ValueError(f"CRC mismatch in {bad}")
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, zlib.error) as error:
        raise ValueError(
            f"Invalid motion archive {Path(path).name}: {error}"
        ) from error


def inspect_motion_files(source, expected=None, *, names=None):
    source = Path(source).expanduser().resolve(strict=True)
    if names is None:
        files = sorted(source.glob("*.npz"))
    else:
        if any(Path(name).name != name or not name.endswith(".npz") for name in names):
            raise ValueError("Motion selection requires flat NPZ file names")
        files = sorted(source / name for name in set(names))
    if not files:
        raise ValueError(f"No converted NPZ motions: {source}")
    if expected is not None:
        selected = {p.name for p in files}
        if (names is None and selected != set(expected)) or not selected.issubset(
            expected
        ):
            raise ValueError("Motion file set differs from the expected manifest")
    hashes = {}
    for path in files:
        digest = snapshot_file_hashes(path)[0]
        if expected is not None and digest != expected[path.name]:
            raise ValueError(
                f"Motion hash mismatch: {path.name}; "
                f"expected {expected[path.name]}, found {digest}"
            )
        check_npz(path)
        hashes[path.name] = digest
    return hashes


def snapshot_motions(source, destination, expected=None, *, names=None):
    """Copy regular files, verify written bytes/CRC, then remove write permission.

    The private copy is used for simulation. Changing the shared source after
    this function returns does not change the copy. Permissions are accidental
    write protection, not a guarantee against owner/root writes or storage faults.
    """
    source = Path(source).expanduser().resolve(strict=True)
    destination = Path(destination).resolve()
    if destination.is_relative_to(source):
        raise ValueError("Motion snapshot must be outside the source directory")
    hashes = inspect_motion_files(source, expected, names=names)
    destination.mkdir(parents=True, exist_ok=False)
    for name, digest in hashes.items():
        target = destination / name
        copied = copy_file_snapshot(source / name, target)
        if copied != digest or snapshot_file_hashes(target)[0] != digest:
            raise ValueError(f"Motion changed during copy: {name}")
        check_npz(target)
        target.chmod(0o444)
    validate_motion_snapshot(destination, hashes)
    return {
        "source": str(source),
        "root": str(destination),
        "files": hashes,
        "zip_crc_verified": True,
        "read_only_copies": True,
        "expected_manifest_verified": expected is not None,
    }


def validate_motion_snapshot(root, hashes):
    root = Path(root)
    if {p.name for p in root.glob("*.npz")} != set(hashes):
        raise ValueError("Private motion snapshot file set changed")
    for name, expected in hashes.items():
        actual = snapshot_file_hashes(root / name)[0]
        if actual != expected:
            raise ValueError(
                f"Private motion changed: {name}; expected {expected}, found {actual}"
            )
