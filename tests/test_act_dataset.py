"""Data identity must follow bytes and paths, not timestamps or a manifest alone."""

import os
import shutil

import pytest

from embodiedforge._act_dataset import (
    fingerprint_dataset,
    verify_dataset_fingerprint,
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    for name, content in {
        "embodiedforge.json": b"{}",
        "meta/stats.json": b'{"mean": 1}',
        "data/chunk-000/file-000.parquet": b"sample actions",
        "images/camera/episode-000/frame-000.png": b"image pixels",
        "videos/camera/chunk-000/file-000.mp4": b"video frames",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


@pytest.mark.parametrize(
    "name",
    [
        "meta/stats.json",
        "data/chunk-000/file-000.parquet",
        "images/camera/episode-000/frame-000.png",
        "videos/camera/chunk-000/file-000.mp4",
    ],
)
def test_same_size_edit_with_restored_mtime_is_rejected(dataset, name):
    original = fingerprint_dataset(dataset)
    path = dataset / name
    stat = path.stat()
    content = path.read_bytes()
    path.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    current = fingerprint_dataset(dataset)
    with pytest.raises(ValueError, match="Dataset content changed") as error:
        verify_dataset_fingerprint(original, current)
    assert name in str(error.value)


@pytest.mark.parametrize("change", ["add", "delete", "rename"])
def test_dataset_file_membership_is_checked(dataset, change):
    original = fingerprint_dataset(dataset)
    path = dataset / "meta/stats.json"
    if change == "add":
        (dataset / "meta/extra.json").write_text("{}")
    elif change == "delete":
        path.unlink()
    else:
        path.rename(dataset / "meta/renamed.json")
    with pytest.raises(ValueError, match="Dataset content changed"):
        verify_dataset_fingerprint(original, fingerprint_dataset(dataset))


def test_identity_survives_move_and_ignores_unrelated_root_files(dataset, tmp_path):
    original = fingerprint_dataset(dataset)
    moved = tmp_path / "moved"
    shutil.copytree(dataset, moved)
    (moved / "README.md").write_text("experiment notes")
    cache = moved / ".cache"
    cache.mkdir()
    (cache / "unrelated.lock").touch()
    current = fingerprint_dataset(moved)
    assert original == current
    assert verify_dataset_fingerprint(original, current)["status"] == "verified"


def test_legacy_identity_is_not_claimed_verified(dataset):
    current = fingerprint_dataset(dataset)
    assert verify_dataset_fingerprint(None, current) == {
        "status": "unverified_legacy",
        "current_sha256": current["sha256"],
    }
    unsupported = {**current, "schema_version": 2}
    with pytest.raises(ValueError, match="Unsupported"):
        verify_dataset_fingerprint(unsupported, current)


def test_symlinked_data_cannot_escape_fingerprint(dataset, tmp_path):
    target = tmp_path / "external"
    target.mkdir()
    (target / "payload").write_text("actions")
    (dataset / "data/linked").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        fingerprint_dataset(dataset)
