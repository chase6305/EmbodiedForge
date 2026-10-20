"""File identities stay byte-exact without allocating an artifact-sized buffer."""

import hashlib
import tracemalloc

import pytest

from embodiedforge._hashing import file_sha256


@pytest.mark.parametrize("size", [0, 13, 1024**2, 3 * 1024**2 + 7])
def test_file_hash_matches_bytes_across_block_boundaries(tmp_path, size):
    pattern = b"artifact identity\x00\xff"
    payload = (pattern * (size // len(pattern) + 1))[:size]
    path = tmp_path / "artifact with spaces.bin"
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
    assert file_sha256(str(path)) == hashlib.sha256(payload).hexdigest()


def test_large_artifact_hash_has_bounded_python_allocations(tmp_path):
    path = tmp_path / "large-artifact.bin"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024**2)
    expected = hashlib.sha256()
    for _ in range(16):
        expected.update(bytes(1024**2))
    tracemalloc.start()
    try:
        result = file_sha256(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == expected.hexdigest()
    assert peak < 4 * 1024**2
