"""File identities stay byte-exact without allocating an artifact-sized buffer."""

import hashlib
import tracemalloc

import pytest

from embodiedforge.runtime.hashing import file_sha256


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


def test_stream_hash_borrows_current_position_without_closing_or_rewinding():
    import io

    from embodiedforge.runtime.hashing import stream_sha256

    stream = io.BytesIO(b"prefix payload bytes")
    stream.seek(7)
    assert stream_sha256(stream) == hashlib.sha256(b"payload bytes").hexdigest()
    assert stream.tell() == len(b"prefix payload bytes")
    assert not stream.closed


def test_file_hash_rejects_file_descriptors_without_closing_them(tmp_path):
    path = tmp_path / "borrowed.bin"
    path.write_bytes(b"caller owns this handle")
    with path.open("rb") as stream:
        with pytest.raises(TypeError):
            file_sha256(stream.fileno())
        assert stream.read() == b"caller owns this handle"


@pytest.mark.parametrize("size", [0, 17, 2 * 1024 * 1024 + 13])
def test_snapshot_copy_hashes_exact_bytes_and_borrows_sink(tmp_path, size):
    import hashlib
    import io

    from embodiedforge.runtime.hashing import snapshot_file_hashes

    class ShortWriter(io.BytesIO):
        def write(self, block):
            return super().write(block[:7919])

    content = b"x" * size
    source = tmp_path / "source"
    source.write_bytes(content)
    sink = ShortWriter()
    digest, blob, info = snapshot_file_hashes(source, "sha1", copy_to=sink)
    assert not sink.closed
    assert sink.getvalue() == content
    assert digest == hashlib.sha256(content).hexdigest()
    assert (
        blob
        == hashlib.sha1(b"blob " + str(size).encode() + b"\0" + content).hexdigest()
    )
    assert info.st_size == size


@pytest.mark.parametrize("result", [None, 0, -1, True, 100])
def test_snapshot_rejects_nonprogressing_or_invalid_sink(tmp_path, result):
    from embodiedforge.runtime.hashing import snapshot_file_hashes

    source = tmp_path / "source"
    source.write_bytes(b"content")

    class InvalidWriter:
        closed = False

        def write(self, block):
            return result

    sink = InvalidWriter()
    with pytest.raises(OSError, match="write progress"):
        snapshot_file_hashes(source, copy_to=sink)
    assert not sink.closed


def test_snapshot_copy_never_overwrites_existing_destination(tmp_path):
    from embodiedforge.runtime.hashing import copy_file_snapshot

    source = tmp_path / "source"
    source.write_bytes(b"new")
    target = tmp_path / "target"
    target.write_bytes(b"existing")
    link = tmp_path / "link"
    link.symlink_to(target)
    for destination in (target, link):
        with pytest.raises(FileExistsError):
            copy_file_snapshot(source, destination)
        assert target.read_bytes() == b"existing"


def test_snapshot_detects_source_mutation_by_sink(tmp_path):
    from embodiedforge.runtime.hashing import snapshot_file_hashes

    source = tmp_path / "source"
    source.write_bytes(b"original")

    class MutatingWriter:
        def write(self, block):
            with source.open("ab") as stream:
                stream.write(b"growth")
            return len(block)

    with pytest.raises(ValueError, match="changed while hashing"):
        snapshot_file_hashes(source, copy_to=MutatingWriter())
