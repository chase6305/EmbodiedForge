"""Motion validation must not repeatedly decompress the same NPZ member."""

import json
from collections import Counter

import numpy as np
import pytest

from embodiedforge._h1_motion import load_motion


@pytest.fixture
def motion_fields():
    metadata = {
        "body_names": ["root", "foot"],
        "joint_names": ["knee"],
        "edges": [[0, 1]],
    }
    return {
        "metadata": np.array(json.dumps(metadata)),
        "time": np.array([0.02, 0.04, 0.06]),
        "positions": np.arange(18, dtype=np.float32).reshape(3, 2, 3) / 10,
        "quaternions": np.tile([0, 0, 0, 1], (3, 2, 1)).astype(np.float32),
        "joints": np.zeros((3, 1), dtype=np.float32),
        "terminated": np.array([False, False, True]),
        "truncated": np.zeros(3, dtype=bool),
    }


def member_reads(monkeypatch, path):
    with np.load(path, allow_pickle=False) as archive:
        archive_type = type(archive)
    original = archive_type.__getitem__
    reads = []

    def track(self, key):
        reads.append(key)
        return original(self, key)

    monkeypatch.setattr(archive_type, "__getitem__", track)
    return reads


def test_valid_motion_avoids_repeated_validation_decompression(
    tmp_path, monkeypatch, motion_fields
):
    path = tmp_path / "motion.npz"
    np.savez_compressed(path, **motion_fields)
    reads = member_reads(monkeypatch, path)
    motion = load_motion(path)
    counts = Counter(reads)
    assert set(counts) == set(motion_fields)
    assert len(reads) <= 9
    for key in ("metadata", "time", "positions", "quaternions", "joints"):
        assert counts[key] == 1
    assert counts["terminated"] <= 2 and counts["truncated"] <= 2
    assert motion["metadata"] == json.loads(str(motion_fields["metadata"]))
    assert motion["time"] == [0.02, 0.04, 0.06]
    assert (
        motion["positions"]
        == (np.arange(18, dtype=np.float64).reshape(3, 2, 3) / 10).tolist()
    )
    assert motion["terminated"] is True and motion["truncated"] is False


@pytest.mark.parametrize(
    "invalid", ["quaternions", "joints", "terminated", "truncated"]
)
def test_invalid_member_is_rejected_once_in_original_validation_order(
    tmp_path, monkeypatch, motion_fields, invalid
):
    members = ["quaternions", "joints", "terminated", "truncated"]
    for key in members[members.index(invalid) :]:
        if key in ("quaternions", "joints"):
            motion_fields[key].fill(np.nan)
        else:
            motion_fields[key][0] = True
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **motion_fields)
    reads = member_reads(monkeypatch, path)
    message = (
        f"Invalid motion {invalid}"
        if invalid in ("quaternions", "joints")
        else "Motion contains frames after a terminal state"
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        load_motion(path)
    assert reads == [
        "metadata",
        "time",
        "positions",
        *members[: members.index(invalid) + 1],
    ]
