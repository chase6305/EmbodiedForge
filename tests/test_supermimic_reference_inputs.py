"""The policy and its quality checks must use immutable, verified motion inputs."""

import hashlib
import stat

import pytest

from embodiedforge.workflows import supermimic_backflip as backflip
from embodiedforge.workflows.supermimic_reference_evaluate import (
    snapshot_diagnostic_reference,
)


def test_diagnostic_snapshot_survives_shared_reference_changes(tmp_path):
    source = backflip.task_directory(tmp_path) / "data/eval/backflip.npz"
    source.parent.mkdir(parents=True)
    payload = b"controlled reference bytes; not a physical rollout"
    source.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    private, actual = snapshot_diagnostic_reference(
        {"workspace": str(tmp_path), "expected_motions": {"backflip.npz": digest}},
        tmp_path / "evaluation",
    )
    source.write_bytes(b"shared cache changed after evaluation began")
    assert private.read_bytes() == payload and actual == digest
    assert not private.stat().st_mode & stat.S_IWUSR
    assert private.stat().st_ino != source.stat().st_ino


def test_changed_reference_is_rejected_against_the_frozen_manifest(tmp_path):
    source = backflip.task_directory(tmp_path) / "data/eval/backflip.npz"
    source.parent.mkdir(parents=True)
    expected = hashlib.sha256(b"original reference").hexdigest()
    source.write_bytes(b"changed before copying")
    with pytest.raises(ValueError, match="expected input manifest"):
        snapshot_diagnostic_reference(
            {
                "workspace": str(tmp_path),
                "expected_motions": {"backflip.npz": expected},
            },
            tmp_path / "evaluation",
        )
