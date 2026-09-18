"""Interrupted checkpoint saves must not replace the last complete model."""

from pathlib import Path

import pytest

from embodiedforge.locomotion.microduck.checkpoint import atomic_save


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt])
def test_partial_save_does_not_publish_or_destroy_previous_checkpoint(tmp_path, error):
    target = tmp_path / "model_0.pt"
    target.write_bytes(b"previous complete model")

    def save(path, infos):
        assert infos == {"counter": 42}
        Path(path).write_bytes(b"partial new model")
        assert list(tmp_path.glob("model_*.pt")) == [target]
        assert target.read_bytes() == b"previous complete model"
        raise error("interrupted save")

    with pytest.raises(error):
        atomic_save(save, target, {"counter": 42})
    assert target.read_bytes() == b"previous complete model"
    assert list(tmp_path.iterdir()) == [target]


def test_complete_model_is_published_then_can_be_replaced(tmp_path):
    target = tmp_path / "model_250.pt"

    def save(path, infos):
        assert infos is None
        assert not target.exists()
        Path(path).write_bytes(b"complete model")

    atomic_save(save, target)
    assert target.read_bytes() == b"complete model"
    assert list(tmp_path.iterdir()) == [target]
    atomic_save(lambda path, infos: Path(path).write_bytes(b"new"), target)
    assert target.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [target]
