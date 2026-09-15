"""Live tasks must use recorded source, including relative imports and scene data."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiedforge._go1_implementation import (
    copy_go1_implementation,
    load_go1_implementation,
)
from embodiedforge.recipes import sha256


@pytest.fixture
def recorded(tmp_path):
    source = tmp_path / "run" / "implementation" / "embodiedforge"
    files = {
        "locomotion/__init__.py": "",
        "locomotion/go1_config.py": "VALUE = 17\n",
        "locomotion/go1.py": (
            "from pathlib import Path\n"
            "from .go1_config import VALUE\n"
            "class Go1: value = VALUE\n"
            "CTRL_DT = 0.02\n"
            "THEME = str(Path(__file__).with_name('go1_scene.xml'))\n"
        ),
        "locomotion/go1_ppo.py": "from .go1 import Go1\nclass ActorCritic: value = Go1.value\n",
        "locomotion/go1_scene.xml": "<mujoco/>\n",
    }
    hashes = {}
    for name, text in files.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        hashes[name] = sha256(path)
    manifest = {
        "implementation": {"path": "implementation/embodiedforge", "files": hashes}
    }
    return tmp_path / "run", source, manifest


def test_frozen_relative_imports_ignore_current_modules_and_unlisted_files(
    recorded, tmp_path, monkeypatch
):
    run, source, manifest = recorded
    for name in ("go1", "go1_config", "go1_ppo"):
        monkeypatch.setitem(
            sys.modules, f"embodiedforge.locomotion.{name}", SimpleNamespace()
        )
    # An unrecorded package would shadow the recorded go1_config.py if the
    # original snapshot directory were imported directly.
    shadow = source / "locomotion" / "go1_config" / "__init__.py"
    shadow.parent.mkdir()
    shadow.write_text("raise RuntimeError('unrecorded code executed')\n")
    record = copy_go1_implementation(run, manifest, tmp_path / "private")
    (source / "locomotion/go1_config.py").write_text("VALUE = 99\n")
    task, learner, dt, details = load_go1_implementation(record)
    assert task.value == learner.value == 17
    assert dt == 0.02
    assert details["mode"] == "training_snapshot"
    assert (
        details["task"]["sha256"]
        == manifest["implementation"]["files"]["locomotion/go1.py"]
    )
    assert Path(details["scene"]["path"]).read_text() == "<mujoco/>\n"
    assert not (tmp_path / "private/locomotion/go1_config").exists()


@pytest.mark.parametrize("damage", ["modify", "delete", "symlink", "parent_symlink"])
def test_damaged_recorded_source_is_rejected(recorded, tmp_path, damage):
    run, source, manifest = recorded
    path = source / "locomotion/go1.py"
    if damage == "modify":
        path.write_text("raise RuntimeError('changed')\n")
    elif damage == "delete":
        path.unlink()
    elif damage == "symlink":
        original = tmp_path / "original.py"
        path.rename(original)
        path.symlink_to(original)
    else:
        directory = source / "locomotion"
        directory.rename(source / "moved")
        directory.symlink_to(source / "moved", target_is_directory=True)
    with pytest.raises(ValueError, match="snapshot (differs|symlink)"):
        copy_go1_implementation(run, manifest, tmp_path / "private")


@pytest.mark.parametrize("name", ["../outside.py", "/outside.py", "a/../../outside.py"])
@pytest.mark.parametrize("field", ["path", "file"])
def test_snapshot_paths_cannot_escape_run(recorded, tmp_path, name, field):
    run, _, manifest = recorded
    if field == "path":
        manifest["implementation"]["path"] = name
    else:
        manifest["implementation"]["files"][name] = "0" * 64
    with pytest.raises(ValueError, match="Invalid implementation snapshot path"):
        copy_go1_implementation(run, manifest, tmp_path / "private")


@pytest.mark.parametrize("record", [None, {}, {"files": {}}, {"files": []}])
def test_present_but_invalid_snapshot_never_falls_back(tmp_path, record):
    with pytest.raises(ValueError, match="missing the Go1 task files"):
        copy_go1_implementation(
            tmp_path, {"implementation": record}, tmp_path / "private"
        )


def test_absent_legacy_snapshot_is_explicit(tmp_path):
    assert copy_go1_implementation(tmp_path, {}, tmp_path / "private") is None


def test_worker_rechecks_private_copy_before_import(recorded, tmp_path):
    run, _, manifest = recorded
    record = copy_go1_implementation(run, manifest, tmp_path / "private")
    (tmp_path / "private/locomotion/go1.py").write_text(
        "raise RuntimeError('executed')"
    )
    with pytest.raises(ValueError, match="snapshot differs in policy worker"):
        load_go1_implementation(record)
