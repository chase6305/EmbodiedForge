"""Load recorded Go1 task code while keeping the live transport up to date."""

import importlib
import importlib.util
import shutil
import sys
import uuid
from pathlib import Path, PurePosixPath

from .recipes import sha256

_REQUIRED = {
    "locomotion/__init__.py",
    "locomotion/go1.py",
    "locomotion/go1_config.py",
    "locomotion/go1_ppo.py",
    "locomotion/go1_scene.xml",
}


def _contained(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("Invalid implementation snapshot path")
    parts = PurePosixPath(relative).parts
    if (
        PurePosixPath(relative).is_absolute()
        or ".." in parts
        or str(PurePosixPath(relative)) != relative
    ):
        raise ValueError(f"Invalid implementation snapshot path: {relative}")
    path = root
    for part in parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f"Implementation snapshot symlink: {path}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Implementation snapshot escapes its directory: {relative}")
    return path


def _files(record):
    files = record.get("files") if isinstance(record, dict) else None
    if not isinstance(files, dict) or not _REQUIRED.issubset(files):
        raise ValueError("Implementation snapshot is missing the Go1 task files")
    for name, digest in files.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"Invalid implementation SHA256: {name}")
    return files


def copy_go1_implementation(run, manifest, destination):
    """Copy only manifest-listed files; never import a run's extra files/pyc."""
    if "implementation" not in manifest:
        return None  # Legacy runs explicitly report that source is unavailable.
    record = manifest["implementation"]
    files = _files(record)
    source = _contained(Path(run), record.get("path"))
    destination = Path(destination)
    for name, digest in files.items():
        path = _contained(source, name)
        target = _contained(destination, name)
        if path.suffix not in (".py", ".xml") and not path.name.startswith("LICENSE"):
            raise ValueError(f"Unsupported implementation snapshot file: {name}")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Implementation snapshot differs: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        if sha256(target) != digest:
            raise ValueError(f"Implementation snapshot changed during copy: {path}")
    return {"path": str(destination), "files": files}


def load_go1_implementation(record=None):
    """Isolate relative task imports from the current installed locomotion code."""
    if record is None:
        task = importlib.import_module("embodiedforge.locomotion.go1")
        learner = importlib.import_module("embodiedforge.locomotion.go1_ppo")
        mode = "current_code_legacy"
    else:
        files = _files(record)
        root = Path(record["path"])
        for name, digest in files.items():
            path = _contained(root, name)
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(
                    f"Implementation snapshot differs in policy worker: {path}"
                )
        name = f"embodiedforge._live_go1_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(
            name, root / "locomotion" / "__init__.py"
        )
        package = importlib.util.module_from_spec(spec)
        sys.modules[name] = package
        try:
            spec.loader.exec_module(package)
            task = importlib.import_module(f"{name}.go1")
            learner = importlib.import_module(f"{name}.go1_ppo")
        except BaseException:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]
            raise
        mode = "training_snapshot"
    details = {"mode": mode}
    for label, path in (
        ("task", Path(task.__file__)),
        ("learner", Path(learner.__file__)),
        ("scene", Path(task.THEME)),
    ):
        details[label] = {"path": str(path), "sha256": sha256(path)}
    return task.Go1, learner.ActorCritic, task.CTRL_DT, details
