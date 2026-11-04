"""Source snapshots shared by native training and external SDK workers."""

from pathlib import Path, PurePosixPath, PureWindowsPath

from embodiedforge.runtime.hashing import copy_file_snapshot, snapshot_file_hashes


def validate_snapshot(project: Path, source: dict, *, label="Cached source") -> None:
    """Validate only files inside the declared snapshot, without linked entries."""
    project = Path(project).resolve()
    files = source.get("files") if isinstance(source, dict) else None
    if not isinstance(files, dict):
        raise ValueError(f"{label} differs: invalid file manifest")
    for relative, expected in files.items():
        if (
            not isinstance(relative, str)
            or not relative
            or "\\" in relative
            or PurePosixPath(relative).is_absolute()
            or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))
        ):
            raise ValueError(f"{label} differs: invalid relative path {relative!r}")
        path = project / relative
        linked = any(
            parent.is_symlink()
            for parent in (path, *path.parents)
            if parent != project and parent.is_relative_to(project)
        )
        if linked or not path.is_file() or snapshot_file_hashes(path)[0] != expected:
            raise ValueError(f"{label} differs: {path}")


def snapshot_implementation(package: Path, destination: Path) -> dict:
    """Freeze Python modules and native-task XML/license files for one run.

    This is an implementation snapshot, not an inventory of every package asset.
    Workflows separately record their model/mesh inputs and viewer resources.
    """
    package, destination = package.resolve(), destination.resolve()
    if not package.is_dir():
        raise ValueError(f"Implementation source is not a directory: {package}")
    if destination.is_relative_to(package):
        raise ValueError(
            "Run implementation snapshot must be outside the source package"
        )
    files = []
    for path in sorted(package.rglob("*")):
        if "__pycache__" in path.relative_to(package).parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Implementation source must not contain symlinks: {path}")
        if path.is_file() and (
            path.suffix in (".py", ".xml") or path.name.startswith("LICENSE")
        ):
            files.append(path)
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for path in files:
        relative = path.relative_to(package)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = copy_file_snapshot(path, target)
        if snapshot_file_hashes(target)[0] != expected:
            raise ValueError(f"Implementation source changed during copy: {path}")
        hashes[relative.as_posix()] = expected
    validate_snapshot(package, {"files": hashes}, label="Implementation source")
    return {"files": hashes}
