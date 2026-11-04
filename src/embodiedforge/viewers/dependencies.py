"""SDK-free installation metadata checks for viewers and frame renderers."""

from importlib import metadata
from platform import python_version
from sys import version_info

from embodiedforge.backends import NEWTON_VERSION


def viewer_dependencies(name: str) -> dict:
    """Read package metadata only; does not certify GPU/display/driver availability."""
    requirements = {
        "viser": ("viser", "pillow"),
        "gl": ("newton", "warp-lang", "pyglet"),
        "rtx": ("newton", "warp-lang", "pyglet", "usd-core", "ovrtx"),
        "web": ("pillow",),
    }
    if name not in requirements:
        raise ValueError(f"Unknown viewer: {name}")
    packages = {}
    optional = {}
    for package in requirements[name]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    if name in ("gl", "rtx"):
        try:
            optional["imgui-bundle"] = metadata.version("imgui-bundle")
        except metadata.PackageNotFoundError:
            optional["imgui-bundle"] = None
    notes = []
    if name == "rtx" and not (3, 11) <= version_info[:2] <= (3, 13):
        notes.append(
            "OVRTX upstream documents Python 3.11-3.13 support. "
            "An installed package on this Python version is not a supported-environment check."
        )
    if name in ("gl", "rtx") and optional.get("imgui-bundle") is None:
        notes.append(
            "Native UI panels are unavailable: install embodiedforge[viz-ui]. "
            "Python 3.10 has no imgui-bundle >=1.92 wheel; use a separate Python 3.11 "
            "viewer environment for binary installation."
        )
    return {
        "viewer": name,
        "python": python_version(),
        "packages": packages,
        "optional_packages": optional,
        "environment_notes": notes,
        "metadata_ok": all(packages.values())
        and ("newton" not in packages or packages["newton"] == NEWTON_VERSION),
        "check_scope": "installation_metadata_only",
        "install_extra": "viz" if name == "viser" else f"viz-{name}",
    }


def require_viewer_dependencies(name: str, *, report: dict | None = None) -> None:
    """Fail before physics initialization with an actionable install command."""
    if report is None:
        report = viewer_dependencies(name)
    if report["metadata_ok"]:
        return
    missing = [
        package for package, version in report["packages"].items() if version is None
    ]
    problems = [f"missing packages: {', '.join(missing)}"] if missing else []
    installed = report["packages"].get("newton")
    if installed is not None and installed != NEWTON_VERSION:
        problems.append(f"Newton {installed} installed; requires {NEWTON_VERSION}")
    raise ImportError(
        f"Viewer {name}: {'; '.join(problems)}. "
        f"In the active environment, run: python -m pip install -e '.[{report['install_extra']}]'"
    )
