"""Enforce dependency direction independently of optional SDK availability.

Edges represent explicit module/symbol imports, including deferred imports.
Importing a named submodule points to that module rather than treating its
package initializer as a symbol dependency. Dynamic import strings are outside
this check; runtime adapter tests cover their behavior.
"""

import ast
from importlib.util import resolve_name
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "src/embodiedforge"


def dependency_graph(root):
    modules = {
        ".".join(
            ("embodiedforge", *path.relative_to(root).with_suffix("").parts)
        ).removesuffix(".__init__"): path
        for path in root.rglob("*.py")
    }
    # Some isolated workers are launched as repository scripts. Include their
    # imports so moving a CLI dependency out of the package cannot hide it.
    modules.update(
        ("benchmarks." + path.stem, path)
        for path in (root.parents[1] / "benchmarks").glob("supermimic_*.py")
    )
    edges = {name: set() for name in modules}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                edges[name].update(
                    alias.name for alias in node.names if alias.name in modules
                )
            elif isinstance(node, ast.ImportFrom):
                base = (
                    resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module
                )
                children = {
                    base + "." + alias.name
                    for alias in node.names
                    if base + "." + alias.name in modules
                }
                edges[name].update(children)
                if len(children) != len(node.names) and base in modules:
                    edges[name].add(base)
        edges[name].discard(name)
    return edges


def cycles(edges):
    visited, active, stack, found = set(), set(), [], []

    def visit(name):
        if name in active:
            found.append(stack[stack.index(name) :] + [name])
            return
        if name in visited:
            return
        active.add(name)
        stack.append(name)
        for child in sorted(edges[name]):
            visit(child)
        stack.pop()
        active.remove(name)
        visited.add(name)

    for name in sorted(edges):
        visit(name)
    return found


GRAPH = dependency_graph(ROOT)


def test_explicit_module_dependencies_have_no_cycles():
    assert not cycles(GRAPH), "Explicit import cycles: " + repr(cycles(GRAPH))


@pytest.mark.parametrize(
    "prefix,allowed",
    [
        ("embodiedforge.runtime", ("embodiedforge.runtime",)),
        ("embodiedforge.viewers._frame_channel", ()),
        (
            "embodiedforge.viewers._frame_snapshot",
            ("embodiedforge.core", "embodiedforge.viewers.robot"),
        ),
        (
            "embodiedforge.learning",
            ("embodiedforge.learning", "embodiedforge.core", "embodiedforge.runtime"),
        ),
        (
            "embodiedforge.envs",
            (
                "embodiedforge.envs",
                "embodiedforge.core",
                "embodiedforge.locomotion.go1_config",
                "embodiedforge.locomotion.h1_config",
            ),
        ),
        (
            "embodiedforge.data",
            (
                "embodiedforge._episode_archive",
                "embodiedforge.core",
                "embodiedforge.envs",
                "embodiedforge.runtime",
            ),
        ),
        (
            "embodiedforge._episode_archive",
            (
                "embodiedforge.core",
                "embodiedforge.envs.specs",
                "embodiedforge.runtime.records",
            ),
        ),
        (
            "embodiedforge._weave_worker",
            (
                "embodiedforge._weave_evaluation",
                "embodiedforge.learning",
                "embodiedforge.runtime",
            ),
        ),
        (
            "embodiedforge._go1_training_viewer",
            ("embodiedforge.viewers.training",),
        ),
        (
            "embodiedforge._wuji_training_viewer",
            ("embodiedforge.viewers.training",),
        ),
    ],
)
def test_shared_layers_only_import_their_declared_dependencies(prefix, allowed):
    violations = [
        (name, child)
        for name, children in GRAPH.items()
        if name == prefix or name.startswith(prefix + ".")
        for child in children
        if not any(
            child == target or child.startswith(target + ".") for target in allowed
        )
    ]
    assert not violations, (
        "Shared layer imports a higher-level implementation: " + repr(violations)
    )


def test_runtime_consumers_do_not_import_command_entry_points():
    entry_points = {
        "embodiedforge.cli",
        "embodiedforge.act",
        "embodiedforge.gmr",
        "embodiedforge.sonic",
        "embodiedforge.supermimic",
        "embodiedforge.supermimic_run",
        "embodiedforge.morphmimic",
        "embodiedforge.motion",
        "embodiedforge.go1_live",
        "embodiedforge.robot_replay",
        "embodiedforge.visualization",
        "embodiedforge.recipes",
        "embodiedforge.h1",
        "embodiedforge.native_h1",
        "embodiedforge.microduck",
        "embodiedforge.rlinf",
        "embodiedforge.weave",
    }
    shared = (
        "embodiedforge.runtime",
        "embodiedforge.learning",
        "embodiedforge.workflows",
        "embodiedforge.envs",
        "embodiedforge.viewers",
        "embodiedforge._rlinf_worker",
        "embodiedforge._rlinf_export",
        "embodiedforge._rlinf_onnx",
    )
    violations = [
        (name, child)
        for name, children in GRAPH.items()
        if any(name == prefix or name.startswith(prefix + ".") for prefix in shared)
        or name.startswith("benchmarks.supermimic_")
        for child in children
        if child in entry_points
    ]
    assert not violations, "Implementation imports a command entry point: " + repr(
        violations
    )
