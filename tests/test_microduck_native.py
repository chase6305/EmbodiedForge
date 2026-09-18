"""Native dispatch, plugin isolation and packaged robot asset integrity."""

import hashlib
import importlib.metadata
import json
import runpy
import sys
import xml.etree.ElementTree as ET

import pytest

from embodiedforge import _microduck_native as native
from embodiedforge import microduck


def test_native_environment_flag_is_not_inherited(monkeypatch):
    monkeypatch.setenv("EF_MICRODUCK_NATIVE", "1")
    assert "EF_MICRODUCK_NATIVE" not in microduck.child_environment()


@pytest.mark.parametrize("existing", [False, True])
def test_default_setup_uses_bundled_dependencies(tmp_path, monkeypatch, existing):
    monkeypatch.chdir(tmp_path)
    environment = tmp_path / ".cache/microduck-native-venv"
    calls = []
    if existing:
        (environment / "bin").mkdir(parents=True)
        (environment / "bin/python").touch()

    def execute(command, *, cwd, env):
        calls.append(command)
        assert env["EF_MICRODUCK_NATIVE"] == "1"
        environment.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(microduck, "run", execute)
    monkeypatch.setattr(
        microduck, "source_identity", lambda _: pytest.fail("read upstream")
    )
    microduck.main(["setup"])
    assert len(calls) == (1 if existing else 2)
    assert calls[-1] == [
        "uv",
        "pip",
        "sync",
        "--python",
        str(environment / "bin/python"),
        str(native.REQUIREMENTS),
    ]
    assert (
        json.loads((environment / "embodiedforge-source.json").read_text())
        == native.dependency_identity()
    )


def test_default_check_bootstraps_local_worker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    environment = tmp_path / ".cache/microduck-native-venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    stamp = environment / "embodiedforge-source.json"
    stamp.write_text(json.dumps(native.dependency_identity()))
    calls = []
    monkeypatch.setattr(microduck, "run", lambda command, **kw: calls.append(command))
    monkeypatch.setattr(
        microduck, "source_identity", lambda _: pytest.fail("read upstream")
    )
    microduck.main(["check"])
    assert calls == [
        [str(environment / "bin/python"), "-I", str(native.WORKER), "check"]
    ]
    stamp.write_text("{}")
    with pytest.raises(SystemExit) as exc:
        microduck.main(["check"])
    assert exc.value.code == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "operation,module",
    [
        ("train", "mjlab_microduck.train_cli"),
        ("export", "mjlab_microduck.export"),
        ("sdk-play", "mjlab.scripts.play"),
    ],
)
def test_native_command_preserves_arguments(operation, module):
    args = [
        "python",
        "-I",
        "-m",
        module,
        microduck.TASK,
        "--arbitrary",
        "path with spaces",
    ]
    assert native.native_command(args) == [
        "python",
        "-I",
        str(native.WORKER),
        operation,
        *args[4:],
    ]
    assert args[2:4] == ["-m", module]


def test_native_worker_refuses_upstream_auto_plugin(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    worker = runpy.run_path(str(native.WORKER))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: object())
    with pytest.raises(RuntimeError, match="without the upstream task plugin"):
        worker["main"]()


def test_native_assets_are_complete_and_match_recorded_origin():
    provenance = json.loads((native.ROOT / "UPSTREAM.json").read_text())
    xml = native.ROOT / "robot/microduck/robot_walk.xml"
    document = ET.parse(xml)
    meshdir = document.find("compiler").attrib["meshdir"]
    meshes = {
        mesh.attrib["file"]
        for mesh in document.findall(".//mesh")
        if "file" in mesh.attrib
    }
    assert len(meshes) == 38
    for path in [xml, *(xml.parent / meshdir / name for name in meshes)]:
        record = provenance["files"][str(path.relative_to(native.ROOT))]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == record["upstream_sha256"]
        )
    assert (
        (native.ROOT / "LICENSE")
        .read_text()
        .startswith("                                 Apache License")
    )
    assert provenance["revision"] == microduck.REVISION


def test_identity_tracks_implementation_separately_from_dependencies():
    identity = native.source_identity()
    assert identity["kind"] == "repository_owned"
    assert "tasks/mdp.py" in identity["files"]
    assert "_microduck_native_worker.py" in identity["files"]
    assert "robot/microduck/robot_walk.xml" in identity["files"]
    assert len(identity["implementation_sha256"]) == 64
    for name, digest in identity["files"].items():
        package_name = (
            name
            if name.startswith("_microduck") or name == "microduck.py"
            else f"locomotion/microduck/{name}"
        )
        assert digest == identity["package_files"][package_name]
    assert identity["requirements_sha256"] == identity["files"]["requirements.txt"]
    dependencies = native.dependency_identity()
    assert set(dependencies) == {"kind", "requirements_sha256"}
    assert dependencies["requirements_sha256"] == identity["requirements_sha256"]
    requirements = native.REQUIREMENTS.read_text()
    assert "mjlab-microduck" not in requirements
    assert "/home/" not in requirements
    assert "mjlab==1.3.0" in requirements


@pytest.mark.parametrize("operation", ["progress", "metrics", "checkpoint", "onnx"])
def test_file_diagnostics_do_not_register_simulation_tasks(monkeypatch, operation):
    from embodiedforge.locomotion import microduck as task_package

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(native.WORKER), operation])
    worker = runpy.run_path(str(native.WORKER))

    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)
    monkeypatch.setattr(
        task_package,
        "register",
        lambda: pytest.fail("Simulator imported for a file diagnostic"),
    )
    calls = []
    monkeypatch.setattr(runpy, "run_module", lambda name, **kwargs: calls.append(name))
    worker["main"]()
    assert calls == ["embodiedforge._microduck_worker"]


def test_training_still_registers_local_task_before_dispatch(monkeypatch):
    from embodiedforge.locomotion import microduck as task_package

    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", [str(native.WORKER), "train", microduck.TASK])
    worker = runpy.run_path(str(native.WORKER))

    def missing_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "distribution", missing_distribution)
    calls = []
    monkeypatch.setattr(task_package, "register", lambda: calls.append("register"))
    monkeypatch.setattr(runpy, "run_module", lambda name, **kwargs: calls.append(name))
    worker["main"]()
    assert calls == ["register", "mjlab.scripts.train"]


def test_implementation_snapshot_copies_verified_bytes_and_survives_edits(
    tmp_path, monkeypatch
):
    package = tmp_path / "source/embodiedforge"
    task = package / "locomotion/microduck"
    task.mkdir(parents=True)
    data = {
        "_microduck_native_worker.py": b"# worker\n",
        "locomotion/microduck/task.py": b"value=1\n",
    }
    for name, content in data.items():
        (package / name).write_bytes(content)
    monkeypatch.setattr(native, "ROOT", task)
    identity = {
        "kind": "repository_owned",
        "package_files": {
            name: hashlib.sha256(content).hexdigest() for name, content in data.items()
        },
    }
    env, snapshot = native.snapshot_runtime(
        tmp_path / "run", identity, {"EF_MICRODUCK_NATIVE": "1"}
    )
    assert snapshot == tmp_path / "run/implementation/embodiedforge"
    for name, content in data.items():
        assert (snapshot / name).read_bytes() == content
        (package / name).write_text("modified later")
        assert (snapshot / name).read_bytes() == content
    assert env["EF_MICRODUCK_SNAPSHOT_WORKER"] == str(
        snapshot / "_microduck_native_worker.py"
    )
    # Sequential evaluation seeds reuse their batch's single source snapshot.
    assert native.snapshot_runtime(tmp_path / "run/seed-1", identity, env) == (
        env,
        snapshot,
    )
    assert not (tmp_path / "run/seed-1/implementation").exists()
    with pytest.raises(RuntimeError, match="changed before snapshot"):
        native.snapshot_runtime(
            tmp_path / "new-run", identity, {"EF_MICRODUCK_NATIVE": "1"}
        )


def test_snapshot_cannot_inherit_from_callers_environment(monkeypatch):
    monkeypatch.setenv("EF_MICRODUCK_SNAPSHOT_WORKER", "/other/worker.py")
    assert "EF_MICRODUCK_SNAPSHOT_WORKER" not in microduck.child_environment()


def test_snapshot_dispatch_preserves_isolated_arguments():
    command = [
        "python",
        "-I",
        str(native.WORKER),
        "train",
        microduck.TASK,
        "--video",
        "False",
    ]
    env = {"EF_MICRODUCK_SNAPSHOT_WORKER": "/frozen/_microduck_native_worker.py"}
    actual = native.snapshot_command(command, env)
    assert actual[2] == env["EF_MICRODUCK_SNAPSHOT_WORKER"]
    assert actual[:2] == command[:2] and actual[3:] == command[3:]
    legacy = ["python", "-I", "-m", "mjlab_microduck.train_cli"]
    assert native.snapshot_command(legacy, env) == legacy
