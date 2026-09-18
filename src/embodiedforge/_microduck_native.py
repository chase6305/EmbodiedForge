"""Dependency and implementation provenance for repository-owned Microduck."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent / "locomotion/microduck"
REQUIREMENTS = ROOT / "requirements.txt"
WORKER = Path(__file__).with_name("_microduck_native_worker.py")


def dependency_identity():
    return {
        "kind": "native_dependencies",
        "requirements_sha256": hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest(),
    }


def source_identity():
    upstream = json.loads((ROOT / "UPSTREAM.json").read_text())
    package = ROOT.parents[1]
    package_files = {
        str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    # Derive both views from the same reads, including large robot assets.
    prefix = ROOT.relative_to(package).as_posix() + "/"
    files = {
        name.removeprefix(prefix): digest
        for name, digest in package_files.items()
        if name.startswith(prefix)
    }
    for name in (
        "_microduck_native_worker.py",
        "_microduck_worker.py",
        "_microduck_native.py",
        "_microduck_process.py",
        "microduck.py",
    ):
        files[name] = package_files[name]
    return {
        "kind": "repository_owned",
        "implementation": "embodiedforge.locomotion.microduck",
        "revision": upstream["revision"],
        "uv_lock_sha256": upstream["uv_lock_sha256"],
        "implementation_sha256": hashlib.sha256(
            json.dumps(package_files, sort_keys=True).encode()
        ).hexdigest(),
        "files": files,
        "package_files": package_files,
        "requirements_sha256": package_files[str(REQUIREMENTS.relative_to(package))],
    }


def native_command(command):
    """Replace upstream task executables with our isolated task bootstrap."""
    if command[2:4] == ["-m", "mjlab_microduck.train_cli"]:
        return [*command[:2], str(WORKER), "train", *command[4:]]
    if command[2:4] == ["-m", "mjlab_microduck.export"]:
        return [*command[:2], str(WORKER), "export", *command[4:]]
    if command[2:4] == ["-m", "mjlab.scripts.play"]:
        return [*command[:2], str(WORKER), "sdk-play", *command[4:]]
    return command


def native_runtime():
    from ._training_runtime import component
    from .locomotion.microduck.actuator.friction_dr_bam import FrictionDRBamActuator
    from .locomotion.microduck.robot.microduck_constants import MICRODUCK_WALK_XML
    from .locomotion.microduck.tasks import MicroduckOnPolicyRunner, mdp
    from .locomotion.microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )

    return {
        "kind": "repository_owned",
        "task": component(make_microduck_velocity_env_cfg),
        "mdp": component(mdp),
        "actuator": component(FrictionDRBamActuator),
        "runner": component(MicroduckOnPolicyRunner),
        "asset": str(MICRODUCK_WALK_XML),
        "core_vector_env": False,
        "source": source_identity(),
    }


def snapshot_runtime(output, identity, env):
    """Freeze source bytes once; all later workers use the same package copy."""
    if (
        env.get("EF_MICRODUCK_NATIVE") != "1"
        or identity.get("kind") != "repository_owned"
    ):
        return env, None
    if env.get("EF_MICRODUCK_SNAPSHOT_WORKER"):
        worker = Path(env["EF_MICRODUCK_SNAPSHOT_WORKER"])
        return env, worker.parent
    package = ROOT.parents[1]
    destination = output / "implementation" / "embodiedforge"
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in identity["package_files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid implementation path: {name}")
        data = (package / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise RuntimeError(
                f"Implementation changed before snapshot: {name}; retry the run"
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    worker = destination / WORKER.name
    return {**env, "EF_MICRODUCK_SNAPSHOT_WORKER": str(worker)}, destination


def snapshot_command(command, env):
    worker = env.get("EF_MICRODUCK_SNAPSHOT_WORKER")
    if worker and command[1:3] == ["-I", str(WORKER)]:
        return [*command[:2], worker, *command[3:]]
    return command
