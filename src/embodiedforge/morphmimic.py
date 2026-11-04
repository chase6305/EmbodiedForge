"""Small, isolated launcher for the pinned author implementation of MorphMimic."""

import argparse
import json
import math
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .runtime.artifacts import write_json
from .runtime.environment import child_environment
from .runtime.hashing import file_sha256, snapshot_file_hashes, stream_sha256
from .runtime.process import run_process
from .runtime.records import run_identity, run_lifecycle

REVISION = "aa40fe46a7fa995ba7a7354ad065a2c8ad64a302"
ARCHIVE_SHA256 = "22cda54a35f06ba675cb74f747fd1a8e17f21dd0fd7bf9307b33ec3af8f9b321"
REPOSITORY = "https://github.com/lzyang2000/morphmimic"
ROBOTS = ("g1", "x2", "t2")
DEFAULT_DATA = "runs/morphmimic-data-v1"


def _source_members(bundle):
    for member in bundle:
        relative = Path(*Path(member.name).parts[1:])
        if not relative.parts:
            continue
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (member.isfile() or member.isdir())
        ):
            raise ValueError(f"Unexpected source archive entry: {member.name}")
        yield relative, member


def verify_source(source, archive):
    """Compare extracted files to a hash-pinned upstream archive, including assets."""
    count = 0
    with open(archive, "rb") as stream:
        if stream_sha256(stream) != ARCHIVE_SHA256:
            raise ValueError(
                "MorphMimic source archive hash differs from the pinned release"
            )
        stream.seek(0)
        with tarfile.open(fileobj=stream) as bundle:
            for relative, member in _source_members(bundle):
                if member.isfile():
                    path = source / relative
                    with bundle.extractfile(member) as incoming:
                        expected = stream_sha256(incoming)
                    if path.is_symlink() or snapshot_file_hashes(path)[0] != expected:
                        raise ValueError(
                            f"Pinned MorphMimic source changed: {relative}"
                        )
                    count += 1
    return dict(
        repository=REPOSITORY,
        revision=REVISION,
        archive_sha256=ARCHIVE_SHA256,
        verified_files=count,
    )


def setup(source, environment, archive):
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install uv first, then run ./morphmimic setup")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        with tempfile.TemporaryDirectory(
            prefix="morphmimic-", dir=archive.parent
        ) as scratch:
            temporary = Path(scratch) / "source.tar.gz"
            with (
                urllib.request.urlopen(
                    f"https://codeload.github.com/lzyang2000/morphmimic/tar.gz/{REVISION}",
                    timeout=120,
                ) as response,
                temporary.open("xb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            if file_sha256(temporary) != ARCHIVE_SHA256:
                raise ValueError("Downloaded archive checksum mismatch")
            temporary.rename(archive)
    if file_sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Source archive checksum mismatch")
    if not source.exists():
        source.mkdir(parents=True)
        with tarfile.open(archive) as bundle:
            for relative, member in _source_members(bundle):
                destination = source / relative
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        bundle.extractfile(member) as incoming,
                        destination.open("xb") as outgoing,
                    ):
                        shutil.copyfileobj(incoming, outgoing)
    provenance = verify_source(source, archive)
    python = environment / "bin/python"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.11", str(environment)], check=True)
    lock = archive.parent / "pylock.runtime.toml"
    subprocess.run(
        [
            uv,
            "export",
            "--project",
            str(source),
            "--frozen",
            "--no-default-groups",
            "--no-dev",
            "--group",
            "export",
            "--no-editable",
            "--no-emit-workspace",
            "--format",
            "pylock.toml",
            "--output-file",
            str(lock),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # Transport mirror only: PEP 751 retains every upstream artifact hash.
    mirrored = lock.with_name("pylock.runtime-mirrored.toml")
    text = lock.read_text()
    for host in ("download.pytorch.org", "download-r2.pytorch.org"):
        text = text.replace(
            f"https://{host}/whl/", "https://mirrors.aliyun.com/pytorch-wheels/"
        )
    text = text.replace(
        "https://files.pythonhosted.org/packages/",
        "https://mirrors.aliyun.com/pypi/packages/",
    )
    mirrored.write_text(text)
    subprocess.run(
        [uv, "pip", "sync", "--python", str(python), str(mirrored)], check=True
    )
    subprocess.run([uv, "pip", "check", "--python", str(python)], check=True)
    write_json(source / "embodiedforge-source.json", provenance)
    print(f"Ready: {python}\nNext: ./morphmimic prepare")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command", choices=("setup", "doctor", "prepare", "train", "export", "eval")
    )
    result.add_argument("--workspace", type=Path, default=Path.cwd())
    result.add_argument("--output", type=Path)
    result.add_argument(
        "--data", type=Path, help="Prepared directory containing <robot>.yaml"
    )
    result.add_argument("--robots", choices=ROBOTS, nargs="+", default=list(ROBOTS))
    result.add_argument(
        "--clips", type=Path, nargs="+", help="Existing 50 Hz G1 NPZ files"
    )
    result.add_argument("--seconds", type=float, default=12)
    result.add_argument("--iterations", type=int, default=600)
    result.add_argument("--num-envs", type=int, default=32)
    result.add_argument("--batch", type=int, default=32)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--resume", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--policy", type=Path, help="Exported policy directory")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if len(set(args.robots)) != len(args.robots):
        cli.error("--robots must be distinct")
    if (
        not math.isfinite(args.seconds)
        or args.seconds <= 0
        or args.num_envs < 2
        or args.batch < 1
        or args.iterations < 501
    ):
        cli.error(
            "Require seconds > 0, num-envs >= 2, batch >= 1, iterations >= 501 (includes actor updates)"
        )
    workspace = args.workspace.resolve()
    source = workspace / ".cache/external/morphmimic-source"
    environment = workspace / ".cache/morphmimic-venv"
    archive = workspace / f".cache/morphmimic-downloads/{REVISION}.tar.gz"
    if args.command == "setup":
        setup(source, environment, archive)
        return 0
    if not (environment / "bin/python").exists() or not source.is_dir():
        cli.error("Run ./morphmimic setup first")
    if args.command == "train" and args.data is None:
        args.data = workspace / DEFAULT_DATA
    if args.command == "export" and args.checkpoint is None:
        cli.error(
            "export requires --checkpoint (checkpoint file or shared run directory)"
        )
    if args.command == "eval" and (args.data is None or args.policy is None):
        cli.error("eval requires --data and --policy")
    if args.command in ("train", "eval"):
        for robot in args.robots:
            if not (args.data / f"{robot}.yaml").is_file():
                cli.error(f"Missing prepared subset: {args.data / (robot + '.yaml')}")
        manifest = args.data / "result.json"
        if manifest.is_file():
            prepared = json.loads(manifest.read_text())
            if prepared.get("dataset") == "existing LAFAN1 G1":
                selected = [c for c in prepared["clips"] if c["robot"] in args.robots]
                if not selected or any(
                    "feasibility" not in c
                    or c["feasibility"]["feas_reject"] is not None
                    for c in selected
                ):
                    cli.error(
                        "LAFAN1 references lack passing feasibility checks; run prepare into a new --output directory"
                    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (
        args.output
        or workspace
        / (
            DEFAULT_DATA
            if args.command == "prepare"
            else f"runs/morphmimic-{args.command}-{stamp}"
        )
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(
        status="preparing",
        run_contract=run_identity("morphmimic", args.command, ",".join(args.robots)),
        workflow="morphmimic",
        command=args.command,
        output=str(output),
    )
    with run_lifecycle(record, lambda r: write_json(output / "run.json", r)):
        record["source"] = verify_source(source, archive)
        record["adapter_sha256"] = {
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("morphmimic.py", "_morphmimic_worker.py")
        }
        request = vars(args).copy()
        request.update(source=str(source), output=str(output))
        for key in ("data", "checkpoint", "policy", "resume"):
            request[key] = (
                str(getattr(args, key).resolve()) if getattr(args, key) else None
            )
        request["workspace"] = str(workspace)
        clips = args.clips or [
            workspace
            / f".cache/external/supermimic-reproduced/data/npz/lafan1/{name}.npz"
            for name in ("walk1_subject1", "run2_subject1", "dance1_subject1")
        ]
        request["clips"] = [str(p.resolve()) for p in clips]
        request["clip_names"] = [p.stem for p in clips]
        write_json(output / "request.json", request)
        env = child_environment(environment)
        env.update(MUJOCO_GL="egl", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        record["status"] = "running"
        write_json(output / "run.json", record)
        run_process(
            [
                str(environment / "bin/python"),
                "-I",
                str(Path(__file__).with_name("_morphmimic_worker.py")),
                str(output / "request.json"),
            ],
            cwd=source,
            env=env,
            log_path=output / "worker.log",
        )
        record["worker_exit_observed"] = True
        record["worker_returncode"] = 0
        record["result"] = json.loads((output / "result.json").read_text())
        verify_source(source, archive)
    print(f"{args.command}: complete — {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
