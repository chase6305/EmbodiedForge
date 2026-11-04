"""Run optional motion SDKs without importing their dependencies into ef."""

import json
import logging
import os
import subprocess
from pathlib import Path

from .runtime.artifacts import write_json
from .runtime.environment import child_environment
from .runtime.hashing import copy_file_snapshot
from .runtime.hashing import file_sha256 as sha256
from .runtime.process import run_process, termination_signals
from .runtime.provenance import snapshot_implementation, validate_snapshot
from .runtime.records import failure_record, run_identity, run_lifecycle

LOGGER = logging.getLogger(__name__)


def checkout(repo, revision):
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise ValueError(f"Expected {repo} at {revision}, found {actual}")
    dirty = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        text=True,
    ).strip()
    if dirty:
        raise ValueError(f"Tracked changes in {repo}; use the pinned clean checkout")


def snapshot(source, destination):
    source = Path(source).resolve(strict=True)
    before = copy_file_snapshot(source, destination)
    if sha256(destination) != before or sha256(source) != before:
        raise ValueError(f"Input changed while copying: {source}")
    return {"source": str(source), "path": str(destination), "sha256": before}


@termination_signals()
def launch(backend, args, revision, options, sources):
    repo = args.root.expanduser().resolve(strict=True)
    python = args.python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError(f"SDK Python is not executable: {python}")
    checkout(repo, revision)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {
        "status": "preparing",
        "backend": backend,
        "run_contract": run_identity(
            backend,
            getattr(args, "command", "retarget" if backend == "gmr" else "evaluate"),
            options.get("robot"),
        ),
    }
    request = {
        "backend": backend,
        "repository": str(repo),
        "revision": revision,
        "python": str(python),
        "output": str(output),
        "options": options,
        "inputs": {},
    }
    with run_lifecycle(
        status,
        lambda record: write_json(output / "run.json", record),
        label=output / "run.json",
    ):
        try:
            inputs = output / "inputs"
            inputs.mkdir()
            for name, path in sources.items():
                request["inputs"][name] = snapshot(path, inputs / name)
            package = output / "implementation" / "embodiedforge"
            request["implementation"] = snapshot_implementation(
                Path(__file__).resolve().parent, package
            )
            request_path = output / "request.json"
            write_json(request_path, request)
            env = child_environment(python.parent.parent)
            env["PYTHONPATH"] = str(package.parent)
            env["OMP_NUM_THREADS"] = "1"
            status["status"] = "running"
            write_json(output / "run.json", status)
            run_process(
                [
                    str(python),
                    "-m",
                    f"embodiedforge._{backend}_worker",
                    str(request_path),
                ],
                cwd=output,
                env=env,
                log_path=output / "worker.log",
            )
            result = json.loads((output / "result.json").read_text())
            if result.get("status") != "complete":
                raise RuntimeError(f"{backend} worker did not finish")
            status["status"] = "complete"
        except BaseException as exc:
            failure = failure_record(exc)
            # run_process has stopped the worker before returning control here.
            # Preserve finished clips while closing a partially written evaluation.
            result_path = output / "result.json"
            try:
                if result_path.is_file():
                    result = json.loads(result_path.read_text())
                    if result.get("status") == "running":
                        result.update(status=failure["status"], error=failure["error"])
                        write_json(result_path, result)
            except BaseException:
                LOGGER.exception(
                    "Failed to finalize %s partial result %s; "
                    "preserving its original error",
                    backend,
                    result_path,
                )
            raise
    return output


def load_request(path):
    request = json.loads(Path(path).read_text())
    # Older runs predate implementation snapshots; retain their request format.
    if "implementation" in request:
        package = Path(request["output"]) / "implementation" / "embodiedforge"
        if Path(__file__).resolve().parent != package.resolve():
            raise ValueError("Motion SDK worker must load its implementation snapshot")
        validate_snapshot(
            package, request["implementation"], label="Motion SDK implementation"
        )
    checkout(Path(request["repository"]), request["revision"])
    for item in request["inputs"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Input snapshot changed: {item['path']}")
    return request
