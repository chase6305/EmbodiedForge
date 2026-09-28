"""Run optional motion SDKs without importing their dependencies into ef."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ._microduck_process import run_process, termination_signals
from .h1 import child_environment, sha256, write_json
from .recipes import snapshot_implementation, validate_snapshot


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
    before = sha256(source)
    shutil.copyfile(source, destination)
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
    status = {"status": "preparing", "backend": backend}
    request = {
        "backend": backend,
        "repository": str(repo),
        "revision": revision,
        "python": str(python),
        "output": str(output),
        "options": options,
        "inputs": {},
    }
    try:
        write_json(output / "run.json", status)
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
            [str(python), "-m", f"embodiedforge._{backend}_worker", str(request_path)],
            cwd=output,
            env=env,
            log_path=output / "worker.log",
        )
        result = json.loads((output / "result.json").read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"{backend} worker did not finish")
        status["status"] = "complete"
    except BaseException as exc:
        status.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=str(exc),
        )
        # run_process has stopped the worker before returning control here.
        # Preserve finished clips while closing a partially written evaluation.
        result_path = output / "result.json"
        try:
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result.get("status") == "running":
                    result.update(status=status["status"], error=status["error"])
                    write_json(result_path, result)
        except Exception as finalization_error:
            print(
                f"Failed to finalize {backend} partial result: {finalization_error}",
                file=sys.stderr,
            )
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            write_json(output / "run.json", status)
        except Exception as exc:
            if not active_error:
                raise
            print(f"Failed to finalize {backend} run: {exc}", file=sys.stderr)
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
