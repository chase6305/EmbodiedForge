"""Isolated worker bootstrapping only the repository-owned walking task."""

import importlib.metadata
import json
import runpy
import sys
from pathlib import Path

# -I omits the script directory (which contains logging.py) and PYTHONPATH.
# Add the package root explicitly, not the directory containing this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    try:
        importlib.metadata.distribution("mjlab-microduck")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "Native Microduck needs an environment without the upstream task plugin; run setup without --repo"
        )

    operation = sys.argv[1]
    # File diagnostics need no task registry, simulator, or actuator SDK. In
    # particular, progress must remain usable while the GPU is fully occupied.
    if operation not in {"progress", "metrics", "checkpoint", "onnx"}:
        from embodiedforge.locomotion.microduck import register

        register()
    if operation in ("train", "sdk-play"):
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        runpy.run_module(
            "mjlab.scripts.train" if operation == "train" else "mjlab.scripts.play",
            run_name="__main__",
        )
    elif operation == "export":
        from embodiedforge.locomotion.microduck.export import main as export

        export(sys.argv[2:])
    elif operation == "implementation":
        from embodiedforge._microduck_native import native_runtime

        print(json.dumps(native_runtime(), indent=2))
    else:
        runpy.run_module("embodiedforge._microduck_worker", run_name="__main__")


if __name__ == "__main__":
    main()
