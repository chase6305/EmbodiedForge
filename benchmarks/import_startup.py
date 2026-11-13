"""Measure fresh-process import/inspection costs without preloading NumPy.

Run each source tree with the same interpreter and environment. Measurements
include interpreter launch and normal exit; child-side timings cover the import
or inspection itself. Linux thread counts and loaded modules expose dependencies.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHILD = """
import contextlib, importlib, io, json, sys, time
from pathlib import Path
began = time.perf_counter()
case = sys.argv[1]
if case == 'runs-inspect':
    from embodiedforge.cli import main
    sys.argv = ['embodiedforge', 'runs', 'inspect', '--run', sys.argv[2]]
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        main()
    assert json.loads(captured.getvalue())['status'] == 'complete'
elif case.startswith('help:'):
    from embodiedforge.cli import main
    sys.argv = ['embodiedforge', case.split(':')[1], '--help']
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        try:
            main()
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('help must exit normally')
    assert 'usage:' in captured.getvalue()
else:
    importlib.import_module(case)
seconds = time.perf_counter() - began
threads = None
rss_bytes = None
status = Path('/proc/self/status')
if status.is_file():
    fields = dict(line.split(':', 1) for line in status.read_text().splitlines())
    threads = int(fields['Threads'])
    rss_bytes = int(fields['VmRSS'].split()[0]) * 1024
print(json.dumps({'seconds': seconds, 'threads': threads,
                  'rss_bytes': rss_bytes,
                  'numpy_loaded': 'numpy' in sys.modules,
                  'environment_loaded': 'embodiedforge.env' in sys.modules,
                  'modules': len(sys.modules)}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("trials must be positive")
    source = args.source.resolve()
    if not (source / "embodiedforge/__init__.py").is_file():
        parser.error("source must contain the embodiedforge package")
    cases = [
        "embodiedforge",
        "embodiedforge.runtime.process",
        "embodiedforge.runtime.records",
        "runs-inspect",
        *(
            f"help:{name}"
            for name in ("recipes", "microduck", "h1", "rlinf", "gmr", "sonic")
        ),
    ]
    samples = {case: [] for case in cases}
    scheduled = cases * args.trials
    random.Random(17).shuffle(scheduled)
    with tempfile.TemporaryDirectory(prefix="embodiedforge-import-") as temporary:
        Path(temporary, "run.json").write_text(
            json.dumps(
                {
                    "run_contract": {
                        "schema_version": 1,
                        "workflow": "fixture",
                        "operation": "train",
                        "task": None,
                    },
                    "status": "complete",
                }
            )
        )
        for case in scheduled:
            began = time.perf_counter()
            result = subprocess.run(
                [sys.executable, "-c", CHILD, case, temporary],
                env={**os.environ, "PYTHONPATH": str(source)},
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=20,
                check=True,
            )
            elapsed = time.perf_counter() - began
            sample = json.loads(result.stdout)
            sample["launch_to_exit_seconds"] = elapsed
            samples[case].append(sample)
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "source": str(source),
        "trials": args.trials,
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "embodiedforge/__init__.py",
                "embodiedforge/cli.py",
                "embodiedforge/_go1_metrics.py",
                "embodiedforge/workflows/recipe_catalog.py",
                "embodiedforge/workflows/recipe_validation.py",
            )
        },
        "cases": {
            case: {
                "median_launch_to_exit_seconds": statistics.median(
                    sample["launch_to_exit_seconds"] for sample in values
                ),
                "median_import_seconds": statistics.median(
                    sample["seconds"] for sample in values
                ),
                "samples": values,
            }
            for case, values in samples.items()
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                case: value["median_launch_to_exit_seconds"]
                for case, value in report["cases"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
