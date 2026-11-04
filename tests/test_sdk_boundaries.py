"""SDK-side format tools load without importing their host command launcher."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.mark.parametrize("name", ["_h1_worker.py", "_microduck_worker.py"])
def test_isolated_artifact_workers_import_without_site_packages(tmp_path, name):
    worker = Path(__file__).resolve().parents[1] / "src/embodiedforge" / name
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1]); "
            "assert not {'torch', 'numpy', 'mujoco', 'mjlab'} & sys.modules.keys()",
            str(worker),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_rlinf_worker_and_policy_formats_do_not_import_cli_or_sdk():
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        forbidden = {'embodiedforge.rlinf', 'torch', 'rlinf', 'ray', 'sapien',
                     'mani_skill', 'onnx', 'onnxruntime'}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in forbidden:
                    raise RuntimeError('Forbidden worker dependency: ' + fullname)
        sys.meta_path.insert(0, Block())
        from embodiedforge import _rlinf_worker, _rlinf_export, _rlinf_onnx
        from embodiedforge.workflows import rlinf_artifacts, rlinf_catalog
        assert _rlinf_worker.checkpoint_step is rlinf_artifacts.checkpoint_step
        assert _rlinf_export.REVISION == _rlinf_onnx.REVISION == rlinf_catalog.REVISION
        assert not forbidden & sys.modules.keys()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
