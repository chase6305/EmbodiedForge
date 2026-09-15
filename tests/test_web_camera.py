"""Run browser-control logic with Node's built-in runner, without npm packages."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_web_camera_interactions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the Web camera interaction tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("web_camera.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
