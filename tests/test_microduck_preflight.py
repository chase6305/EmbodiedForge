"""Preflight errors retain their original exception and useful headless context."""

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from embodiedforge import _microduck_worker as worker
from embodiedforge import microduck


def test_failed_preflight_preserves_reason_and_process_environment(
    tmp_path, monkeypatch
):
    def fail():
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(worker, "check", fail)
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.delenv("DISPLAY", raising=False)
    path = tmp_path / "runtime.json"
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        worker.runtime_check(path)
    assert not path.exists()
    report = json.loads(path.with_suffix(".failure.json").read_text())
    assert (
        report["error_type"] == "RuntimeError"
        and report["error"] == "CUDA out of memory"
    )
    assert report["process_environment"]["mujoco_gl"] == "egl"
    assert report["process_environment"]["display_present"] is False


def test_successful_preflight_writes_runtime_without_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "check", lambda: {"cuda_check": "passed"})
    path = tmp_path / "runtime.json"
    assert worker.runtime_check(path) == {"cuda_check": "passed"}
    assert json.loads(path.read_text()) == {"cuda_check": "passed"}
    assert not path.with_suffix(".failure.json").exists()


def test_failure_report_write_error_preserves_original_error(tmp_path, monkeypatch):
    def fail():
        raise RuntimeError("original error")

    monkeypatch.setattr(worker, "check", fail)
    with pytest.raises(RuntimeError, match="original error"):
        worker.runtime_check(tmp_path / "missing/runtime.json")

    # Invalid arguments fail before importing the simulator. Exercise the actual
    # evaluation entry point with both writable and unavailable report paths.
    for output in (tmp_path / "evaluation.json", tmp_path / "missing/evaluation.json"):
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                worker.__file__,
                "evaluate",
                "unused.pt",
                "invalid",
                "4",
                "0",
                str(output),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "ValueError: invalid literal" in result.stderr
        assert "During handling of the above exception" not in result.stderr
        if output.parent.exists():
            report = json.loads(output.with_suffix(".failure.json").read_text())
            assert report["error_type"] == "ValueError"
            assert "invalid literal" in report["error"]


def test_training_manifest_links_preflight_failure(tmp_path, monkeypatch):
    output = tmp_path / "run"

    def fail(command, *, cwd, env, log_path=None, echo=True):
        manifest = json.loads((cwd / "run.json").read_text())
        assert manifest["num_envs"] == 64 and manifest["iterations"] == 5
        (cwd / "runtime.failure.json").write_text(
            json.dumps({"error": "CUDA out of memory"})
        )
        raise RuntimeError("worker failed")

    monkeypatch.setattr(microduck, "run", fail)
    with pytest.raises(RuntimeError, match="worker failed"):
        microduck.train_run(
            Namespace(command="train", num_envs=64, iterations=5, output=output),
            Path("python"),
            {},
            {},
        )
    report = json.loads((output / "run.json").read_text())
    assert report["failed_phase"] == "check"
    assert report["runtime_failure"]["error"] == "CUDA out of memory"
