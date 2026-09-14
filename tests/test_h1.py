"""H1 process isolation, failure reporting and checkpoint integrity."""

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from embodiedforge import h1


def test_child_environment_removes_caller_python_paths(monkeypatch):
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        monkeypatch.setenv(name, "/wrong/environment")
    result = h1.child_environment(Path("/selected"))
    assert all(
        name not in result for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")
    )
    assert result["CONDA_PREFIX"] == "/selected"
    assert result["PATH"].split(os.pathsep)[0] == "/selected/bin"
    assert os.environ["PYTHONPATH"] == "/wrong/environment"


def test_nonzero_exit_and_native_output_are_preserved(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as caught:
        h1.run_process(
            [sys.executable, "-c", "print('native diagnostic'); raise SystemExit(7)"],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=5,
        )
    assert caught.value.returncode == 7
    assert "native diagnostic" in (tmp_path / "console.log").read_text()


def test_timeout_remains_failure_when_child_handles_interrupt(tmp_path):
    child = """
import signal, time
from pathlib import Path
def stop(*args):
    Path('cleaned').touch()
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
time.sleep(30)
"""
    with pytest.raises(subprocess.TimeoutExpired):
        h1.run_process(
            [sys.executable, "-c", child],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=1,
        )
    assert (tmp_path / "cleaned").is_file()


def args_for(tmp_path):
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    return argparse.Namespace(
        repo=tmp_path,
        environment=environment,
        output=tmp_path / "output",
        resume_run=None,
        updates=5,
        num_envs=64,
        seed=0,
        timeout=10,
    )


@pytest.mark.parametrize(
    "error,status",
    [
        (subprocess.TimeoutExpired(["worker"], 1), "timed_out"),
        (subprocess.CalledProcessError(3, ["worker"]), "failed"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
def test_manifest_records_failure_and_completion_time(
    tmp_path, monkeypatch, error, status
):
    args = args_for(tmp_path)
    monkeypatch.setattr(h1, "source_identity", lambda repo: {"revision": h1.REVISION})

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(h1, "run_process", fail)
    with pytest.raises(type(error)):
        h1.train(args)
    manifest = json.loads((args.output / "run.json").read_text())
    assert manifest["status"] == status
    assert manifest["finished_at"] >= manifest["started_at"]
    assert "result" not in manifest


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    monkeypatch.setattr(h1, "source_identity", lambda repo: {})
    args.output.mkdir()
    (args.output / "run.json").write_text("original")
    with pytest.raises(FileExistsError):
        h1.train(args)
    assert (args.output / "run.json").read_text() == "original"


def make_resume(tmp_path):
    checkpoint = tmp_path / "model_4.pt"
    checkpoint.write_bytes(b"checkpoint fixture")
    manifest = {
        "schema": 1,
        "workflow": "h1_train",
        "task": h1.TASK,
        "physics": h1.PHYSICS,
        "source": {"revision": h1.REVISION},
        "status": "complete",
        "result": {
            "checkpoint": checkpoint.name,
            "checkpoint_iteration": 4,
            "checkpoint_sha256": h1.sha256(checkpoint),
        },
    }
    h1.write_json(tmp_path / "run.json", manifest)
    return checkpoint, manifest


def test_resume_rejects_changed_checkpoint(tmp_path):
    checkpoint, _ = make_resume(tmp_path)
    assert h1.resume_input(tmp_path)[0] == checkpoint
    checkpoint.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SHA256"):
        h1.resume_input(tmp_path)


@pytest.mark.parametrize("field,value", [("status", "failed"), ("task", "other")])
def test_resume_rejects_unverified_recipe(tmp_path, field, value):
    _, manifest = make_resume(tmp_path)
    manifest[field] = value
    h1.write_json(tmp_path / "run.json", manifest)
    with pytest.raises(ValueError, match="completed H1 run"):
        h1.resume_input(tmp_path)


def test_resume_rejects_path_outside_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    _, manifest = make_resume(run)
    (tmp_path / "outside.pt").write_bytes(b"checkpoint fixture")
    manifest["result"]["checkpoint"] = "../outside.pt"
    h1.write_json(run / "run.json", manifest)
    with pytest.raises(ValueError, match="outside"):
        h1.resume_input(run)


def test_sigterm_is_recorded_and_forwarded(tmp_path):
    # Exercise the actual main handler and process group with no GPU dependency.
    source = str(Path(h1.__file__).parents[1])
    script = f"""
import os, sys
from pathlib import Path
from embodiedforge import h1
h1.source_identity = lambda repo: {{'revision': h1.REVISION}}
original = h1.run_process
def worker(command, *, cwd, env, timeout):
    child = '''import os, signal, time
from pathlib import Path
def stop(*args):
    Path('cleaned').touch()
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
os.kill({{parent}}, signal.SIGTERM)
time.sleep(30)
'''.format(parent=os.getpid())
    original([sys.executable, '-c', child], cwd=cwd, env=env, timeout=5)
h1.run_process = worker
h1.main(['train', '--environment', sys.prefix, '--repo', '.', '--output', {str(tmp_path / "out")!r}])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": source},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 130, result.stderr
    assert (tmp_path / "out/cleaned").exists()
    manifest = json.loads((tmp_path / "out/run.json").read_text())
    assert manifest["status"] == "interrupted"
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


@pytest.mark.parametrize(
    "flag,value", [("--updates", "0"), ("--num-envs", "-1"), ("--timeout", "0")]
)
def test_invalid_training_bounds_fail_before_execution(flag, value):
    with pytest.raises(SystemExit) as caught:
        h1.main(["train", "--output", "unused", flag, value])
    assert caught.value.code == 2
