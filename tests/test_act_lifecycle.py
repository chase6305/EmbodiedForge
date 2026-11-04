"""ACT launcher ownership and terminal records without installing LeRobot."""

import json
import os
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from embodiedforge import act
from embodiedforge.runtime.process import ProcessInterrupted


@pytest.fixture
def act_runtime(monkeypatch):
    monkeypatch.setattr(act, "_require_lerobot", lambda: None)
    monkeypatch.setattr(act, "runtime_report", lambda: {"implementation": "fixture"})
    monkeypatch.setattr(act, "version", lambda name: "fixture")
    monkeypatch.setitem(
        sys.modules,
        "lerobot.policies.act.modeling_act",
        SimpleNamespace(ACTPolicy=object),
    )


@pytest.mark.parametrize(
    "failure,status",
    [
        (subprocess.CalledProcessError(7, ["trainer"]), "failed"),
        (subprocess.TimeoutExpired(["trainer"], 1), "timed_out"),
        (ProcessInterrupted(signal.SIGTERM), "interrupted"),
    ],
)
def test_act_records_cancellation_and_timeout_and_allows_recovery(
    tmp_path, monkeypatch, act_runtime, failure, status
):
    output = tmp_path / "training"

    def run(command, *, cwd, env, pass_fds):
        assert cwd.is_dir() and env == os.environ.copy()
        assert len(pass_fds) == 1
        os.fstat(pass_fds[0])
        output.mkdir()
        raise failure

    monkeypatch.setattr(act, "run_process", run)
    with pytest.raises(type(failure)) as caught:
        act._execute_training(output, {}, ["trainer"])
    assert caught.value is failure
    startup, _ = act._attempt_paths(output)
    saved = json.loads(startup.read_text())
    assert saved == json.loads((output / act.RUN_MANIFEST).read_text())
    assert saved["status"] == status
    assert saved["finished_at"] >= saved["started_at"]
    assert act._resume_report(output)["status"] == status
    if status == "interrupted":
        assert saved["interrupt_signal"] == signal.SIGTERM


def test_act_child_inherits_lock_and_preserves_exit_code(tmp_path, act_runtime):
    output = tmp_path / "training with spaces"
    command = [
        sys.executable,
        "-c",
        "import fcntl,sys;from pathlib import Path;"
        "lock=Path(sys.argv[1]).open('a');"
        "\ntry: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)"
        "\nexcept BlockingIOError: sys.exit(7)"
        "\nraise AssertionError('trainer did not inherit the run lock')",
        str(act._attempt_paths(output)[1]),
    ]
    with pytest.raises(subprocess.CalledProcessError) as caught:
        act._execute_training(output, {}, command)
    assert caught.value.returncode == 7
    assert act._resume_report(output)["status"] == "failed"


def test_act_final_record_error_does_not_hide_worker_failure(
    tmp_path, monkeypatch, act_runtime, caplog
):
    output = tmp_path / "training"
    primary = RuntimeError("trainer crashed")
    original = act._write_json

    def publish(path, record):
        if record["status"] == "failed":
            raise OSError("disk full")
        original(path, record)

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(act, "_write_json", publish)
    monkeypatch.setattr(act, "run_process", fail)
    with pytest.raises(RuntimeError) as caught:
        act._execute_training(output, {}, ["trainer"])
    assert caught.value is primary
    assert "preserving its original error" in caplog.text
