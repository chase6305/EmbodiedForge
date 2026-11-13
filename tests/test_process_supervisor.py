"""A published benchmark summary must not override its actual process exit."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SUPERVISOR = Path(__file__).resolve().parents[1] / "benchmarks/process_supervisor.py"
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX supervisor")


def run(tmp_path, code, *options):
    output = tmp_path / "supervision"
    result = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output",
            str(output),
            *options,
            "--",
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        timeout=15,
    )
    return result, json.loads((output / "result.json").read_text())


@pytest.mark.parametrize("exit_code", [0, 7, 143])
def test_observes_actual_exit_code(tmp_path, exit_code):
    completed, evidence = run(tmp_path, f"raise SystemExit({exit_code})")
    assert evidence["process_returncode"] == exit_code
    assert evidence["termination_signal"] is None
    assert evidence["exit_observed"]
    assert evidence["completed_successfully"] == (exit_code == 0)
    assert completed.returncode == exit_code


@pytest.mark.parametrize("trace", [False, True])
def test_complete_summary_then_sigterm_remains_failure(tmp_path, trace):
    if trace and not shutil.which("strace"):
        pytest.skip("strace is optional")
    summary = tmp_path / "summary.json"
    completed, evidence = run(
        tmp_path,
        "import os,signal;from pathlib import Path;"
        f"Path({str(summary)!r}).write_text('{{\"all_scenarios_passed\":true}}');"
        "os.kill(os.getpid(),signal.SIGTERM)",
        *(["--trace"] if trace else []),
    )
    assert json.loads(summary.read_text())["all_scenarios_passed"]
    assert evidence["process_returncode"] == -signal.SIGTERM
    assert evidence["termination_signal"] == signal.SIGTERM
    assert not evidence["completed_successfully"]
    assert evidence["exit_observed"]
    assert completed.returncode == 143
    if trace:
        kernel = (tmp_path / "supervision/signals.strace").read_text()
        assert "si_pid=" in kernel
        assert "killed by SIGTERM" in kernel


def test_timeout_kills_and_reaps_worker_ignoring_termination(tmp_path):
    ready = tmp_path / "ready"
    completed, evidence = run(
        tmp_path,
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(60)",
        "--timeout",
        "1",
        "--grace",
        "0.1",
    )
    assert evidence["timed_out"]
    assert evidence["forced_stop"]
    assert evidence["process_returncode"] == -signal.SIGKILL
    assert evidence["exit_observed"]
    assert not evidence["completed_successfully"]
    assert completed.returncode == 124
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)


def test_failed_exec_publishes_failure_without_inventing_child_exit(tmp_path):
    output = tmp_path / "supervision"
    completed = subprocess.run(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output",
            str(output),
            "--",
            str(tmp_path / "missing-command"),
        ],
        capture_output=True,
        timeout=10,
    )
    evidence = json.loads((output / "result.json").read_text())
    assert completed.returncode == 125
    assert evidence["supervisor_error"]["type"] == "FileNotFoundError"
    assert evidence["process_returncode"] is None
    assert not evidence["exit_observed"]
    assert not evidence["completed_successfully"]


def test_supervisor_cancellation_is_failure_even_when_worker_cleans_up(tmp_path):
    output = tmp_path / "supervision"
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    worker = (
        "import signal,time;from pathlib import Path\n"
        "def stop(*args):\n"
        f"    Path({str(cleaned)!r}).touch()\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        f"Path({str(ready)!r}).touch()\n"
        "while True: time.sleep(.01)\n"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            str(SUPERVISOR),
            "--output",
            str(output),
            "--",
            sys.executable,
            "-c",
            worker,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert parent.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        parent.send_signal(signal.SIGTERM)
        stdout, stderr = parent.communicate(timeout=5)
        assert parent.returncode == 143, (stdout, stderr)
        evidence = json.loads((output / "result.json").read_text())
        assert evidence["supervisor_signal"] == signal.SIGTERM
        assert evidence["process_returncode"] == 0
        assert evidence["exit_observed"]
        assert not evidence["timed_out"]
        assert not evidence["completed_successfully"]
        assert cleaned.exists()
        with pytest.raises(ProcessLookupError):
            os.kill(evidence["pid"], 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.communicate(timeout=5)
        evidence_path = output / "started.json"
        if evidence_path.exists() and ready.exists() and not cleaned.exists():
            # The supervisor's worker has its own group; never leave a failed
            # test fixture behind if the supervisor itself could not finish.
            for event in (output / "events.jsonl").read_text().splitlines():
                data = json.loads(event)
                if data["kind"] == "popen_after":
                    try:
                        os.killpg(data["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def test_cancellation_after_wait_during_cleanup_is_not_success(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("process_supervisor", SUPERVISOR)
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    original = os.killpg

    def cancel_during_cleanup(pid, sig):
        assert sig == signal.SIGKILL
        os.kill(os.getpid(), signal.SIGTERM)
        original(pid, sig)

    monkeypatch.setattr(os, "killpg", cancel_during_cleanup)
    evidence = supervisor.supervise(
        [sys.executable, "-c", "pass"], tmp_path / "supervision"
    )
    assert evidence["process_returncode"] == 0
    assert evidence["exit_observed"]
    assert evidence["supervisor_signal"] == signal.SIGTERM
    assert not evidence["completed_successfully"]


@pytest.mark.parametrize("echo", [False, True])
def test_startup_probe_records_early_worker_exit_and_reaps_without_pid_marker(
    tmp_path, monkeypatch, echo
):
    path = SUPERVISOR.with_name("process_fault_soak.py")
    spec = importlib.util.spec_from_file_location("process_fault_soak", path)
    faults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(faults)
    original = subprocess.Popen
    children = []

    class EarlyExit(original):
        def __init__(self, command, **kwargs):
            super().__init__([sys.executable, "-c", "raise SystemExit(23)"], **kwargs)
            children.append(self)

    monkeypatch.setattr(subprocess, "Popen", EarlyExit)
    output = tmp_path / "case"
    with pytest.raises(RuntimeError, match="worker exited before ready"):
        faults.startup(output, signal.SIGTERM, "after", echo)
    evidence = json.loads((output / "startup-error.json").read_text())
    assert evidence == {
        "pid": children[0].pid,
        "returncode": 23,
        "condition": "worker_exited_before_ready",
    }
    assert not (output / "child.pid").exists()
    assert children[0].returncode == 23
    assert children[0].stdout is None or children[0].stdout.closed
    with pytest.raises(ChildProcessError):
        os.waitpid(children[0].pid, os.WNOHANG)


@pytest.mark.parametrize("first", ["interrupt", "timeout"])
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("stage", ["forward", "kill", "reap"])
def test_repeated_cancellation_probe_keeps_first_manifest_status(
    tmp_path, first, echo, stage
):
    path = SUPERVISOR.with_name("process_fault_soak.py")
    spec = importlib.util.spec_from_file_location("process_fault_soak", path)
    faults = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(faults)
    faults.repeated_cancellation(tmp_path / "case", first, echo, stage)
