"""Inherited worker stdout cannot keep the launcher blocked during teardown."""

import errno
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from embodiedforge.runtime import process
from embodiedforge.runtime.records import managed_run, read_run, run_identity


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="Requires /dev/full")
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout"])
def test_full_log_preserves_capture_error_or_timeout(tmp_path, caplog, outcome):
    worker = (
        "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "os.write(1,b'worker output\\n');"
        + (
            "time.sleep(30)"
            if outcome == "timeout"
            else f"raise SystemExit({7 if outcome == 'failure' else 0})"
        )
    )
    expected = subprocess.TimeoutExpired if outcome == "timeout" else RuntimeError
    with pytest.raises(expected) as caught:
        with managed_run(tmp_path, run_identity("probe", "disk-full")):
            process.run_process(
                [sys.executable, "-c", worker],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=Path("/dev/full"),
                append_log=True,
                timeout=0.15 if outcome == "timeout" else 3,
            )
    if outcome != "timeout":
        assert "Failed to capture training log" in str(caught.value)
        assert isinstance(caught.value.__cause__, OSError)
        assert caught.value.__cause__.errno == errno.ENOSPC
    assert read_run(tmp_path)["status"] == (
        "timed_out" if outcome == "timeout" else "failed"
    )
    assert "Could not close training log /dev/full" in caplog.text


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="Requires /dev/full")
def test_full_log_close_cannot_replace_launcher_cancellation(
    tmp_path, monkeypatch, caplog
):
    original = os.killpg

    def cancel_on_capture_failure(pid, sig):
        original(pid, sig)
        if sig == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(process.os, "killpg", cancel_on_capture_failure)
    with pytest.raises(process.ProcessInterrupted) as caught:
        with managed_run(tmp_path, run_identity("probe", "disk-full-interrupt")):
            process.run_process(
                [
                    sys.executable,
                    "-c",
                    "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                    "os.write(1,b'worker output\\n');time.sleep(30)",
                ],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=Path("/dev/full"),
                append_log=True,
                timeout=3,
            )
    assert caught.value.signum == signal.SIGTERM
    assert read_run(tmp_path)["status"] == "interrupted"
    assert "Could not close training log /dev/full" in caplog.text


@pytest.mark.parametrize("echo", [False, True])
def test_independent_log_close_failure_in_a_retry_is_reported(
    tmp_path, monkeypatch, echo
):
    log = tmp_path / "worker.log"
    original = Path.open
    failure = OSError("log close failed")

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            self.stream.close()
            raise failure

    def open_log(path, *args, **kwargs):
        stream = original(path, *args, **kwargs)
        return FailingClose(stream) if path == log and stream.writable() else stream

    monkeypatch.setattr(Path, "open", open_log)
    try:
        raise ValueError("previous attempt already handled")
    except ValueError:
        with pytest.raises(OSError) as caught:
            with managed_run(tmp_path, run_identity("probe", "retry-log-close")):
                process.run_process(
                    [sys.executable, "-c", "print('worker finished')"],
                    cwd=tmp_path,
                    env=os.environ.copy(),
                    log_path=log,
                    echo=echo,
                )
    assert caught.value is failure
    assert read_run(tmp_path)["status"] == "failed"
    assert log.read_text() == "worker finished\n"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper fixture")
@pytest.mark.parametrize("worker_code", [0, 7, "timeout"])
def test_detached_stdout_holder_does_not_block_log_cleanup(tmp_path, worker_code):
    descendant = tmp_path / "descendant.pid"
    holder = textwrap.dedent(f"""
        import os,time
        from pathlib import Path
        Path({str(descendant)!r}).write_text(str(os.getpid()))
        time.sleep(30)
    """)
    worker = textwrap.dedent(f"""
        import subprocess,sys,time
        from pathlib import Path
        subprocess.Popen([sys.executable,'-c',{holder!r}], start_new_session=True)
        while not Path({str(descendant)!r}).exists(): time.sleep(.005)
        print('worker output',flush=True)
        if {worker_code!r}=='timeout': time.sleep(30)
        raise SystemExit({worker_code!r})
    """)
    wrapper = textwrap.dedent(f"""
        import ctypes,os,signal,subprocess,sys,threading
        from pathlib import Path
        sys.path.insert(0,{str(Path(__file__).resolve().parents[1] / "src")!r})
        from embodiedforge.runtime import process
        assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
        # Shorten only the reader grace period; a stuck close still blocks.
        original = process.threading.Thread
        class Reader(original):
            def join(self,timeout=None): super().join(timeout=.2)
        process.threading.Thread=Reader
        try:
            try:
                process.run_process([sys.executable,'-c',{worker!r}],
                    cwd=Path({str(tmp_path)!r}),env=os.environ.copy(),
                    log_path=Path({str(tmp_path / "worker.log")!r}),
                    timeout=.5 if {worker_code!r}=='timeout' else 3)
            except subprocess.CalledProcessError as error:
                assert error.returncode=={worker_code!r} and {worker_code!r}!=0
            except subprocess.TimeoutExpired:
                assert {worker_code!r}=='timeout'
            else:
                assert {worker_code!r}==0
            assert threading.enumerate()==[threading.main_thread()]
            # The escaped descendant is outside the launcher's owned group.
            os.kill(int(Path({str(descendant)!r}).read_text()),0)
        finally:
            pid=int(Path({str(descendant)!r}).read_text())
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass
            os.waitpid(pid,0)
    """)
    owner = subprocess.Popen(
        [sys.executable, "-c", wrapper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    stalled = False
    try:
        try:
            stdout, stderr = owner.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stalled = True
            if descendant.exists():
                try:
                    os.kill(int(descendant.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout, stderr = owner.communicate(timeout=5)
    finally:
        if owner.poll() is None:
            os.killpg(owner.pid, signal.SIGKILL)
            owner.communicate(timeout=5)
    assert not stalled, f"Log cleanup blocked on detached stdout: {stderr}"
    assert owner.returncode == 0, (stdout, stderr)
    assert (tmp_path / "worker.log").read_text().startswith("worker output\n")
    with pytest.raises(ProcessLookupError):
        os.kill(int(descendant.read_text()), 0)


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="Requires /dev/full")
def test_cleanup_cancellation_preserves_observed_capture_failure(tmp_path, monkeypatch):
    import threading

    reported = threading.Event()
    original_popen, original_killpg = subprocess.Popen, os.killpg
    children = []
    cancelled = False

    class Observed(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            children.append(self)

        def wait(self, timeout=None):
            code = super().wait(timeout=timeout)
            # The reader appends its capture error before sending SIGTERM.
            assert reported.wait(3), "capture failure was not observed"
            return code

    def killpg(pid, sig):
        nonlocal cancelled
        if sig == signal.SIGTERM:
            reported.set()
        elif sig == signal.SIGKILL and not cancelled:
            cancelled = True
            os.kill(os.getpid(), signal.SIGTERM)
        original_killpg(pid, sig)

    monkeypatch.setattr(process.subprocess, "Popen", Observed)
    monkeypatch.setattr(process.os, "killpg", killpg)
    with pytest.raises(BaseException) as caught:
        with managed_run(tmp_path, run_identity("probe", "capture-cleanup-cancel")):
            process.run_process(
                [
                    sys.executable,
                    "-c",
                    "import os,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);os.write(1,b'output')",
                ],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=Path("/dev/full"),
                append_log=True,
                echo=True,
            )
    assert isinstance(caught.value, RuntimeError)
    assert "Failed to capture training log" in str(caught.value)
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.__cause__.errno == errno.ENOSPC
    assert cancelled and children[0].returncode == 0
    assert read_run(tmp_path)["status"] == "failed"
    assert threading.enumerate() == [threading.main_thread()]
