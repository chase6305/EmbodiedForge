"""Real child startup failures must not hang while unwinding Popen ownership."""

import json
import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from embodiedforge.runtime import process


@pytest.mark.parametrize(
    "timeout", [float("nan"), float("inf"), -float("inf"), -1, True, "1", 10**1000]
)
def test_invalid_timeout_does_not_spawn_worker_or_claim_log(tmp_path, timeout):
    marker = tmp_path / "worker-started"
    log = tmp_path / "worker.log"
    with pytest.raises(ValueError, match="timeout must be finite and nonnegative"):
        process.run_process(
            [
                sys.executable,
                "-c",
                f"from pathlib import Path;Path({str(marker)!r}).touch()",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout=timeout,
            log_path=log,
        )
    assert not marker.exists()
    assert not log.exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process ownership")
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGHUP])
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("handled_error", [False, True])
def test_cancellation_during_popen_return_reaps_the_started_child(
    tmp_path, monkeypatch, signum, echo, handled_error
):
    marker = tmp_path / "child.pid"
    ready = tmp_path / "ready"
    worker = textwrap.dedent(f"""
        import os,signal,time
        from pathlib import Path
        Path({str(marker)!r}).write_text(str(os.getpid()))
        def stop(*args):
            Path({str(tmp_path / "cleaned")!r}).touch()
            raise SystemExit(0)
        signal.signal(signal.SIGINT,stop)
        Path({str(ready)!r}).touch()
        # Bound Python handler latency even when SIGINT arrives just before
        # the blocking sleep syscall; this fixture tests graceful cancellation.
        for _ in range(600):
            time.sleep(0.05)
    """)
    original = subprocess.Popen
    saved = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    class InterruptOnReturn(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            import time

            deadline = time.monotonic() + 3
            while not ready.exists():
                if time.monotonic() >= deadline:
                    raise AssertionError("worker did not become ready")
                time.sleep(0.005)
            os.kill(os.getpid(), signum)

    monkeypatch.setattr(process.subprocess, "Popen", InterruptOnReturn)

    def launch():
        process.run_process(
            [sys.executable, "-c", worker],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=tmp_path / "worker.log",
            echo=echo,
        )

    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            if handled_error:
                try:
                    raise ValueError("previous attempt already handled")
                except ValueError:
                    launch()
            else:
                launch()
        assert getattr(caught.value, "signum", signal.SIGINT) == signum
        assert (tmp_path / "cleaned").exists()
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
        assert {sig: signal.getsignal(sig) for sig in saved} == saved
    finally:
        if marker.exists():
            pid = int(marker.read_text())
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal masks")
def test_startup_cancellation_protection_does_not_block_worker_signals(tmp_path):
    mask = tmp_path / "mask.json"
    inherited = set(signal.pthread_sigmask(signal.SIG_BLOCK, []))
    process.run_process(
        [
            sys.executable,
            "-c",
            "import json,signal;from pathlib import Path;"
            f"Path({str(mask)!r}).write_text(json.dumps(sorted(signal.pthread_sigmask(signal.SIG_BLOCK,[]))))",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=tmp_path / "worker.log",
    )
    assert set(json.loads(mask.read_text())) == inherited


@pytest.mark.skipif(sys.platform != "linux", reason="Linux SIGSTOP fixture")
def test_stopped_worker_is_force_killed_and_reaped_after_startup_cancellation(
    tmp_path, monkeypatch
):
    import time

    ready = tmp_path / "ready"
    original = subprocess.Popen
    started = []

    class Stopped(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            started.append(self)
            deadline = time.monotonic() + 3
            while not ready.exists():
                if time.monotonic() >= deadline:
                    raise AssertionError("worker did not become ready")
                time.sleep(0.005)
            os.kill(self.pid, signal.SIGSTOP)
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(process.subprocess, "Popen", Stopped)
    try:
        with pytest.raises(process.ProcessInterrupted) as caught:
            process.run_process(
                [
                    sys.executable,
                    "-c",
                    f"import time;from pathlib import Path;Path({str(ready)!r}).touch();time.sleep(30)",
                ],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=tmp_path / "worker.log",
            )
        assert caught.value.signum == signal.SIGTERM
        assert started[0].returncode == -signal.SIGKILL
        with pytest.raises(ProcessLookupError):
            os.kill(started[0].pid, 0)
    finally:
        for child in started:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process ownership")
@pytest.mark.parametrize("stage", ["construct", "start", "started"])
def test_log_thread_startup_failure_stops_and_reaps_the_owned_child(tmp_path, stage):
    ready = tmp_path / "child.pid"
    child = (
        "import os,time;from pathlib import Path;"
        f"Path({str(ready)!r}).write_text(str(os.getpid()));time.sleep(30)"
    )
    wrapper = textwrap.dedent(f"""
        import os,sys,time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        from embodiedforge.runtime import process
        original = process.threading.Thread
        primary = RuntimeError('log thread startup failed')
        def wait_ready():
            deadline=time.monotonic()+3
            while not Path({str(ready)!r}).is_file():
                if time.monotonic()>deadline: raise AssertionError('child not ready')
                time.sleep(.01)
        class Thread(original):
            def __init__(self, *args, **kwargs):
                if {stage!r}=='construct':
                    wait_ready()
                    raise primary
                super().__init__(*args,**kwargs)
            def start(self):
                if {stage!r}=='started': super().start()
                wait_ready()
                raise primary
        process.threading.Thread=Thread
        try:
            process.run_process([sys.executable,'-c',{child!r}],cwd=Path({str(tmp_path)!r}),
                env=os.environ.copy(),log_path=Path({str(tmp_path / "output.log")!r}))
        except RuntimeError as error:
            assert error is primary, repr(error)
        else:
            raise AssertionError('startup error was hidden')
        assert not any(t.is_alive() for t in process.threading.enumerate()
                       if t is not process.threading.main_thread())
        print('original error preserved; reader stopped; child reaped')
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
            stdout, stderr = owner.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            stalled = True
            # Reap the baseline failure's child through its still-owning wrapper.
            if ready.exists():
                try:
                    os.killpg(int(ready.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            stdout, stderr = owner.communicate(timeout=3)
    finally:
        if owner.poll() is None:
            os.killpg(owner.pid, signal.SIGKILL)
            owner.communicate(timeout=3)
    assert not stalled, f"Startup failure left Popen waiting for its child: {stderr}"
    assert owner.returncode == 0, stderr
    assert "child reaped" in stdout
    with pytest.raises(ProcessLookupError):
        os.kill(int(ready.read_text()), 0)


@pytest.mark.skipif(os.name != "posix", reason="POSIX termination handlers")
@pytest.mark.parametrize("after_install", [False, True])
def test_partial_termination_handler_setup_restores_previous_handlers(
    monkeypatch, after_install
):
    original = signal.signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    primary = OSError("handler installation failed")
    injected = False

    def fail(sig, handler):
        nonlocal injected
        if sig == signal.SIGHUP and not injected:
            injected = True
            if after_install:
                original(sig, handler)
            raise primary
        return original(sig, handler)

    try:
        original(signal.SIGHUP, signal.SIG_DFL)
        monkeypatch.setattr(process.signal, "signal", fail)
        with pytest.raises(OSError) as caught:
            with process.termination_signals():
                pytest.fail("Partial setup entered the protected body")
        assert caught.value is primary
        assert signal.getsignal(signal.SIGTERM) == saved[signal.SIGTERM]
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
    finally:
        for sig, handler in saved.items():
            original(sig, handler)


@pytest.mark.skipif(os.name != "posix", reason="POSIX termination handlers")
def test_restore_failure_during_retry_is_not_hidden_by_a_handled_outer_error(
    monkeypatch,
):
    original = signal.signal
    previous = signal.getsignal(signal.SIGTERM)
    failure = OSError("signal restore failed during retry")

    def restore(sig, handler):
        result = original(sig, handler)
        if sig == signal.SIGTERM and handler == previous:
            raise failure
        return result

    monkeypatch.setattr(process.signal, "signal", restore)
    try:
        raise ValueError("previous attempt already handled")
    except ValueError:
        with pytest.raises(OSError) as caught:
            with process.termination_signals():
                pass
    assert caught.value is failure
    assert signal.getsignal(signal.SIGTERM) == previous


@pytest.mark.skipif(os.name != "posix", reason="POSIX termination handlers")
@pytest.mark.parametrize(
    "primary", [None, RuntimeError("worker failed"), KeyboardInterrupt()]
)
@pytest.mark.parametrize("after_restore", [False, True])
def test_signal_restore_failure_attempts_all_handlers_and_keeps_primary_error(
    monkeypatch, caplog, primary, after_restore
):
    original = signal.signal
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    restore_error = OSError("signal restore failed")
    restored = []

    def fail(sig, handler):
        if handler == signal.SIG_DFL:
            restored.append(sig)
            if sig == signal.SIGTERM:
                if after_restore:
                    original(sig, handler)
                raise restore_error
        return original(sig, handler)

    try:
        for sig in saved:
            original(sig, signal.SIG_DFL)
        monkeypatch.setattr(process.signal, "signal", fail)
        expected = primary if primary is not None else restore_error
        with pytest.raises(type(expected)) as caught:
            with process.termination_signals():
                if primary is not None:
                    raise primary
        assert caught.value is expected
        assert restored == [signal.SIGTERM, signal.SIGHUP]
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_DFL
        assert "signal restore failed" in caplog.text
    finally:
        for sig, handler in saved.items():
            original(sig, handler)


@pytest.mark.skipif(os.name != "posix", reason="POSIX startup cancellation")
@pytest.mark.parametrize("output_size", [64, 131072])
def test_startup_cancellation_drains_worker_shutdown_output(
    tmp_path, monkeypatch, capsys, output_size
):
    import threading
    import time

    ready = tmp_path / "ready"
    chunk = b"worker shutdown\xff\n"
    payload = chunk * output_size
    worker = textwrap.dedent(f"""
        import os,signal,time
        from pathlib import Path
        def stop(*args):
            data={chunk!r}*{output_size}
            while data:
                data=data[os.write(1,data):]
            Path('cleaned').touch()
            raise SystemExit(0)
        signal.signal(signal.SIGINT,stop)
        Path('ready').touch()
        for _ in range(600): time.sleep(.05)
    """)
    original = subprocess.Popen
    children = []

    class CancelOnReturn(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            children.append(self)
            deadline = time.monotonic() + 3
            while not ready.exists():
                assert time.monotonic() < deadline, "worker not ready"
                time.sleep(0.005)
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(process.subprocess, "Popen", CancelOnReturn)
    try:
        with pytest.raises(process.ProcessInterrupted) as caught:
            process.run_process(
                [sys.executable, "-c", worker],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=tmp_path / "worker.log",
            )
        assert caught.value.signum == signal.SIGTERM
        assert children[0].returncode == 0
        assert (tmp_path / "cleaned").exists()
        assert (tmp_path / "worker.log").read_bytes() == payload
        assert capsys.readouterr().out == payload.decode("utf-8", errors="replace")
        assert threading.enumerate() == [threading.main_thread()]
        with pytest.raises(ProcessLookupError):
            os.kill(children[0].pid, 0)
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper fixture")
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("primary_failure", [False, True])
@pytest.mark.parametrize("cancel_stage", ["group", "handler_setup"])
def test_cancellation_during_group_cleanup_still_reaps_owned_processes(
    tmp_path, echo, primary_failure, cancel_stage
):
    descendant = tmp_path / "descendant.pid"
    worker = textwrap.dedent(f"""
        import os,subprocess,sys,time
        from pathlib import Path
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
        Path({str(descendant) + ".tmp"!r}).write_text(str(child.pid))
        os.replace({str(descendant) + ".tmp"!r},{str(descendant)!r})
    """)
    wrapper = textwrap.dedent(f"""
        import ctypes,os,signal,subprocess,sys,threading,time
        from pathlib import Path
        sys.path.insert(0,{str(Path(__file__).resolve().parents[1] / "src")!r})
        from embodiedforge.runtime import process
        assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
        marker=Path({str(descendant)!r})
        original_killpg,original_popen=os.killpg,subprocess.Popen
        original_signal=signal.signal
        primary=RuntimeError('independent wait failure')
        signalled=False
        failed=False
        children=[]
        deferrals=0
        saved={{s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}}
        def cancel_once(pid,sig):
            global signalled
            if {cancel_stage!r}=='group' and sig==signal.SIGKILL and not signalled:
                signalled=True
                os.kill(os.getpid(),signal.SIGTERM)
            original_killpg(pid,sig)
        def install(sig,handler):
            global signalled,deferrals
            result=original_signal(sig,handler)
            if sig==signal.SIGINT and getattr(handler,'__name__',None)=='defer':
                deferrals+=1
                if {cancel_stage!r}=='handler_setup' and deferrals==2:
                    signalled=True
                    os.kill(os.getpid(),signal.SIGTERM)
            return result
        class Child(original_popen):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                children.append(self)
            def wait(self,timeout=None):
                global failed
                if {primary_failure!r} and not failed:
                    failed=True
                    deadline=time.monotonic()+3
                    while not marker.exists():
                        assert time.monotonic()<deadline,'descendant not ready'
                        time.sleep(.005)
                    raise primary
                return super().wait(timeout=timeout)
        process.os.killpg=cancel_once
        process.subprocess.Popen=Child
        process.signal.signal=install
        try:
            try:
                process.run_process([sys.executable,'-c',{worker!r}],
                    cwd=Path({str(tmp_path)!r}),env=os.environ.copy(),
                    log_path=Path({str(tmp_path / "worker.log")!r}),echo={echo!r})
            except BaseException as caught:
                if {primary_failure!r}: assert caught is primary,repr(caught)
                else:
                    assert isinstance(caught,process.ProcessInterrupted),repr(caught)
                    assert caught.signum==signal.SIGTERM
            else: raise AssertionError('cancellation or primary error was lost')
            assert signalled
            assert children[0].returncode is not None,'direct child not reaped'
            pid=int(marker.read_text())
            deadline=time.monotonic()+3
            while True:
                waited,status=os.waitpid(pid,os.WNOHANG)
                if waited:
                    assert os.waitstatus_to_exitcode(status)==-signal.SIGKILL
                    break
                assert time.monotonic()<deadline,'owned descendant survived cleanup'
                time.sleep(.005)
            assert threading.enumerate()==[threading.main_thread()]
            assert {{s:signal.getsignal(s) for s in saved}}==saved
        finally:
            process.os.killpg=original_killpg
            process.subprocess.Popen=original_popen
            process.signal.signal=original_signal
            for child in children:
                if child.poll() is None:
                    original_killpg(child.pid,signal.SIGKILL)
                    child.wait()
            if marker.exists():
                pid=int(marker.read_text())
                try: os.kill(pid,signal.SIGKILL)
                except ProcessLookupError: pass
                try: os.waitpid(pid,0)
                except ChildProcessError: pass
    """)
    completed = subprocess.run(
        [sys.executable, "-c", wrapper], capture_output=True, text=True, timeout=10
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process ownership")
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("stage", ["forward", "kill", "reap"])
@pytest.mark.parametrize("first_error", ["interrupt", "timeout"])
def test_second_cancellation_during_shutdown_preserves_first_failure(
    tmp_path, monkeypatch, echo, stage, first_error
):
    import threading

    original_popen, original_killpg = subprocess.Popen, os.killpg
    children = []
    first = (
        process.ProcessInterrupted(signal.SIGTERM)
        if first_error == "interrupt"
        else subprocess.TimeoutExpired([sys.executable, "-c", "pass"], 0)
    )
    raised = False
    cancelled = False
    shutdown = False
    saved = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    def interrupt_again():
        nonlocal cancelled
        cancelled = True
        os.kill(os.getpid(), signal.SIGINT)

    class Child(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            children.append(self)

        def wait(self, timeout=None):
            nonlocal raised
            if not raised:
                raised = True
                raise first
            if stage == "reap" and shutdown and not cancelled:
                interrupt_again()
            return super().wait(timeout=timeout)

    def killpg(pid, sig):
        nonlocal shutdown
        if sig == signal.SIGINT and stage == "forward" and not cancelled:
            interrupt_again()
        if sig == signal.SIGKILL:
            shutdown = True
            if stage == "kill" and not cancelled:
                interrupt_again()
        original_killpg(pid, sig)

    monkeypatch.setattr(process.subprocess, "Popen", Child)
    monkeypatch.setattr(process.os, "killpg", killpg)
    # Exit immediately so the grace-period wait is independent of scheduler speed.
    try:
        with pytest.raises(BaseException) as caught:
            process.run_process(
                [sys.executable, "-c", "pass"],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=tmp_path / "worker.log",
                echo=echo,
                timeout=0 if first_error == "timeout" else None,
            )
        if first_error == "interrupt":
            assert caught.value is first
        else:
            assert isinstance(caught.value, subprocess.TimeoutExpired)
            assert caught.value.cmd == first.cmd
            assert caught.value.timeout == first.timeout
        assert cancelled
        assert children[0].returncode is not None
        assert threading.enumerate() == [threading.main_thread()]
        assert {sig: signal.getsignal(sig) for sig in saved} == saved
        with pytest.raises(ChildProcessError):
            os.waitpid(children[0].pid, os.WNOHANG)
    finally:
        for child in children:
            if child.poll() is None:
                original_killpg(child.pid, signal.SIGKILL)
                child.wait()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process ownership")
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("worker_code", [7, -signal.SIGKILL])
@pytest.mark.parametrize("stage", ["group", "handler_setup"])
def test_cleanup_cancellation_preserves_observed_worker_failure(
    tmp_path, monkeypatch, echo, worker_code, stage
):
    from embodiedforge.runtime.records import managed_run, read_run, run_identity

    original_killpg, original_signal = os.killpg, signal.signal
    cancelled = False
    deferrals = 0

    def cancel_once():
        nonlocal cancelled
        cancelled = True
        os.kill(os.getpid(), signal.SIGTERM)

    def killpg(pid, sig):
        if stage == "group" and sig == signal.SIGKILL and not cancelled:
            cancel_once()
        original_killpg(pid, sig)

    def install(sig, handler):
        nonlocal deferrals
        previous = original_signal(sig, handler)
        if sig == signal.SIGINT and getattr(handler, "__name__", None) == "defer":
            deferrals += 1
            if stage == "handler_setup" and deferrals == 2:
                cancel_once()
        return previous

    monkeypatch.setattr(process.os, "killpg", killpg)
    monkeypatch.setattr(process.signal, "signal", install)
    command = [
        sys.executable,
        "-c",
        "import os,signal;os.write(1,b'worker output\\n');"
        + (
            "raise SystemExit(7)"
            if worker_code == 7
            else "os.kill(os.getpid(),signal.SIGKILL)"
        ),
    ]
    with pytest.raises(BaseException) as caught:
        with managed_run(tmp_path, run_identity("probe", "worker-failure")):
            process.run_process(
                command,
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=tmp_path / "worker.log",
                echo=echo,
            )
    assert isinstance(caught.value, subprocess.CalledProcessError)
    assert caught.value.returncode == worker_code
    assert caught.value.cmd == command
    assert cancelled
    assert read_run(tmp_path)["status"] == "failed"
    assert "interrupt_signal" not in json.loads((tmp_path / "run.json").read_text())
    assert (tmp_path / "worker.log").read_bytes() == b"worker output\n"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process ownership")
@pytest.mark.parametrize("echo", [False, True])
@pytest.mark.parametrize("worker_code", [0, 7])
def test_repeated_cancellation_during_cleanup_handler_setup_keeps_first_error(
    tmp_path, monkeypatch, echo, worker_code
):
    import threading

    from embodiedforge.runtime.records import managed_run, read_run, run_identity

    original = signal.signal
    deferrals = 0
    saved = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }

    def install(sig, handler):
        nonlocal deferrals
        if sig == signal.SIGINT and getattr(handler, "__name__", None) == "defer":
            deferrals += 1
            # Interrupt two consecutive attempts before any temporary handler
            # is installed. The constructor's startup deferral is attempt one.
            if deferrals in (2, 3):
                os.kill(
                    os.getpid(), signal.SIGTERM if deferrals == 2 else signal.SIGINT
                )
        return original(sig, handler)

    monkeypatch.setattr(process.signal, "signal", install)
    with pytest.raises(BaseException) as caught:
        with managed_run(tmp_path, run_identity("probe", "setup-cancel-twice")):
            process.run_process(
                [
                    sys.executable,
                    "-c",
                    f"print('finished');raise SystemExit({worker_code})",
                ],
                cwd=tmp_path,
                env=os.environ.copy(),
                log_path=tmp_path / "worker.log",
                echo=echo,
            )
    if worker_code:
        assert isinstance(caught.value, subprocess.CalledProcessError)
        assert caught.value.returncode == worker_code
        assert read_run(tmp_path)["status"] == "failed"
    else:
        assert isinstance(caught.value, process.ProcessInterrupted)
        assert caught.value.signum == signal.SIGTERM
        assert read_run(tmp_path)["status"] == "interrupted"
    assert deferrals >= 4
    assert threading.enumerate() == [threading.main_thread()]
    assert {sig: signal.getsignal(sig) for sig in saved} == saved
    assert (tmp_path / "worker.log").read_bytes() == b"finished\n"
