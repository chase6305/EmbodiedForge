"""Real thread/process shutdown tests without requiring a GPU or desktop."""

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from embodiedforge._microduck_worker import JoinedRenderThreadMixin
from embodiedforge.microduck import run


def test_sdk_module_does_not_import_host_site_packages(tmp_path, monkeypatch):
    from embodiedforge import _microduck_process as process

    host = tmp_path / "host site-packages"
    package = host / "embodiedforge"
    package.mkdir(parents=True)
    (host / "numpy.py").write_text("raise RuntimeError('wrong Python environment')\n")
    (package / "__init__.py").write_text("import numpy\n")
    (package / "probe.py").write_text(
        "import json, multiprocessing, sys, embodiedforge, numpy\n"
        "def report():\n"
        "    print(json.dumps({'arguments': sys.argv[1:], 'numpy': numpy.__file__, "
        "'package': embodiedforge.__file__, 'search_path': sys.path}))\n"
        "if __name__ == '__main__':\n"
        "    child = multiprocessing.get_context('spawn').Process(target=report)\n"
        "    child.start()\n"
        "    child.join(5)\n"
        "    assert child.exitcode == 0\n"
        "    report()\n"
    )
    monkeypatch.setattr(process, "__file__", str(package / "_microduck_process.py"))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(process.sdk_package_path(tmp_path))
    result = subprocess.run(
        [sys.executable, "-m", "embodiedforge.probe", "with spaces"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    reports = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(reports) == 2
    for report in reports:
        assert report["arguments"] == ["with spaces"]
        assert Path(report["package"]).resolve() == package / "__init__.py"
        assert not Path(report["numpy"]).is_relative_to(host)
        assert str(host) not in report["search_path"]


@pytest.mark.parametrize("setup_fails", [False, True])
def test_join_only_owned_render_thread_even_after_failed_setup(setup_fails):
    stopped, cleaned = threading.Event(), threading.Event()
    unrelated_stop, unrelated_cleaned = threading.Event(), threading.Event()

    def renderer(stop, complete):
        stop.wait()
        time.sleep(0.04)  # Simulate asynchronous native resource destruction.
        complete.set()

    class Backend:
        def setup(self):
            self.thread = threading.Thread(target=renderer, args=(stopped, cleaned))
            self.thread.start()
            if setup_fails:
                raise RuntimeError("setup failed")

        def close(self):
            stopped.set()

    class Viewer(JoinedRenderThreadMixin, Backend):
        _render_target = staticmethod(renderer)

    unrelated = threading.Thread(
        target=renderer, args=(unrelated_stop, unrelated_cleaned)
    )
    unrelated.start()
    viewer = Viewer()
    try:
        if setup_fails:
            with pytest.raises(RuntimeError, match="setup failed"):
                viewer.setup()
        else:
            viewer.setup()
        viewer.close()
        assert cleaned.is_set()
        assert not viewer.thread.is_alive()
        assert unrelated.is_alive()
        viewer.close()  # Safe after BaseViewer.run already performed cleanup.
    finally:
        stopped.set()
        unrelated_stop.set()
        unrelated.join(timeout=2)
        viewer.thread.join(timeout=2)


def test_stalled_render_thread_is_reported():
    class Stalled:
        def join(self, timeout):
            assert timeout == 5

        def is_alive(self):
            return True

    class Backend:
        def close(self):
            pass

    class Viewer(JoinedRenderThreadMixin, Backend):
        pass

    viewer = Viewer()
    viewer._owned_render_threads = [Stalled()]
    with pytest.raises(RuntimeError, match="did not stop"):
        viewer.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal handling")
@pytest.mark.parametrize(
    "frontend,capture_log,echo,interrupt_signal",
    [
        ("microduck", False, True, signal.SIGINT),
        ("microduck", True, True, signal.SIGTERM),
        ("microduck", True, False, signal.SIGHUP),
        ("microduck", True, False, None),
        ("h1", True, False, None),
        ("recipes", True, False, None),
    ],
)
def test_failure_and_interrupt_clean_owned_process_group(
    tmp_path, frontend, capture_log, echo, interrupt_signal
):
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    descendant = tmp_path / "descendant.pid"
    stubborn = textwrap.dedent(f"""
        import os, signal, time
        from pathlib import Path
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        Path({str(descendant)!r}).write_text(str(os.getpid()))
        time.sleep(30)
    """)
    child = textwrap.dedent(f"""
        import signal, subprocess, sys, time
        from pathlib import Path
        def cleanup(*args):
            time.sleep(0.6)
            print("cleanup complete", flush=True)
            Path({str(cleaned)!r}).touch()
            raise SystemExit({7 if interrupt_signal is None else 0})
        signal.signal(signal.SIGINT, cleanup)
        if sys.platform == 'linux':
            subprocess.Popen([sys.executable, '-c', {stubborn!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            while not Path({str(descendant)!r}).exists():
                time.sleep(0.01)
        Path({str(ready)!r}).touch()
        if {interrupt_signal is None!r}:
            cleanup()
        while True:
            time.sleep(0.01)
    """)
    # Isolate the test's SIGINT from pytest and the user's terminal session.
    wrapper = textwrap.dedent(f"""
        import os, signal, subprocess, sys, threading, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        from embodiedforge import microduck, h1, recipes
        if sys.platform == 'linux':
            import ctypes
            # Reap the fixture's orphaned descendant instead of leaving zombies.
            assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
        def interrupt():
            if {interrupt_signal is None!r}:
                return
            deadline = time.monotonic() + 4
            while not Path({str(ready)!r}).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            os.kill(os.getpid(), {int(interrupt_signal or 0)})
        thread = threading.Thread(target=interrupt)
        thread.start()
        try:
            command = [sys.executable, '-c', {child!r}]
            output = Path({str(tmp_path)!r})
            if {frontend!r} == 'h1':
                h1.run_process(command, cwd=output, env=os.environ.copy(), timeout=None)
            elif {frontend!r} == 'recipes':
                recipes.run_process(command, output, os.environ.copy(), None)
            else:
                microduck.run(command, cwd=output, env=os.environ.copy(), log_path=output / 'interrupt.log' if {capture_log!r} else None, echo={echo!r})
        except (KeyboardInterrupt, subprocess.CalledProcessError) as exc:
            thread.join()
            if sys.platform == 'linux':
                pid = int(Path({str(descendant)!r}).read_text())
                reaped = 0
                deadline = time.monotonic() + 1
                try:
                    while not reaped and time.monotonic() < deadline:
                        reaped, status = os.waitpid(pid, os.WNOHANG)
                        time.sleep(0.01)
                finally:
                    if not reaped:
                        os.kill(pid, signal.SIGKILL)
                        os.waitpid(pid, 0)
                assert reaped, 'Stopped wrapper left a descendant running'
            if isinstance(exc, subprocess.CalledProcessError):
                raise SystemExit(exc.returncode)
            raise SystemExit(128 + getattr(exc, "signum", signal.SIGINT))
    """)
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        timeout=10,
        start_new_session=True,
    )
    expected = 128 + interrupt_signal if interrupt_signal is not None else 7
    assert result.returncode == expected, result.stderr
    assert cleaned.exists(), "Child was killed before native cleanup could finish"
    if capture_log:
        log = "interrupt.log" if frontend == "microduck" else "console.log"
        assert "cleanup complete" in (tmp_path / log).read_text()


def test_log_captures_stdout_stderr_unicode_and_nonzero_exit(tmp_path, capsys):
    log = tmp_path / "train.log"
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('训练开始'); print('failure detail', file=sys.stderr); raise SystemExit(9)",
            ],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=log,
        )
    assert exc.value.returncode == 9
    assert log.read_text() == "训练开始\nfailure detail\n"
    assert "训练开始\nfailure detail" in capsys.readouterr().out


def test_log_drains_more_than_pipe_capacity(tmp_path, capsys):
    log = tmp_path / "large.log"
    run(
        [sys.executable, "-c", "import sys;sys.stdout.write('x'*1000000)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log,
    )
    assert log.stat().st_size == 1000000
    assert len(capsys.readouterr().out) == 1000000


def test_existing_log_is_never_overwritten(tmp_path):
    log = tmp_path / "existing.log"
    log.write_text("previous run")
    with pytest.raises(FileExistsError):
        run(
            [sys.executable, "-c", "pass"],
            cwd=tmp_path,
            env=os.environ.copy(),
            log_path=log,
        )
    assert log.read_text() == "previous run"


def test_closed_console_does_not_discard_worker_log(tmp_path, monkeypatch):
    class Closed:
        def write(self, text):
            raise BrokenPipeError("downstream reader exited")

        def flush(self):
            raise BrokenPipeError("downstream reader exited")

    log = tmp_path / "closed.log"
    monkeypatch.setattr(sys, "stdout", Closed())
    run(
        [sys.executable, "-c", "print('durable output')"],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log,
    )
    assert log.read_text() == "durable output\n"


def test_termination_handlers_restore_and_preserve_nohup():
    from embodiedforge._microduck_process import termination_signals

    original_term = signal.getsignal(signal.SIGTERM)
    original_hup = signal.getsignal(signal.SIGHUP)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        with termination_signals():
            assert signal.getsignal(signal.SIGTERM) != original_term
            assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
        assert signal.getsignal(signal.SIGTERM) == original_term
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        signal.signal(signal.SIGHUP, original_hup)


def test_quiet_keeps_complete_log_without_console_echo(tmp_path, capsys):
    from embodiedforge._microduck_process import run_process

    log = tmp_path / "quiet.log"
    run_process(
        [
            sys.executable,
            "-c",
            "import sys;sys.stdout.buffer.write(b'\\xff'*1000000);"
            "sys.stdout.flush();print('error too',file=sys.stderr)",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        log_path=log,
        echo=False,
    )
    assert log.read_bytes() == b"\xff" * 1000000 + b"error too\n"
    assert capsys.readouterr().out == ""
