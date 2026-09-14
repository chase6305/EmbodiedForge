"""Real thread/process shutdown tests without requiring a GPU or desktop."""

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from embodiedforge._microduck_worker import JoinedRenderThreadMixin
from embodiedforge.microduck import run


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


def test_child_nonzero_exit_is_preserved(tmp_path):
    with pytest.raises(subprocess.CalledProcessError) as exc:
        run(
            [sys.executable, "-c", "raise SystemExit(7)"],
            cwd=tmp_path,
            env=os.environ.copy(),
        )
    assert exc.value.returncode == 7


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group signal handling")
def test_interrupt_waits_for_child_cleanup(tmp_path):
    ready, cleaned = tmp_path / "ready", tmp_path / "cleaned"
    child = textwrap.dedent(f"""
        import signal, time
        from pathlib import Path
        def cleanup(*args):
            time.sleep(0.6)
            Path({str(cleaned)!r}).touch()
            raise SystemExit(0)
        signal.signal(signal.SIGINT, cleanup)
        Path({str(ready)!r}).touch()
        while True:
            time.sleep(0.01)
    """)
    # Isolate the test's SIGINT from pytest and the user's terminal session.
    wrapper = textwrap.dedent(f"""
        import os, signal, sys, threading, time
        from pathlib import Path
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / "src")!r})
        from embodiedforge.microduck import run
        def interrupt():
            deadline = time.monotonic() + 4
            while not Path({str(ready)!r}).exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGINT)
        thread = threading.Thread(target=interrupt)
        thread.start()
        try:
            run([sys.executable, '-c', {child!r}], cwd=Path({str(tmp_path)!r}), env=os.environ.copy())
        except KeyboardInterrupt:
            thread.join()
            raise SystemExit(130)
    """)
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        capture_output=True,
        text=True,
        timeout=10,
        start_new_session=True,
    )
    assert result.returncode == 130, result.stderr
    assert cleaned.exists(), "Child was killed before native cleanup could finish"
