"""Bounded subprocess log capture with process-group interrupt forwarding."""

import codecs
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path


def sdk_package_path(output):
    """Expose only this package to the SDK, including its spawned processes."""
    directory = Path(output) / "python"
    directory.mkdir()
    (directory / "embodiedforge").symlink_to(
        Path(__file__).resolve().parent, target_is_directory=True
    )
    return directory


class ProcessInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__(f"Interrupted by signal {signum}")
        self.signum = signum


@contextmanager
def termination_signals():
    """Use normal cleanup for scheduler termination; respect nohup's SIGHUP ignore."""
    saved = {}
    if threading.current_thread() is threading.main_thread():

        def interrupt(signum, frame):
            raise ProcessInterrupted(signum)

        for sig in (signal.SIGTERM, signal.SIGHUP):
            previous = signal.getsignal(sig)
            if sig == signal.SIGHUP and previous == signal.SIG_IGN:
                continue
            saved[sig] = previous
            signal.signal(sig, interrupt)
    try:
        yield
    finally:
        for sig, previous in saved.items():
            signal.signal(sig, previous)


def run_process(
    command, *, cwd, env, log_path=None, echo=True, timeout=None, append_log=False
):
    with ExitStack() as resources:
        resources.enter_context(termination_signals())
        log = (
            resources.enter_context(log_path.open("ab" if append_log else "xb"))
            if log_path
            else None
        )
        # Quiet workers can write bytes straight to disk; only terminal echo
        # needs a pipe, decoder, and reader thread.
        capture = log is not None and echo
        process = resources.enter_context(
            subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                start_new_session=True,
                stdout=subprocess.PIPE if capture else log,
                stderr=subprocess.STDOUT if log else None,
            )
        )

        def send(sig):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        errors = []

        def copy_output():
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            console_open = echo

            def write_console(text):
                nonlocal console_open
                if console_open:
                    try:
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    except (OSError, ValueError):
                        # A closed terminal must not discard the durable log.
                        console_open = False

            try:
                while chunk := process.stdout.read1(65536):
                    log.write(chunk)
                    log.flush()
                    write_console(decoder.decode(chunk))
                write_console(decoder.decode(b"", final=True))
            except Exception as exc:
                errors.append(exc)
                # A broken log must not leave a silent training process running.
                send(signal.SIGTERM)

        reader = None
        if capture:
            reader = threading.Thread(target=copy_output, daemon=True)
            reader.start()
        deadline = time.monotonic() + timeout if timeout is not None else None
        try:
            try:
                while True:
                    try:
                        remaining = (
                            max(0.0, deadline - time.monotonic())
                            if deadline is not None
                            else None
                        )
                        wait = remaining
                        if capture:
                            wait = (
                                min(0.25, remaining) if remaining is not None else 0.25
                            )
                        returncode = process.wait(timeout=wait)
                        break
                    except subprocess.TimeoutExpired:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise subprocess.TimeoutExpired(command, timeout) from None
                        if errors:
                            send(signal.SIGKILL)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                send(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    pass
                finally:
                    # The wrapper may exit before descendants that ignore SIGINT
                    # and redirected stdout. They cannot be detected by the log
                    # reader, so always clean the owned group after the grace period.
                    send(signal.SIGKILL)
                    process.wait()
                raise
        finally:
            if errors or process.returncode:
                # A failed wrapper can leave quiet descendants running as well.
                send(signal.SIGKILL)
            if reader:
                reader.join(timeout=5)
                if reader.is_alive():
                    # A child that inherited stdout may outlive the main worker.
                    send(signal.SIGKILL)
                    reader.join(timeout=5)
                if reader.is_alive() and sys.exc_info()[0] is None:
                    raise RuntimeError(f"Training log reader did not stop: {log_path}")
        if errors:
            # The last flush can fail while joining an already exited worker's
            # reader, after the earlier failure check in finally.
            send(signal.SIGKILL)
            raise RuntimeError(
                f"Failed to capture training log {log_path}: {errors[0]}"
            ) from errors[0]
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
