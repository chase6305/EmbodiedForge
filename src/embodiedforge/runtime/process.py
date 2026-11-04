"""Bounded subprocess log capture with process-group interrupt forwarding."""

import codecs
import logging
import math
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from numbers import Real
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def sdk_package_path(output, *, package=None):
    """Expose only this package to the SDK, including its spawned processes."""
    directory = Path(output) / "python"
    directory.mkdir()
    (directory / "embodiedforge").symlink_to(
        Path(package).resolve()
        if package is not None
        else Path(__file__).resolve().parents[1],
        target_is_directory=True,
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
    active_error = False
    try:
        if threading.current_thread() is threading.main_thread():

            def interrupt(signum, frame):
                raise ProcessInterrupted(signum)

            for name in ("SIGTERM", "SIGHUP"):
                sig = getattr(signal, name, None)
                if sig is None:
                    continue
                previous = signal.getsignal(sig)
                if name == "SIGHUP" and previous == signal.SIG_IGN:
                    continue
                saved[sig] = previous
                signal.signal(sig, interrupt)
        yield
    except BaseException:
        active_error = True
        raise
    finally:
        failures = []
        for sig, previous in saved.items():
            try:
                signal.signal(sig, previous)
            except BaseException as error:
                failures.append(error)
                LOGGER.exception(
                    "Could not restore signal handler %s; preserving %s",
                    sig,
                    "the original error" if active_error else "the first cleanup error",
                )
        if failures and not active_error:
            raise failures[0]


@contextmanager
def _log_file(path, append):
    stream = path.open("ab" if append else "xb")
    active_error = False
    try:
        yield stream
    except BaseException:
        active_error = True
        raise
    finally:
        try:
            stream.close()
        except BaseException:
            if not active_error:
                raise
            # Buffered output can fail once during capture and again on close
            # (for example ENOSPC). Keep the execution/capture error authoritative.
            LOGGER.exception(
                "Could not close training log %s; preserving its original error", path
            )


@contextmanager
def _defer_process_signals(*, preserve_error=False):
    """Deliver cancellation after acquiring or releasing owned process resources.

    Blocking signals with pthread_sigmask would also block them in the worker.
    Temporarily defer callable handlers in Python instead; ignored/default signal
    dispositions and the child's signal mask remain unchanged.
    """
    saved = {}
    pending = []
    active_error = False

    def defer(signum, frame):
        if not pending:
            pending.append((signum, frame))

    try:
        if threading.current_thread() is threading.main_thread():
            for name in ("SIGINT", "SIGTERM", "SIGHUP"):
                sig = getattr(signal, name, None)
                if sig is not None:
                    previous = signal.getsignal(sig)
                    if callable(previous):
                        saved[sig] = previous
                        signal.signal(sig, defer)
        yield
    except BaseException:
        active_error = True
        raise
    finally:
        failures = []
        for sig, previous in saved.items():
            try:
                signal.signal(sig, previous)
            except BaseException as error:
                failures.append(error)
                LOGGER.exception("Could not restore process signal handler %s", sig)
        if not active_error and not preserve_error:
            if pending:
                signum, frame = pending[0]
                saved[signum](signum, frame)
            if failures:
                raise failures[0]


def _copy_output(pipe, log, stop_reader, errors, send):
    """Copy raw pipe bytes to disk and echo decoded text until EOF or teardown."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    console_open = True

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
        # A detached descendant can retain stdout after the owned job exits.
        # Read the descriptor directly: blocking on the buffered pipe would
        # prevent teardown from joining this reader and closing the pipe.
        descriptor = pipe.fileno()
        os.set_blocking(descriptor, False)
        with selectors.DefaultSelector() as readable:
            readable.register(descriptor, selectors.EVENT_READ)
            remaining = None
            while True:
                if remaining is None and stop_reader.is_set():
                    import fcntl
                    import termios
                    from array import array

                    # Drain only bytes already queued after the worker exits;
                    # an escaped process may otherwise produce output forever.
                    pending = array("i", [0])
                    fcntl.ioctl(descriptor, termios.FIONREAD, pending)
                    remaining = pending[0]
                if remaining == 0:
                    break
                try:
                    chunk = os.read(
                        descriptor,
                        65536 if remaining is None else min(65536, remaining),
                    )
                except BlockingIOError:
                    if stop_reader.is_set():
                        break
                    readable.select(timeout=0.1)
                    continue
                if not chunk:
                    break
                if remaining is not None:
                    remaining -= len(chunk)
                log.write(chunk)
                log.flush()
                write_console(decoder.decode(chunk))
        write_console(decoder.decode(b"", final=True))
    except Exception as error:
        errors.append(error)
        # A broken log must not leave a silent training process running.
        send(signal.SIGTERM)


def run_process(
    command,
    *,
    cwd,
    env,
    log_path=None,
    echo=True,
    timeout=None,
    append_log=False,
    pass_fds=(),
):
    """Run a POSIX job and stop its owned group on exit or cancellation.

    timeout is a finite nonnegative wait budget, or None. Captured logs keep raw
    bytes; terminal echo decodes UTF-8 with replacement. Teardown drains buffered
    output without waiting for stdout holders outside the owned process group.
    """
    if timeout is not None:
        try:
            valid_timeout = (
                isinstance(timeout, Real)
                and not isinstance(timeout, bool)
                and math.isfinite(timeout)
                and timeout >= 0
            )
        except OverflowError:
            valid_timeout = False
        if not valid_timeout:
            raise ValueError("timeout must be finite and nonnegative, or None")
    with ExitStack() as resources:
        resources.enter_context(termination_signals())
        log = (
            resources.enter_context(_log_file(log_path, append_log))
            if log_path
            else None
        )
        # Quiet workers can write bytes straight to disk; only terminal echo
        # needs a pipe, decoder, and reader thread.
        capture = log is not None and echo
        stop_reader = threading.Event() if capture else None
        process = None

        def send(sig):
            if process is None:
                return
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass

        errors = []

        reader = None
        active_error = False
        cleanup_finished = False

        def cleanup():
            nonlocal cleanup_finished
            send(signal.SIGKILL)
            if process is not None:
                process.wait()
            if stop_reader is not None:
                stop_reader.set()
            if reader is not None and reader.ident is not None:
                reader.join(timeout=5)
                if reader.is_alive():
                    # A child that inherited stdout may outlive the main worker.
                    send(signal.SIGKILL)
                    reader.join(timeout=5)
                if reader.is_alive() and not active_error:
                    raise RuntimeError(f"Training log reader did not stop: {log_path}")
            cleanup_finished = True

        try:
            try:
                # Popen may be interrupted after fork/exec but before returning
                # the handle. Replay cancellation only after cleanup owns it.
                with _defer_process_signals():
                    process = resources.enter_context(
                        subprocess.Popen(
                            command,
                            cwd=cwd,
                            env=env,
                            start_new_session=True,
                            stdout=subprocess.PIPE if capture else log,
                            stderr=subprocess.STDOUT if log else None,
                            pass_fds=pass_fds,
                        )
                    )
                    # Cancellation can make the worker emit its final logs.
                    # Start draining before replaying a deferred signal so those
                    # bytes survive and pipe backpressure cannot block teardown.
                    # Thread startup failures still use the owned group cleanup.
                    if capture:
                        reader = threading.Thread(
                            target=_copy_output,
                            args=(process.stdout, log, stop_reader, errors, send),
                            daemon=True,
                        )
                        reader.start()
                deadline = time.monotonic() + timeout if timeout is not None else None
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
                # These failures are already known, even though their exceptions
                # are raised after the final log drain below. A later cleanup
                # cancellation must not turn an observed worker/capture failure
                # into a launcher interruption.
                active_error = bool(returncode or errors)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                if process is None:
                    raise
                try:
                    # A repeated cancellation can arrive while forwarding the
                    # first one, as well as during the graceful wait. It ends
                    # that wait without replacing the original interruption or
                    # timeout. The protected outer cleanup kills/reaps the group
                    # and drains output exactly once in either case.
                    send(signal.SIGINT)
                    process.wait(timeout=10)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    pass
                raise
        except BaseException:
            active_error = True
            raise
        finally:
            # The owned job ends with its wrapper, including a successful exit.
            # Quiet descendants do not keep the log pipe open and would otherwise
            # survive indefinitely. Unrelated jobs use different process groups.
            # A first cancellation can arrive during normal teardown too.
            # Complete group cleanup, reaping, and log drain before replaying
            # it; an existing execution failure remains authoritative.
            cleanup_interruption = None
            while not cleanup_finished:
                try:
                    with _defer_process_signals(
                        preserve_error=active_error or cleanup_interruption is not None
                    ):
                        cleanup()
                except KeyboardInterrupt as error:
                    # More than one cancellation can interrupt handler setup
                    # before cleanup starts. Retry unfinished cleanup, retaining
                    # the first cancellation or the earlier execution failure.
                    if cleanup_interruption is None:
                        cleanup_interruption = error
            if cleanup_interruption is not None and not active_error:
                raise cleanup_interruption
        if errors:
            # The last flush can fail while joining an already exited worker's
            # reader, after the earlier failure check in finally.
            send(signal.SIGKILL)
            raise RuntimeError(
                f"Failed to capture training log {log_path}: {errors[0]}"
            ) from errors[0]
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
