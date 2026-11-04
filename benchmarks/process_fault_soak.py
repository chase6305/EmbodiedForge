"""Sustained real-worker startup cancellation and full-log failure validation.

Signals are injected at Popen entry, return and context registration. The fixture
waits for a real worker and reaps it even when assertions fail. /dev/full produces
real ENOSPC errors; run records are written separately on the normal filesystem.
Use an external supervisor to verify interpreter shutdown as well as this report.
"""

import argparse
import gc
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from embodiedforge.runtime import process
from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import file_sha256
from embodiedforge.runtime.records import (
    managed_run,
    read_run,
    run_identity,
    run_lifecycle,
)

SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
STAGES = ("before", "after", "enter")


def resources():
    root = Path("/proc/self")
    status = dict(
        line.split(":", 1) for line in (root / "status").read_text().splitlines()
    )
    return {
        "rss_bytes": int(status["VmRSS"].split()[0]) * 1024,
        "threads": int(status["Threads"]),
        "fds": len(list((root / "fd").iterdir())),
    }


@contextmanager
def preserve_popen():
    original = process.subprocess.Popen
    try:
        yield original
    finally:
        process.subprocess.Popen = original


def startup(root, signum, stage, echo):
    root.mkdir()
    marker, ready = root / "child.pid", root / "ready"
    worker = """import json,os,signal,time
from pathlib import Path
Path('child.pid').write_text(str(os.getpid()))
Path('mask.json').write_text(json.dumps(sorted(signal.pthread_sigmask(signal.SIG_BLOCK,[]))))
def stop(*args):
    Path('cleaned').touch()
    raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
Path('ready').touch()
time.sleep(30)
"""
    saved = {sig: signal.getsignal(sig) for sig in SIGNALS}
    mask = set(signal.pthread_sigmask(signal.SIG_BLOCK, []))
    caught = None
    started = []
    try:
        with preserve_popen() as original:

            class Inject(original):
                def __init__(self, *args, **kwargs):
                    if stage == "before":
                        os.kill(os.getpid(), signum)
                    super().__init__(*args, **kwargs)
                    started.append(self)
                    deadline = time.monotonic() + 3
                    while not ready.exists():
                        returncode = self.poll()
                        if returncode is not None:
                            write_json(
                                root / "startup-error.json",
                                {
                                    "pid": self.pid,
                                    "returncode": returncode,
                                    "condition": "worker_exited_before_ready",
                                },
                            )
                            raise RuntimeError(
                                f"worker exited before ready: pid={self.pid}, "
                                f"returncode={returncode}"
                            )
                        if time.monotonic() >= deadline:
                            write_json(
                                root / "startup-error.json",
                                {
                                    "pid": self.pid,
                                    "returncode": self.poll(),
                                    "condition": "worker_readiness_timeout",
                                },
                            )
                            raise AssertionError(
                                f"worker did not become ready: pid={self.pid}"
                            )
                        time.sleep(0.001)
                    if stage == "after":
                        os.kill(os.getpid(), signum)

                def __enter__(self):
                    result = super().__enter__()
                    if stage == "enter":
                        os.kill(os.getpid(), signum)
                    return result

            process.subprocess.Popen = Inject
            try:
                with managed_run(root, run_identity("probe", "startup-cancel")):
                    process.run_process(
                        [sys.executable, "-c", worker],
                        cwd=root,
                        env=os.environ.copy(),
                        log_path=root / "worker.log",
                        echo=echo,
                    )
            except KeyboardInterrupt as error:
                caught = error
        assert caught is not None, (stage, signum, echo)
        assert getattr(caught, "signum", signal.SIGINT) == signum
        graceful = (root / "cleaned").is_file()
        # A cancelled worker need not handle SIGINT within the grace period.
        # Verify the documented forced-stop fallback, recording it separately.
        assert graceful or started[0].returncode == -signal.SIGKILL
        assert set(json.loads((root / "mask.json").read_text())) == mask
        record = read_run(root)
        assert record["status"] == "interrupted"
        assert json.loads((root / "run.json").read_text())["interrupt_signal"] == signum
        pid = int(marker.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError("cancelled worker survived startup")
        assert {sig: signal.getsignal(sig) for sig in SIGNALS} == saved
        assert threading.enumerate() == [threading.main_thread()]
        return {
            "signal": signum,
            "stage": stage,
            "echo": echo,
            "graceful": graceful,
            "returncode": started[0].returncode,
        }
    finally:
        # The process can exit before writing its first marker, or the fixture
        # constructor can fail before run_process owns the returned handle.
        for child in started:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


def disk_full(root, outcome):
    root.mkdir()
    original = os.killpg
    caught = None

    def cancel(pid, sig):
        original(pid, sig)
        if outcome == "interrupt" and sig == signal.SIGTERM:
            os.kill(os.getpid(), signal.SIGTERM)

    process.os.killpg = cancel
    try:
        try:
            with managed_run(root, run_identity("probe", "disk-full")):
                process.run_process(
                    [
                        sys.executable,
                        "-c",
                        "import os,signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
                        "os.write(1,b'worker output\\n');"
                        + (
                            "time.sleep(30)"
                            if outcome in ("timeout", "interrupt")
                            else f"raise SystemExit({7 if outcome == 'failure' else 0})"
                        ),
                    ],
                    cwd=root,
                    env=os.environ.copy(),
                    log_path=Path("/dev/full"),
                    append_log=True,
                    timeout=0.15 if outcome == "timeout" else 3,
                )
        except BaseException as error:
            caught = error
    finally:
        process.os.killpg = original
    if outcome == "timeout":
        assert isinstance(caught, subprocess.TimeoutExpired), caught
        expected = "timed_out"
    elif outcome == "interrupt":
        assert isinstance(caught, process.ProcessInterrupted), caught
        assert caught.signum == signal.SIGTERM
        expected = "interrupted"
    else:
        assert isinstance(caught, RuntimeError), caught
        assert isinstance(caught.__cause__, OSError), caught
        assert caught.__cause__.errno == 28
        expected = "failed"
    assert read_run(root)["status"] == expected
    assert threading.enumerate() == [threading.main_thread()]


def finalization_failure(root, stage):
    root.mkdir()
    failure = OSError("injected finalization failure")
    record = {
        "status": "running",
        "run_contract": run_identity("probe", "finalization"),
    }

    def publish(current):
        if stage == "publish" and current["status"] == "complete":
            raise failure
        write_json(root / "run.json", current)

    def finalize(current):
        if stage == "finalize":
            raise failure

    caught = None
    try:
        with run_lifecycle(record, publish, finalize=finalize):
            pass
    except OSError as error:
        caught = error
    assert caught is failure
    assert read_run(root)["status"] == "failed"


def cleanup_cancellation(
    root, signum, echo, primary_failure, cancel_stage, *, worker_returncode=0
):
    """Cancel group cleanup with a real same-group descendant still owned."""
    root.mkdir()
    marker = root / "descendant.pid"
    worker = (
        "import os,subprocess,sys;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
        f"Path({str(marker) + '.tmp'!r}).write_text(str(child.pid));"
        f"os.replace({str(marker) + '.tmp'!r},{str(marker)!r})"
    )
    if worker_returncode == 7:
        worker += ";raise SystemExit(7)"
    elif worker_returncode == -signal.SIGKILL:
        worker += ";os.kill(os.getpid(),9)"
    original_killpg, original_popen = os.killpg, subprocess.Popen
    original_signal = signal.signal
    primary = RuntimeError("injected wait failure")
    signalled = failed = reaped = False
    deferrals = 0
    children = []
    saved = {sig: signal.getsignal(sig) for sig in SIGNALS}

    def cancel_once(pid, sig):
        nonlocal signalled
        if cancel_stage == "group" and sig == signal.SIGKILL and not signalled:
            signalled = True
            os.kill(os.getpid(), signum)
        original_killpg(pid, sig)

    def install(sig, handler):
        nonlocal deferrals, signalled
        installing = (
            sig == signal.SIGINT and getattr(handler, "__name__", None) == "defer"
        )
        if installing:
            deferrals += 1
            if cancel_stage == "handler_repeat" and deferrals in (2, 3):
                signalled = True
                os.kill(os.getpid(), signum if deferrals == 2 else signal.SIGINT)
        previous = original_signal(sig, handler)
        if installing and cancel_stage == "handler_setup" and deferrals == 2:
            signalled = True
            os.kill(os.getpid(), signum)
        return previous

    class Child(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            children.append(self)

        def wait(self, timeout=None):
            nonlocal failed
            if primary_failure and not failed:
                failed = True
                deadline = time.monotonic() + 3
                while not marker.exists():
                    assert time.monotonic() < deadline, "descendant not ready"
                    time.sleep(0.005)
                raise primary
            return super().wait(timeout=timeout)

    os.killpg, subprocess.Popen = cancel_once, Child
    signal.signal = install
    try:
        caught = None
        try:
            with managed_run(root, run_identity("probe", "cleanup-cancel")):
                process.run_process(
                    [sys.executable, "-c", worker],
                    cwd=root,
                    env=os.environ.copy(),
                    log_path=root / "worker.log",
                    echo=echo,
                )
        except BaseException as error:
            caught = error
        if primary_failure:
            assert caught is primary, repr(caught)
        elif worker_returncode:
            assert isinstance(caught, subprocess.CalledProcessError), repr(caught)
            assert caught.returncode == worker_returncode
        else:
            assert isinstance(caught, KeyboardInterrupt), repr(caught)
            assert getattr(caught, "signum", signal.SIGINT) == signum
        assert signalled
        assert children[0].returncode is not None, "direct child not reaped"
        pid = int(marker.read_text())
        deadline = time.monotonic() + 3
        while True:
            waited, status = os.waitpid(pid, os.WNOHANG)
            if waited:
                reaped = True
                assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
                break
            assert time.monotonic() < deadline, "owned descendant survived cleanup"
            time.sleep(0.005)
        assert read_run(root)["status"] == (
            "failed" if primary_failure or worker_returncode else "interrupted"
        )
        assert {sig: signal.getsignal(sig) for sig in SIGNALS} == saved
        assert threading.enumerate() == [threading.main_thread()]
    finally:
        os.killpg, subprocess.Popen = original_killpg, original_popen
        signal.signal = original_signal
        for child in children:
            try:
                original_killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        if marker.exists() and not reaped:
            pid = int(marker.read_text())
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass


def repeated_cancellation(root, first, echo, stage):
    """Keep real initial cancellation/timeout authoritative through shutdown."""
    root.mkdir()
    ready = root / "ready"
    worker = """import signal,time
from pathlib import Path
def stop(*args):
    Path('cleaned').touch()
    raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
Path('ready').touch()
while True: time.sleep(.01)
"""
    original_popen, original_killpg = subprocess.Popen, os.killpg
    children = []
    initial = secondary = killing = False
    saved = {sig: signal.getsignal(sig) for sig in SIGNALS}

    def cancel_again():
        nonlocal secondary
        secondary = True
        os.kill(os.getpid(), signal.SIGINT)

    class Child(original_popen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            children.append(self)
            deadline = time.monotonic() + 3
            while not ready.exists():
                assert self.poll() is None, "worker exited before ready"
                assert time.monotonic() < deadline, "worker not ready"
                time.sleep(0.001)

        def wait(self, timeout=None):
            nonlocal initial
            if first == "interrupt" and not initial:
                initial = True
                os.kill(os.getpid(), signal.SIGTERM)
            if stage == "reap" and killing and not secondary:
                cancel_again()
            return super().wait(timeout=timeout)

    def killpg(pid, sig):
        nonlocal killing
        if sig == signal.SIGINT and stage == "forward" and not secondary:
            cancel_again()
        if sig == signal.SIGKILL:
            killing = True
            if stage == "kill" and not secondary:
                cancel_again()
        original_killpg(pid, sig)

    subprocess.Popen, os.killpg = Child, killpg
    try:
        caught = None
        try:
            with managed_run(root, run_identity("probe", "repeated-cancel")):
                process.run_process(
                    [sys.executable, "-c", worker],
                    cwd=root,
                    env=os.environ.copy(),
                    log_path=root / "worker.log",
                    echo=echo,
                    timeout=0 if first == "timeout" else None,
                )
        except BaseException as error:
            caught = error
        if first == "interrupt":
            assert isinstance(caught, process.ProcessInterrupted), repr(caught)
            assert caught.signum == signal.SIGTERM
            assert read_run(root)["status"] == "interrupted"
            assert (
                json.loads((root / "run.json").read_text())["interrupt_signal"]
                == signal.SIGTERM
            )
        else:
            assert isinstance(caught, subprocess.TimeoutExpired), repr(caught)
            assert caught.timeout == 0
            assert read_run(root)["status"] == "timed_out"
        assert secondary
        assert children[0].returncode is not None, "direct child not reaped"
        try:
            os.waitpid(children[0].pid, os.WNOHANG)
        except ChildProcessError:
            pass
        else:
            raise AssertionError("child was not reaped")
        assert {sig: signal.getsignal(sig) for sig in SIGNALS} == saved
        assert threading.enumerate() == [threading.main_thread()]
    finally:
        subprocess.Popen, os.killpg = original_popen, original_killpg
        for child in children:
            try:
                original_killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            for stream in (child.stdin, child.stdout, child.stderr):
                if stream is not None:
                    stream.close()


def run_cycle(
    cycle,
    *,
    include_cleanup=False,
    include_repeated=False,
    include_worker_failures=False,
):
    stopped = []
    for signum in SIGNALS:
        for stage in STAGES:
            for echo in (False, True):
                result = startup(
                    cycle / f"{signum}-{stage}-{echo}", signum, stage, echo
                )
                if not result["graceful"]:
                    stopped.append(result)
    for outcome in ("success", "failure", "timeout", "interrupt"):
        disk_full(cycle / outcome, outcome)
    for stage in ("finalize", "publish"):
        finalization_failure(cycle / stage, stage)
    if include_cleanup:
        for sig in SIGNALS:
            for echo in (False, True):
                for primary in (False, True):
                    for stage in ("group", "handler_setup"):
                        cleanup_cancellation(
                            cycle / f"cleanup-{sig}-{echo}-{primary}-{stage}",
                            sig,
                            echo,
                            primary,
                            stage,
                        )
    if include_repeated:
        for first in ("interrupt", "timeout"):
            for echo in (False, True):
                for stage in ("forward", "kill", "reap"):
                    repeated_cancellation(
                        cycle / f"repeated-{first}-{echo}-{stage}", first, echo, stage
                    )
        for primary in (False, True):
            for code in (0, 7):
                for echo in (False, True):
                    cleanup_cancellation(
                        cycle / f"setup-twice-{primary}-{code}-{echo}",
                        signal.SIGTERM,
                        echo,
                        primary,
                        "handler_repeat",
                        worker_returncode=code,
                    )
    if include_worker_failures:
        for code in (7, -signal.SIGKILL):
            for echo in (False, True):
                for stage in ("group", "handler_setup"):
                    cleanup_cancellation(
                        cycle / f"worker-failure-{code}-{echo}-{stage}",
                        signal.SIGTERM,
                        echo,
                        False,
                        stage,
                        worker_returncode=code,
                    )
    return stopped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preload-core", action="store_true")
    parser.add_argument("--cleanup-cancellation", action="store_true")
    parser.add_argument("--repeated-cancellation", action="store_true")
    parser.add_argument("--worker-failure-cancellation", action="store_true")
    parser.add_argument("--gc-threshold", type=int)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/dev/full").exists():
        parser.error("Requires Linux /proc, POSIX signals and /dev/full")
    if not 0 < args.duration < float("inf"):
        parser.error("duration must be positive and finite")
    if (
        args.cleanup_cancellation
        or args.worker_failure_cancellation
        or args.repeated_cancellation
    ):
        import ctypes

        # Adopt this fixture's orphaned same-group descendants and verify their
        # exit statuses, rather than leaving zombie collection to PID 1.
        if ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) != 0:
            parser.error("Could not enable the cleanup fixture's child subreaper")
    if args.gc_threshold is not None:
        if args.gc_threshold <= 0:
            parser.error("gc-threshold must be positive")
        gc.set_threshold(args.gc_threshold, args.gc_threshold, args.gc_threshold)
    if args.preload_core:
        import embodiedforge

        for name in embodiedforge.__all__:
            getattr(embodiedforge, name)
    # Each full-log close emits an expected secondary diagnostic. Unit tests
    # verify that diagnostic; retain bounded data during the sustained probe.
    logging.getLogger(process.__name__).addHandler(logging.NullHandler())
    logging.getLogger(process.__name__).propagate = False
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "benchmark.py")
    began = time.monotonic()
    cycles = scenarios = 0
    forced_stops = 0
    forced_examples = []
    samples = []
    with managed_run(
        args.output,
        run_identity("benchmark", "validate", "process-faults"),
        metadata={
            "python": sys.executable,
            "python_version": platform.python_version(),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "process_sha256": file_sha256(Path(process.__file__)),
            "gc_threshold": gc.get_threshold(),
            "core_preloaded": args.preload_core,
            "cleanup_cancellation": args.cleanup_cancellation,
            "repeated_cancellation": args.repeated_cancellation,
            "worker_failure_cancellation": args.worker_failure_cancellation,
        },
    ) as record:
        with (args.output / "samples.jsonl").open("x") as stream:
            initial = resources()
            while time.monotonic() - began < args.duration:
                cycle = args.output / "cycle"
                cycle.mkdir()
                if cycles % 2:
                    try:
                        raise ValueError("previous attempt already handled")
                    except ValueError:
                        stopped = run_cycle(
                            cycle,
                            include_cleanup=args.cleanup_cancellation,
                            include_repeated=args.repeated_cancellation,
                            include_worker_failures=args.worker_failure_cancellation,
                        )
                else:
                    stopped = run_cycle(
                        cycle,
                        include_cleanup=args.cleanup_cancellation,
                        include_repeated=args.repeated_cancellation,
                        include_worker_failures=args.worker_failure_cancellation,
                    )
                forced_stops += len(stopped)
                for result in stopped:
                    if len(forced_examples) < 20:
                        forced_examples.append({"cycle": cycles + 1, **result})
                scenarios += len(SIGNALS) * len(STAGES) * 2 + 4 + 2
                if args.cleanup_cancellation:
                    scenarios += len(SIGNALS) * 2 * 2 * 2
                if args.repeated_cancellation:
                    scenarios += 2 * 2 * 3 + 2 * 2 * 2
                if args.worker_failure_cancellation:
                    scenarios += 2 * 2 * 2
                shutil.rmtree(cycle)
                cycles += 1
                gc.collect()
                sample = {
                    "seconds": time.monotonic() - began,
                    "cycles": cycles,
                    "verified_scenarios": scenarios,
                    "forced_startup_stops": forced_stops,
                    "resources": resources(),
                }
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                if not samples or sample["seconds"] - samples[-1]["seconds"] >= 30:
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
            final = resources()
        result = {
            "all_scenarios_passed": True,
            "cycles": cycles,
            "verified_scenarios": scenarios,
            "forced_startup_stops": forced_stops,
            "forced_startup_examples": forced_examples,
            "seconds": time.monotonic() - began,
            "initial_resources": initial,
            "final_resources": final,
            "samples": samples,
        }
        write_json(args.output / "summary.json", result)
        record.update(cycles=cycles, verified_scenarios=scenarios)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
