"""Independently record benchmark exit, including death after summary publication.

Standard-library-only POSIX supervisor. Its result describes the observed child
exit rather than interpreting a benchmark's own summary. --trace retains Linux
strace instruction pointers, signal senders and exit syscalls (including threads).
The command owns cleanup of descendants that start separate sessions.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def publish(path, data):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def supervise(command, output, *, timeout=None, grace=5, trace=False):
    """Return observed exit evidence; never infer success from child output."""
    if os.name != "posix":
        raise ValueError("POSIX process groups are required")
    for name, value in (("timeout", timeout), ("grace", grace)):
        if value is None and name == "timeout":
            continue
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    if not command:
        raise ValueError("a command is required")
    launched_command = list(command)
    if trace:
        tracer = shutil.which("strace")
        if not tracer:
            raise ValueError("--trace requires strace")
        launched_command = [
            tracer,
            "-f",
            "-ttt",
            "-i",
            "-e",
            "trace=kill,tgkill,tkill,rt_sigaction,exit_group",
            "-e",
            "signal=SIGINT,SIGTERM,SIGHUP,SIGKILL,SIGSEGV,SIGABRT",
            "-o",
            str(Path(output).resolve() / "signals.strace"),
            "--",
            *command,
        ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {
        "command": list(command),
        "launched_command": launched_command,
        "supervisor_pid": os.getpid(),
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "timeout": timeout,
        "grace": grace,
        "trace": trace,
        "process_returncode": None,
        "termination_signal": None,
        "timed_out": False,
        "supervisor_signal": None,
        "forced_stop": False,
        "exit_observed": False,
        "completed_successfully": False,
    }
    saved = {}
    child = None
    stop_at = None
    with (output / "events.jsonl").open("x", encoding="utf-8") as events:

        def event(kind, **values):
            events.write(
                json.dumps(
                    {"kind": kind, "monotonic_ns": time.monotonic_ns(), **values}
                )
                + "\n"
            )
            events.flush()

        def interrupted(signum, frame):
            # Do not write to an interrupted buffered stream from the handler.
            if result["supervisor_signal"] is None:
                result["supervisor_signal"] = signum

        def send(signum):
            event("group_signal_before", pid=child.pid, signal=int(signum))
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                event("group_already_absent", pid=child.pid)

        try:
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                saved[sig] = signal.signal(sig, interrupted)
            publish(output / "started.json", result)
            event("popen_before")
            with (output / "command.log").open("xb") as log:
                child = subprocess.Popen(
                    launched_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                result["pid"] = child.pid
                event("popen_after", pid=child.pid)
                while child.poll() is None:
                    now = time.monotonic()
                    if stop_at is None and (
                        result["supervisor_signal"] is not None
                        or (timeout is not None and now - started >= timeout)
                    ):
                        result["timed_out"] = result["supervisor_signal"] is None
                        send(result["supervisor_signal"] or signal.SIGTERM)
                        stop_at = now + grace
                    elif stop_at is not None and now >= stop_at:
                        result["forced_stop"] = True
                        send(signal.SIGKILL)
                        stop_at = float("inf")
                    time.sleep(0.05)
                code = child.wait()
                result.update(
                    process_returncode=code,
                    termination_signal=-code if code < 0 else None,
                    exit_observed=True,
                )
                event("wait_returned", returncode=code)
        except OSError as error:
            result["supervisor_error"] = {
                "type": type(error).__name__,
                "errno": error.errno,
                "message": str(error),
            }
        finally:
            if child is not None:
                # A command can leave quiet descendants in its owned group.
                send(signal.SIGKILL)
                child.wait()
            for sig, previous in saved.items():
                signal.signal(sig, previous)
            # Cancellation can arrive after wait() but during group cleanup.
            # Decide success only after the protected cleanup has finished.
            result["completed_successfully"] = (
                result["exit_observed"]
                and result["process_returncode"] == 0
                and not result["timed_out"]
                and result["supervisor_signal"] is None
            )
            result["seconds"] = time.monotonic() - started
            result["finished_at"] = datetime.now(timezone.utc).isoformat()
            publish(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--grace", type=float, default=5)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        result = supervise(
            command,
            args.output,
            timeout=args.timeout,
            grace=args.grace,
            trace=args.trace,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result), flush=True)
    if result["completed_successfully"]:
        return 0
    if result["supervisor_signal"] is not None:
        return 128 + result["supervisor_signal"]
    if result["timed_out"]:
        return 124
    code = result["process_returncode"]
    return (128 - code if code < 0 else min(code, 255)) if code is not None else 125


if __name__ == "__main__":
    sys.exit(main())
