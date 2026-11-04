"""Trace real worker cancellation, acknowledgement and force-kill fallback.

Requires Linux /proc and /dev/full. Run under an external supervisor to verify
interpreter shutdown. --worker-trace adds a separate inherited acknowledgement
pipe. --wakeup-trace also records the C handler notification on that pipe
using signal.set_wakeup_fd; omit both flags to keep the original worker code unchanged. Successful cycle
traces are discarded after retaining the first cycle; forced and delayed cases
remain available for inspection.

--observe-late-handler waits up to 25 extra seconds after the normal ten-second
grace expires. This is a diagnostic fixture override, not production behavior.
It can distinguish a delayed Python handler from an unresponsive worker.
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
import time
from pathlib import Path

import process_fault_soak as faults

process = faults.process


def proc_state(pid):
    root = os.path.join("/proc", str(pid))
    values = {}
    for name in ("status", "stat", "wchan", "syscall"):
        try:
            with open(os.path.join(root, name), encoding="utf-8") as stream:
                values[name] = stream.read()
        except OSError as error:
            values[name] = {"error": type(error).__name__, "errno": error.errno}
    return values


def startup(
    root,
    signum,
    stage,
    echo,
    worker_trace,
    observe_late_handler=False,
    wakeup_trace=False,
):
    original_popen, original_killpg = subprocess.Popen, os.killpg
    events = []
    children = []
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    acknowledgements = bytearray()

    def collect_acknowledgements():
        while True:
            try:
                chunk = os.read(read_fd, 65536)
            except BlockingIOError:
                break
            if not chunk:
                break
            acknowledgements.extend(chunk)
            events.append(
                {
                    "kind": "acknowledgement_observed",
                    "monotonic_ns": time.monotonic_ns(),
                    "raw_hex": chunk.hex(),
                }
            )

    def emit(kind, **values):
        collect_acknowledgements()
        events.append({"kind": kind, "monotonic_ns": time.monotonic_ns(), **values})

    def killpg(pid, sig):
        emit("killpg_before", pid=pid, signal=int(sig), state=proc_state(pid))
        try:
            original_killpg(pid, sig)
        except BaseException as error:
            emit(
                "killpg_error",
                pid=pid,
                signal=int(sig),
                error=type(error).__name__,
                errno=getattr(error, "errno", None),
            )
            raise
        else:
            emit("killpg_after", pid=pid, signal=int(sig))

    class Observed(original_popen):
        def __init__(self, *args, **kwargs):
            if worker_trace:
                command = list(args[0])
                code = command[-1]
                assert code.count("Path('ready').touch()") == 1
                assert code.count("def stop(*args):\n") == 1
                prefix = f"os.write({write_fd},(str(time.monotonic_ns())+' ready '+str(os.getpid())+'\\n').encode())\n"
                code = code.replace(
                    "Path('ready').touch()", prefix + "Path('ready').touch()"
                )
                code = code.replace(
                    "def stop(*args):\n",
                    f"def stop(*args):\n    os.write({write_fd},(str(time.monotonic_ns())+' handler '+str(os.getpid())+'\\n').encode())\n",
                )
                if wakeup_trace:
                    code = code.replace(
                        "signal.signal(signal.SIGINT,stop)",
                        f"signal.signal(signal.SIGINT,stop)\nos.set_blocking({write_fd},False)\nsignal.set_wakeup_fd({write_fd})",
                    )
                command[-1] = code
                args = (command, *args[1:])
                kwargs["pass_fds"] = (*kwargs.get("pass_fds", ()), write_fd)
            emit("popen_before")
            super().__init__(*args, **kwargs)
            children.append(self)
            emit("popen_after", pid=self.pid)

        def wait(self, timeout=None):
            emit(
                "wait_before", pid=self.pid, timeout=timeout, returncode=self.returncode
            )
            try:
                result = super().wait(timeout=timeout)
            except BaseException as error:
                emit(
                    "wait_error",
                    pid=self.pid,
                    error=type(error).__name__,
                    timeout=timeout,
                    state=proc_state(self.pid),
                )
                if (
                    observe_late_handler
                    and isinstance(error, subprocess.TimeoutExpired)
                    and timeout == 10
                ):
                    try:
                        late = super().wait(timeout=25)
                        emit("late_wait_after", pid=self.pid, returncode=late)
                    except subprocess.TimeoutExpired:
                        emit(
                            "late_wait_timeout",
                            pid=self.pid,
                            state=proc_state(self.pid),
                        )
                raise
            emit("wait_after", pid=self.pid, timeout=timeout, returncode=result)
            return result

    subprocess.Popen, os.killpg = Observed, killpg
    result = None
    error = None
    try:
        result = faults.startup(root, signum, stage, echo)
        return result
    except BaseException as caught:
        error = {"type": type(caught).__name__, "message": str(caught)}
        raise
    finally:
        subprocess.Popen, os.killpg = original_popen, original_killpg
        os.close(write_fd)
        collect_acknowledgements()
        os.close(read_fd)
        wakeups = [byte for byte in acknowledgements if byte < 32 and byte != 10]
        text_acknowledgements = bytes(
            byte for byte in acknowledgements if byte >= 32 or byte == 10
        ).decode("ascii", errors="replace")
        data = {
            "signal": int(signum),
            "stage": stage,
            "echo": echo,
            "worker_trace": worker_trace,
            "result": result,
            "error": error,
            "events": events,
            "worker_acknowledgements": text_acknowledgements,
            "worker_wakeup_notifications": wakeups,
            "worker_raw_acknowledgements_hex": acknowledgements.hex(),
            "wakeup_trace": wakeup_trace,
            "child_returncodes": [
                {"pid": child.pid, "returncode": child.returncode} for child in children
            ],
        }
        (root / "signal-trace.json").write_text(json.dumps(data, indent=2) + "\n")
        if worker_trace and result is not None:
            observed = [line.split() for line in text_acknowledgements.splitlines()]
            assert any(line[1] == "ready" for line in observed), data
            if result["graceful"]:
                assert any(line[1] == "handler" for line in observed), data
            if wakeup_trace:
                assert signal.SIGINT in wakeups, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-trace", action="store_true")
    parser.add_argument("--wakeup-trace", action="store_true")
    parser.add_argument("--preload-sdk", action="store_true")
    parser.add_argument("--observe-late-handler", action="store_true")
    parser.add_argument("--gc-threshold", type=int, default=10)
    args = parser.parse_args()
    if sys.platform != "linux" or not Path("/dev/full").exists():
        parser.error("Requires Linux /proc, POSIX signals and /dev/full")
    if not 0 < args.duration < float("inf"):
        parser.error("duration must be positive and finite")
    if args.gc_threshold <= 0:
        parser.error("gc-threshold must be positive")
    if args.wakeup_trace and not args.worker_trace:
        parser.error("--wakeup-trace requires --worker-trace")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "probe.py")
    packages = {}
    if args.preload_sdk:
        import mujoco
        import numpy
        import torch

        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        packages = {
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "numpy": numpy.__version__,
            "cuda_initialized": False,
        }
    import embodiedforge

    for name in embodiedforge.__all__:
        getattr(embodiedforge, name)
    gc.set_threshold(args.gc_threshold, args.gc_threshold, args.gc_threshold)
    logging.getLogger(process.__name__).addHandler(logging.NullHandler())
    logging.getLogger(process.__name__).propagate = False
    configuration = {
        "python": sys.executable,
        "version": platform.python_version(),
        "packages": packages,
        "worker_trace": args.worker_trace,
        "wakeup_trace": args.wakeup_trace,
        "gc_threshold": gc.get_threshold(),
        "allocator": os.environ.get("PYTHONMALLOC"),
        "parent_pid": os.getpid(),
        "initial_resources": faults.resources(),
        "observe_late_handler": args.observe_late_handler,
        "probe_sha256": faults.file_sha256(Path(__file__)),
        "fixture_sha256": faults.file_sha256(Path(faults.__file__)),
        "initial_signal_handlers": {
            str(s): str(signal.getsignal(s)) for s in faults.SIGNALS
        },
        "production_process_sha256": faults.file_sha256(Path(process.__file__)),
    }
    faults.write_json(args.output / "configuration.json", configuration)
    began = time.monotonic()
    cycles = scenarios = forced = late_handlers = 0
    samples = []
    with faults.managed_run(
        args.output, faults.run_identity("benchmark", "validate", "signal-observation")
    ) as record:
        with (args.output / "samples.jsonl").open("x") as stream:
            while time.monotonic() - began < args.duration:
                cycle = args.output / "cycle"
                cycle.mkdir()

                def run(cycle, cycle_number):
                    nonlocal scenarios, forced, late_handlers
                    for sig in faults.SIGNALS:
                        for stage in faults.STAGES:
                            for echo in (False, True):
                                case = cycle / f"{sig}-{stage}-{echo}"
                                result = startup(
                                    case,
                                    sig,
                                    stage,
                                    echo,
                                    args.worker_trace,
                                    args.observe_late_handler,
                                    args.wakeup_trace,
                                )
                                trace = json.loads(
                                    (case / "signal-trace.json").read_text()
                                )
                                if any(
                                    e["kind"] == "late_wait_after"
                                    for e in trace["events"]
                                ):
                                    late_handlers += 1
                                    shutil.copytree(
                                        case,
                                        args.output
                                        / f"late-{cycle_number}-{sig}-{stage}-{echo}",
                                    )
                                scenarios += 1
                                if not result["graceful"]:
                                    forced += 1
                                    destination = (
                                        args.output
                                        / f"forced-{cycle_number}-{sig}-{stage}-{echo}"
                                    )
                                    shutil.copytree(case, destination)
                                    print(
                                        json.dumps({"forced_stop": str(destination)}),
                                        flush=True,
                                    )
                    for outcome in ("success", "failure", "timeout", "interrupt"):
                        faults.disk_full(cycle / outcome, outcome)
                        scenarios += 1
                    for stage in ("finalize", "publish"):
                        faults.finalization_failure(cycle / stage, stage)
                        scenarios += 1

                if cycles % 2:
                    try:
                        raise ValueError("previous attempt already handled")
                    except ValueError:
                        run(cycle, cycles)
                else:
                    run(cycle, cycles)
                if cycles == 0:
                    shutil.copytree(cycle, args.output / "first-cycle")
                shutil.rmtree(cycle)
                cycles += 1
                gc.collect()
                sample = {
                    "seconds": time.monotonic() - began,
                    "cycles": cycles,
                    "verified_scenarios": scenarios,
                    "forced_startup_stops": forced,
                    "late_handlers": late_handlers,
                    "resources": faults.resources(),
                }
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                if not samples or sample["seconds"] - samples[-1]["seconds"] >= 30:
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
            summary = {
                "all_scenarios_passed": True,
                "seconds": time.monotonic() - began,
                "cycles": cycles,
                "verified_scenarios": scenarios,
                "forced_startup_stops": forced,
                "late_handlers": late_handlers,
                "final_resources": faults.resources(),
                "samples": samples,
            }
            faults.write_json(args.output / "summary.json", summary)
            record.update(
                cycles=cycles,
                verified_scenarios=scenarios,
                forced_startup_stops=forced,
                late_handlers=late_handlers,
            )
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
