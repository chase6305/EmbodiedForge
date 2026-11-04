"""Sustained Linux worker ownership, inherited locks and durable log validation.

Each cycle checks successful, failed, timed-out and interrupted jobs, including
SDK-shaped descendants that detach while retaining stdout. The fixture alone
owns/reaps escaped descendants; the production launcher owns only its job group.
A separate supervisor must verify this interpreter's exit status as well as the
in-workload summary. All retained cycle directories belong to this benchmark.
"""

import argparse
import ctypes
import gc
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import file_sha256
from embodiedforge.runtime.process import run_process
from embodiedforge.runtime.records import managed_run, read_run, run_identity


def resources():
    proc = Path("/proc/self")
    status = dict(
        line.split(":", 1) for line in (proc / "status").read_text().splitlines()
    )
    return {
        "rss_bytes": int(status["VmRSS"].split()[0]) * 1024,
        "threads": int(status["Threads"]),
        "fds": len(list((proc / "fd").iterdir())),
    }


def probe(root, mode):
    try:
        _probe(root, mode)
    finally:
        # Assertion failures must also release this fixture's escaped session.
        path = root / "descendant.pid"
        if path.is_file():
            pid = int(path.read_text())
            try:
                reaped, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            else:
                if not reaped:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)


def _probe(root, mode):
    import fcntl

    detached = mode.startswith("detached-")
    outcome = mode.removeprefix("detached-")

    # Reap the deliberately orphaned fixture descendant instead of relying on PID 1.
    assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
    root.mkdir()
    descendant_code = "import os,signal,time;from pathlib import Path;signal.signal(signal.SIGINT,signal.SIG_IGN);Path('descendant.pid').write_text(str(os.getpid()));time.sleep(30)"
    child_code = """import fcntl,os,signal,subprocess,sys,time
from pathlib import Path
assert os.fstat(int(sys.argv[3])).st_ino==Path('inherited.lock').stat().st_ino
lock=Path('inherited.lock').open('a')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: pass
else: raise AssertionError('trainer did not inherit the run lock')
def stop(*args):
    Path('cleaned').touch()
    raise SystemExit(0)
signal.signal(signal.SIGINT,stop)
subprocess.Popen([sys.executable,'-c',sys.argv[1]],start_new_session=sys.argv[4]=='True',stdout=None if sys.argv[4]=='True' else subprocess.DEVNULL,stderr=subprocess.DEVNULL)
while not Path('descendant.pid').exists(): time.sleep(.005)
Path('ready').touch()
os.write(1, bytes((255,10)))
if sys.argv[2]=='success': raise SystemExit(0)
if sys.argv[2]=='failure': raise SystemExit(7)
time.sleep(30)
"""
    trigger = None
    triggered = []

    def interrupt():
        deadline = time.monotonic() + 8
        while not (root / "ready").exists():
            if time.monotonic() >= deadline:
                return
            time.sleep(0.005)
        triggered.append(True)
        os.kill(os.getpid(), signal.SIGTERM)

    caught = None
    try:
        with managed_run(root, run_identity("probe", mode)):
            if outcome == "interrupt":
                trigger = threading.Thread(target=interrupt)
                trigger.start()
            with (root / "inherited.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_process(
                    [
                        sys.executable,
                        "-c",
                        child_code,
                        descendant_code,
                        outcome,
                        str(lock.fileno()),
                        str(detached),
                    ],
                    cwd=root,
                    env=os.environ.copy(),
                    log_path=root / "worker.log",
                    echo=True,
                    timeout=1.0 if outcome == "timeout" else 10,
                    pass_fds=(lock.fileno(),),
                )
    except BaseException as error:
        caught = error
    finally:
        if trigger is not None:
            trigger.join(timeout=9)
    assert threading.enumerate() == [threading.main_thread()]
    assert (root / "worker.log").read_bytes() == bytes((255, 10))
    expected = {
        "success": "complete",
        "failure": "failed",
        "timeout": "timed_out",
        "interrupt": "interrupted",
    }[outcome]
    observed = read_run(root)
    assert observed["status"] == expected, (mode, observed, caught)
    if outcome == "success":
        assert caught is None
    elif outcome == "failure":
        assert (
            isinstance(caught, subprocess.CalledProcessError) and caught.returncode == 7
        ), caught
    elif outcome == "timeout":
        assert isinstance(caught, subprocess.TimeoutExpired), caught
    else:
        assert (
            triggered
            and isinstance(caught, KeyboardInterrupt)
            and caught.signum == signal.SIGTERM
        ), caught
    if outcome in ("timeout", "interrupt"):
        assert (root / "cleaned").is_file()
    pid = int((root / "descendant.pid").read_text())
    if detached:
        # This fixture deliberately owns a second session. Verify the launcher
        # left that session alone, then kill/reap it using its recorded PID.
        os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    deadline = time.monotonic() + 3
    reaped = 0
    while not reaped and time.monotonic() < deadline:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if not reaped:
            time.sleep(0.005)
    if not reaped:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        raise AssertionError("owned descendant survived process-group cleanup")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise AssertionError("reaped descendant still exists")
    with (root / "inherited.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(
        root / "verified.json",
        {
            "mode": mode,
            "status": observed["status"],
            "descendant_reaped": True,
            "detached_fixture_cleaned": detached,
            "lock_released": True,
        },
    )


OUTCOMES = ("success", "failure", "timeout", "interrupt")
MODES = (*OUTCOMES, *(f"detached-{outcome}" for outcome in OUTCOMES))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODES)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument(
        "--preload-core",
        action="store_true",
        help="Load public core exports in the parent, modeling a NumPy host launcher",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This probe uses Linux subreapers and /proc resource samples")
    if not 0 < args.duration < float("inf"):
        parser.error("duration must be positive and finite")
    args.output = args.output.resolve()
    if args.worker:
        probe(args.output, args.worker)
        return
    if args.preload_core:
        import embodiedforge

        for name in embodiedforge.__all__:
            getattr(embodiedforge, name)
    args.output.mkdir(parents=True)
    shutil.copyfile(__file__, args.output / "benchmark.py")
    began = time.monotonic()
    samples = []
    cycles = 0
    modes = MODES
    with managed_run(
        args.output,
        run_identity("benchmark", "validate", "process-lifecycle"),
        metadata={
            "python": sys.executable,
            "python_version": platform.python_version(),
            "benchmark_sha256": file_sha256(Path(__file__)),
            "process_sha256": file_sha256(
                Path(sys.modules[run_process.__module__].__file__)
            ),
            "numpy_loaded": "numpy" in sys.modules,
            "core_preloaded": args.preload_core,
        },
    ) as record:
        initial = resources()
        with (args.output / "samples.jsonl").open("x") as log:
            while time.monotonic() - began < args.duration:
                cycle = args.output / "cycle"
                cycle.mkdir()
                for mode in modes:
                    run_process(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            mode,
                            "--output",
                            str(cycle / mode),
                        ],
                        cwd=args.output,
                        env={
                            **os.environ,
                            "PYTHONPATH": str(
                                Path(sys.modules["embodiedforge"].__file__)
                                .resolve()
                                .parent.parent
                            ),
                        },
                        log_path=cycle / (mode + ".log"),
                        echo=False,
                        timeout=20,
                    )
                    assert json.loads((cycle / mode / "verified.json").read_text())[
                        "descendant_reaped"
                    ]
                shutil.rmtree(cycle)
                cycles += 1
                gc.collect()
                sample = {
                    "seconds": time.monotonic() - began,
                    "cycles": cycles,
                    "verified_scenarios": cycles * len(modes),
                    "resources": resources(),
                }
                log.write(json.dumps(sample) + "\n")
                log.flush()
                if not samples or sample["seconds"] - samples[-1]["seconds"] >= 30:
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
        result = {
            "scope": "Linux captured invalid UTF-8, joined log threads, inherited locks, real process groups and escaped stdout holders; fixture reaps all descendants",
            "cycles": cycles,
            "verified_scenarios": cycles * len(modes),
            "seconds": time.monotonic() - began,
            "initial_resources": initial,
            "final_resources": resources(),
            "samples": samples,
            "all_scenarios_passed": True,
        }
        write_json(args.output / "summary.json", result)
        record.update(cycles=cycles, verified_scenarios=cycles * len(modes))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
