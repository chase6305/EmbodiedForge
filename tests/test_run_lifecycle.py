"""Lifecycle publication must retain primary failures and truthful terminal states."""

import json
import subprocess
from datetime import datetime

import pytest

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.records import (
    RunRejected,
    managed_run,
    read_run,
    run_identity,
    run_lifecycle,
)


@pytest.mark.parametrize("status", ["complete", "rejected"])
def test_successful_body_preserves_explicit_outcome_and_records_timing(
    tmp_path, status
):
    with managed_run(
        tmp_path, run_identity("example", "evaluate"), metadata={"input": "verified"}
    ) as record:
        running = read_run(tmp_path)
        assert running["status"] == "running"
        assert not running["terminal"]
        record["status"] = status
        record["result"] = {"path": "evaluation.json"}
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == status
    assert saved["input"] == "verified"
    assert saved["result"]["path"] == "evaluation.json"
    assert saved["seconds"] >= 0
    assert datetime.fromisoformat(saved["finished_at"]) >= datetime.fromisoformat(
        saved["started_at"]
    )


@pytest.mark.parametrize(
    "failure,status",
    [
        (RuntimeError("worker failed"), "failed"),
        (KeyboardInterrupt(), "interrupted"),
        (subprocess.TimeoutExpired("worker", 1), "timed_out"),
        (SystemExit(3), "failed"),
    ],
)
def test_failure_is_reported_and_the_same_exception_is_reraised(
    tmp_path, failure, status
):
    with pytest.raises(type(failure)) as caught:
        with managed_run(tmp_path, run_identity("example", "train")):
            raise failure
    assert caught.value is failure
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == status
    assert type(failure).__name__ in saved["error"]


@pytest.mark.parametrize(
    "failure", [RuntimeError("worker failed"), KeyboardInterrupt(), None]
)
def test_final_write_error_never_hides_primary_failure(tmp_path, caplog, failure):
    disk_error = OSError("disk full")

    def publish(path, record):
        if record["status"] != "running":
            raise disk_error
        write_json(path, record)

    expected = failure if failure is not None else disk_error
    with pytest.raises(type(expected)) as caught:
        with managed_run(tmp_path, run_identity("example", "train"), writer=publish):
            if failure is not None:
                raise failure
    assert caught.value is expected
    assert read_run(tmp_path)["status"] == "running"
    if failure is not None:
        assert "preserving its original error" in caplog.text


@pytest.mark.parametrize("stage", ["finalize", "publish"])
@pytest.mark.parametrize(
    "failure", [OSError("new run cleanup failed"), KeyboardInterrupt()]
)
def test_handled_outer_exception_does_not_hide_this_runs_finalization_failure(
    tmp_path, stage, failure
):
    path = tmp_path / "run.json"
    record = {"status": "running", "run_contract": run_identity("example", "retry")}

    def publish(current):
        if stage == "publish" and current["status"] == "complete":
            raise failure
        write_json(path, current)

    def finalize(current):
        if stage == "finalize":
            raise failure

    try:
        raise ValueError("previous attempt already handled")
    except ValueError:
        with pytest.raises(type(failure)) as caught:
            with run_lifecycle(record, publish, finalize=finalize):
                pass
    assert caught.value is failure
    expected = "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
    assert read_run(tmp_path)["status"] == expected


def test_existing_record_is_not_overwritten(tmp_path):
    path = tmp_path / "run.json"
    write_json(path, {"old": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        with managed_run(tmp_path, run_identity("example", "train")):
            pytest.fail("must not enter an existing run")
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "field", ["run_contract", "status", "started_at", "finished_at", "seconds"]
)
def test_metadata_cannot_override_lifecycle_fields(tmp_path, field):
    with pytest.raises(ValueError, match="lifecycle fields"):
        with managed_run(
            tmp_path, run_identity("example", "train"), metadata={field: "override"}
        ):
            pytest.fail("invalid metadata must fail before publication")
    assert not list(tmp_path.iterdir())


def test_successful_body_cannot_leave_a_running_or_unknown_terminal_state(tmp_path):
    with pytest.raises(ValueError, match="complete or rejected"):
        with managed_run(tmp_path, run_identity("example", "train")) as record:
            record["status"] = "preparing"
    assert read_run(tmp_path)["status"] == "failed"


def test_domain_rejection_is_not_reported_as_a_worker_failure(tmp_path):
    from embodiedforge.microduck import EvaluationRejected

    rejection = EvaluationRejected("policy missed the requested survival threshold")
    assert isinstance(rejection, RunRejected)
    with pytest.raises(EvaluationRejected) as caught:
        with managed_run(tmp_path, run_identity("example", "evaluate")):
            raise rejection
    assert caught.value is rejection
    assert read_run(tmp_path)["status"] == "rejected"


def test_process_interrupt_retains_its_signal_in_the_record(tmp_path):
    import signal

    from embodiedforge.runtime.process import ProcessInterrupted

    with pytest.raises(ProcessInterrupted):
        with managed_run(tmp_path, run_identity("example", "evaluate")):
            raise ProcessInterrupted(signal.SIGTERM)
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == "interrupted"
    assert saved["interrupt_signal"] == signal.SIGTERM


@pytest.mark.parametrize("primary", [None, RuntimeError("worker failed")])
@pytest.mark.parametrize("publication_fails", [False, True])
def test_terminal_metadata_failure_still_publishes_and_preserves_primary_error(
    tmp_path, caplog, primary, publication_fails
):
    path = tmp_path / "run.json"
    record = {"status": "preparing", "run_contract": run_identity("sdk", "train")}
    scan_error = OSError("checkpoint directory unreadable")

    def publish(current):
        if publication_fails and current["status"] not in ("preparing", "running"):
            raise OSError("final publication failed")
        write_json(path, current)

    def finalize(current):
        assert current["status"] == ("failed" if primary else "complete")
        raise scan_error

    expected = primary if primary is not None else scan_error
    with pytest.raises(type(expected)) as caught:
        with run_lifecycle(record, publish, finalize=finalize):
            record["status"] = "running"
            if primary is not None:
                raise primary
    assert caught.value is expected
    assert record["status"] == "failed"
    if not publication_fails:
        saved = json.loads(path.read_text())
        assert saved["status"] == "failed"
        assert str(expected) in saved["error"]
    if primary is not None:
        assert "preserving its original error" in caplog.text


def test_new_lifecycle_retains_workflow_fields_and_rejects_unfinished_preparation(
    tmp_path,
):
    path = tmp_path / "run.json"
    record = {"status": "preparing", "schema": 7, "backend": "custom"}
    with pytest.raises(ValueError, match="complete or rejected"):
        with run_lifecycle(record, lambda current: write_json(path, current)):
            pass
    saved = json.loads(path.read_text())
    assert saved["schema"] == 7 and saved["backend"] == "custom"
    assert saved["status"] == "failed"


@pytest.mark.parametrize("after_commit", [False, True])
def test_terminal_publication_failure_is_corrected_without_losing_its_exception(
    tmp_path, after_commit
):
    import signal

    from embodiedforge.runtime.process import ProcessInterrupted

    failure = ProcessInterrupted(signal.SIGTERM)
    calls = []

    def publish(path, record):
        calls.append(record["status"])
        if record["status"] == "complete":
            if after_commit:
                write_json(path, record)
            raise failure
        write_json(path, record)

    with pytest.raises(ProcessInterrupted) as caught:
        with managed_run(tmp_path, run_identity("sdk", "train"), writer=publish):
            pass
    assert caught.value is failure
    assert calls == ["running", "complete", "interrupted"]
    saved = json.loads((tmp_path / "run.json").read_text())
    assert saved["status"] == "interrupted"
    assert saved["interrupt_signal"] == signal.SIGTERM


@pytest.mark.parametrize("after_commit", [False, True])
def test_startup_publication_failure_never_enters_body_and_finalizes_best_effort(
    tmp_path, after_commit
):
    import signal

    from embodiedforge.runtime.process import ProcessInterrupted

    failure = (
        ProcessInterrupted(signal.SIGTERM)
        if after_commit
        else OSError("initial write failed")
    )
    calls = []

    def publish(path, record):
        calls.append(record["status"])
        if len(calls) == 1:
            if after_commit:
                write_json(path, record)
            raise failure
        write_json(path, record)

    with pytest.raises(type(failure)) as caught:
        with managed_run(tmp_path, run_identity("example", "evaluate"), writer=publish):
            pytest.fail("body must not start after an initial publication failure")
    assert caught.value is failure
    saved = json.loads((tmp_path / "run.json").read_text())
    expected = "interrupted" if after_commit else "failed"
    assert saved["status"] == expected
    assert calls == ["running", expected]
    if after_commit:
        assert saved["interrupt_signal"] == signal.SIGTERM


@pytest.mark.parametrize("stage", ["startup", "body"])
def test_real_sigterm_finalizes_record_in_a_child_process(tmp_path, stage):
    import os
    import signal
    import sys
    import textwrap
    import time
    from pathlib import Path

    if os.name != "posix":
        pytest.skip("POSIX signal delivery contract")
    script = textwrap.dedent("""
        import signal, sys
        from pathlib import Path
        from embodiedforge.runtime.artifacts import write_json
        from embodiedforge.runtime.process import ProcessInterrupted, termination_signals
        from embodiedforge.runtime.records import managed_run, run_identity

        root, stage = Path(sys.argv[1]), sys.argv[2]
        def wait_for_signal():
            (root / 'ready').touch()
            signal.pause()
        def publish(path, record):
            write_json(path, record)
            if stage == 'startup' and record['status'] == 'running':
                wait_for_signal()
        try:
            with termination_signals():
                with managed_run(root, run_identity('example', 'evaluate'), writer=publish):
                    (root / 'entered').touch()
                    wait_for_signal()
        except ProcessInterrupted as error:
            sys.exit(128 + error.signum)
    """)
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script, str(tmp_path), stage],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists():
            assert process.poll() is None, process.communicate(timeout=5)
            assert time.monotonic() < deadline, "child never reached signal barrier"
            time.sleep(0.01)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 128 + signal.SIGTERM, (stdout, stderr)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["interrupt_signal"] == signal.SIGTERM
    assert "ProcessInterrupted" in manifest["error"]
    assert (tmp_path / "entered").exists() == (stage == "body")
    assert not list(tmp_path.glob("*.tmp"))
