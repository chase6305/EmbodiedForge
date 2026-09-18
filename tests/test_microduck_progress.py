"""File-only progress must not mistake stale manifests or stray files for recovery."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiedforge import _microduck_worker as worker


def test_progress_cli_needs_no_source_checkout_or_environment_stamp(
    tmp_path, monkeypatch
):
    from embodiedforge import _microduck_native as native
    from embodiedforge import microduck

    environment = tmp_path / "environment"
    python = environment / "bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    run_dir = tmp_path / "training"

    def unexpected_source(*args):
        pytest.fail("Progress must not validate current training sources")

    monkeypatch.setattr(native, "source_identity", unexpected_source)
    monkeypatch.setattr(microduck, "source_identity", unexpected_source)
    calls = []
    monkeypatch.setattr(microduck, "run", lambda command, **kw: calls.append(command))
    args = ["progress", "--env-dir", str(environment), "--run", str(run_dir)]
    microduck.main(args)
    # Legacy logs remain readable after their upstream checkout is removed.
    microduck.main(args + ["--repo", str(tmp_path / "removed-checkout")])
    assert (
        calls
        == [[str(python), "-I", str(microduck.WORKER), "progress", str(run_dir)]] * 2
    )


@pytest.fixture
def progress(tmp_path, monkeypatch):
    log = tmp_path / "logs/rsl_rl/microduck/current"
    log.mkdir(parents=True)
    (log / "events.out.tfevents.fixture").touch()
    scalars = {
        "Loss/value": [
            SimpleNamespace(step=i, value=0.1, wall_time=1.0) for i in (0, 1)
        ],
        "Perf/collection_time": [SimpleNamespace(step=1, value=0.8)],
        "Perf/learning_time": [SimpleNamespace(step=1, value=0.2)],
    }

    class Accumulator:
        def __init__(self, *a, **kw):
            pass

        def Reload(self):
            pass

        def Tags(self):
            return {"scalars": list(scalars)}

        def Scalars(self, tag):
            return scalars[tag]

    monkeypatch.setitem(
        sys.modules,
        "tensorboard.backend.event_processing.event_accumulator",
        SimpleNamespace(EventAccumulator=Accumulator),
    )
    manifest = {
        "workflow": "train",
        "status": "running",
        "phase": "train",
        "commands": [["python", "--agent.max-iterations", "5"]],
    }
    state = {"alive": True, "reason": "identity_matches"}
    monkeypatch.setattr(worker, "launcher_status", lambda identity: state.copy())

    def read():
        (tmp_path / "run.json").write_text(json.dumps(manifest))
        return worker.training_progress(tmp_path)

    return read, manifest, state, scalars, log


def test_dead_launcher_does_not_keep_running_eta(progress):
    read, manifest, state, _, log = progress
    (log / "model_1.pt").write_bytes(b"unvalidated candidate")
    assert read()["estimated_remaining_seconds"] == pytest.approx(3)
    state.update(alive=False, reason="exited")
    result = read()
    assert result["run_status"] == "running"
    assert result["effective_status"] == "orphaned"
    assert result["estimated_remaining_seconds"] is None
    assert result["observed_updates"] == 2
    assert result["latest_checkpoint"].endswith("model_1.pt")
    assert result["checkpoint_validation"] == "not_performed"
    assert json.loads((log.parents[3] / "run.json").read_text())["status"] == "running"


@pytest.mark.parametrize(
    "phase", ["check", "checkpoint", "snapshot", "metrics", "export", "validate"]
)
def test_nontraining_phases_have_no_training_eta(progress, phase):
    read, manifest, _, _, _ = progress
    manifest["phase"] = phase
    manifest.update(iterations=5, commands=[])
    result = read()
    assert result["requested_updates"] == 5
    assert result["estimated_remaining_seconds"] is None


@pytest.mark.parametrize("status", ["failed", "interrupted", "complete"])
def test_terminal_status_not_reclassified_as_orphaned(progress, status):
    read, manifest, state, _, _ = progress
    manifest["status"] = status
    state.update(alive=False, reason="exited")
    result = read()
    assert result["effective_status"] == status
    assert result["estimated_remaining_seconds"] == (
        0.0 if status == "complete" else None
    )


def test_unknown_launcher_does_not_claim_process_is_dead(progress):
    read, _, state, _, _ = progress
    state.update(alive=None, reason="different_host")
    result = read()
    assert (
        result["effective_status"] == "running"
        and result["launcher_status"]["alive"] is None
    )


def test_unreliable_iteration_logs_do_not_produce_eta(progress):
    read, manifest, _, scalars, _ = progress
    scalars["Perf/learning_time"][0].value = -2
    assert read()["estimated_remaining_seconds"] is None
    scalars["Perf/learning_time"][0].value = 0.2
    for steps in ([], [0, 2], [1, 2], [-1, 0], list(range(6))):
        scalars["Loss/value"] = [
            SimpleNamespace(step=step, value=0.1, wall_time=1.0) for step in steps
        ]
        result = read()
        assert result["estimated_remaining_seconds"] is None
        assert result["observed_updates"] == len(steps)

    manifest["resume"] = {"iteration": 42}
    scalars["Loss/value"] = [
        SimpleNamespace(step=step, value=0.1, wall_time=1.0) for step in (42, 43)
    ]
    for tag in ("Perf/collection_time", "Perf/learning_time"):
        scalars[tag][0].step = 43
        # Timing from a different update cannot distort this run's ETA.
        scalars[tag].append(SimpleNamespace(step=99, value=1000.0))
    assert read()["estimated_remaining_seconds"] == pytest.approx(3)


def test_checkpoint_discovery_ignores_nonmodels_and_external_links(progress, tmp_path):
    read, _, _, _, log = progress
    (log / "model_backup.pt").touch()
    (log / "model_5.pt.dead.tmp").touch()
    (log / "model_300.pt").mkdir()
    (log / "model_999.pt").symlink_to(Path(__file__).resolve())
    (log / "model_40.pt").touch()
    (log / "model_9.pt").touch()
    assert read()["latest_checkpoint"] == str(log / "model_40.pt")


def test_progress_does_not_mix_training_sessions(progress):
    read, _, _, _, log = progress
    other = log.parent / "other"
    other.mkdir()
    (other / "model_900.pt").touch()
    # One log directory and one checkpoint directory can still disagree.
    with pytest.raises(ValueError, match="Ambiguous training session"):
        read()
    (log / "model_1.pt").touch()
    with pytest.raises(ValueError, match="Ambiguous training session"):
        read()
