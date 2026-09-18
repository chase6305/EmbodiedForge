"""Run status must distinguish a live launcher, stale PID, and unknown host."""

import pytest

from embodiedforge import _microduck_worker as worker


def test_current_launcher_matches_proc_identity():
    identity = worker.launcher_identity()
    if "start_ticks" not in identity or "boot_id" not in identity:
        pytest.skip("Requires Linux procfs")
    assert worker.launcher_status(identity) == {
        "alive": True,
        "reason": "identity_matches",
    }
    identity["start_ticks"] += 1
    assert worker.launcher_status(identity) == {"alive": False, "reason": "pid_reused"}


def test_foreign_and_legacy_launchers_are_unknown():
    assert worker.launcher_status(None)["alive"] is None
    assert worker.launcher_status({"hostname": "elsewhere.invalid"})["alive"] is None
    identity = worker.launcher_identity()
    identity.pop("start_ticks", None)
    assert worker.launcher_status(identity)["alive"] is None


@pytest.mark.parametrize("state", ["Z", "X"])
def test_zombie_is_not_a_running_training_job(monkeypatch, state):
    identity = worker.launcher_identity()
    if "boot_id" not in identity:
        pytest.skip("Requires Linux procfs")
    monkeypatch.setattr(
        worker, "_process_state", lambda pid: (state, identity["start_ticks"])
    )
    assert worker.launcher_status(identity) == {"alive": False, "reason": "exited"}


def test_missing_process_is_distinguished_from_permission_error(monkeypatch):
    identity = worker.launcher_identity()
    if "boot_id" not in identity:
        pytest.skip("Requires Linux procfs")

    def missing(pid):
        raise FileNotFoundError("gone")

    monkeypatch.setattr(worker, "_process_state", missing)
    assert worker.launcher_status(identity) == {"alive": False, "reason": "exited"}

    def denied(pid):
        raise PermissionError("not readable")

    monkeypatch.setattr(worker, "_process_state", denied)
    assert worker.launcher_status(identity) == {
        "alive": None,
        "reason": "proc_unreadable",
    }
