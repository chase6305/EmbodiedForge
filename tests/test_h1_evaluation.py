"""Metrics must retain terminal states and ignore the next auto-reset episode."""

import json

import numpy as np
import pytest

from embodiedforge import h1
from embodiedforge._h1_metrics import BASIC_COMMANDS, FirstEpisodeMetrics, assess


def test_terminal_frame_is_included_and_reset_episode_is_excluded():
    metrics = FirstEpisodeMetrics(2, 3, 0.02)
    command = np.zeros((2, 3))
    metrics.update([[3, 4, 2], [0, 0, 0]], command, [True, False], [False, False])
    # First environment is now a different episode, whose values must not count.
    metrics.update(
        [[999, 999, 999], [0, 0, 0]], command, [False, False], [False, False]
    )
    metrics.update(
        [[999, 999, 999], [0, 0, 0]], command, [False, False], [False, False]
    )
    report = metrics.report()
    assert report["per_env"]["observed_steps"] == [1, 3]
    assert report["per_env"]["planar_rmse"] == [5, 0]
    assert report["planar_rmse"] == 2.5
    assert report["yaw_rmse"] == 1.0
    assert report["survival_fraction"] == 0.5
    assert report["fall_fraction"] == 0.5
    assert report["mean_observed_seconds"] == 0.04
    assert report["mean_velocity"] == [0.75, 1.0, 0.5]
    assert report["per_env"]["mean_velocity"] == [[3, 4, 2], [0, 0, 0]]


def test_timeout_is_not_a_fall_or_a_full_horizon_survival():
    metrics = FirstEpisodeMetrics(1, 10, 0.02)
    metrics.update([[0, 0, 0]], [[0, 0, 0]], [False], [True])
    report = metrics.report()
    assert report["fall_fraction"] == 0
    assert report["truncation_fraction"] == 1
    assert report["survival_fraction"] == 0


def test_fall_at_last_step_is_still_a_failure():
    metrics = FirstEpisodeMetrics(1, 1, 0.02)
    metrics.update([[0, 0, 0]], [[0, 0, 0]], [True], [False])
    assert metrics.report()["survival_fraction"] == 0


def test_incomplete_rollout_cannot_produce_success_report():
    metrics = FirstEpisodeMetrics(1, 2, 0.02)
    metrics.update([[0, 0, 0]], [[0, 0, 0]], [False], [False])
    with pytest.raises(ValueError, match="Incomplete"):
        metrics.report()


def test_nonfinite_active_state_is_rejected():
    metrics = FirstEpisodeMetrics(1, 2, 0.02)
    with pytest.raises(ValueError, match="Non-finite"):
        metrics.update([[np.nan, 0, 0]], [[0, 0, 0]], [False], [False])


def test_acceptance_is_checked_for_every_seed():
    reports = [
        {"seed": 0, "survival_fraction": 1.0},
        {"seed": 1, "survival_fraction": 0.5},
    ]
    result = assess(reports, {"min_survival_fraction": 0.75})
    assert result["passed"] is False
    assert [check["passed"] for check in result["checks"]] == [True, False]
    assert assess(reports, {})["passed"] is None


def test_mean_speed_does_not_replace_tracking_error():
    metrics = FirstEpisodeMetrics(2, 1, 0.02)
    metrics.update(
        [[1, 0, 0], [-1, 0, 0]], np.zeros((2, 3)), [False, False], [False, False]
    )
    report = metrics.report()
    assert report["mean_velocity"] == [0, 0, 0]
    assert report["planar_rmse"] == 1.0


def test_suite_acceptance_does_not_average_over_cases():
    reports = [
        {"case": "stand", "seed": 0, "survival_fraction": 1.0},
        {"case": "turn_left", "seed": 0, "survival_fraction": 0.5},
    ]
    result = assess(reports, {"min_survival_fraction": 0.75})
    assert result["passed"] is False
    assert [check["case"] for check in result["checks"]] == ["stand", "turn_left"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--seeds", "0", "0"],
        ["--seeds", "-1"],
        ["--velocity", "nan", "0", "0"],
        ["--min-survival-fraction", "1.1"],
        ["--max-planar-rmse", "-0.1"],
        ["--max-yaw-rmse", "inf"],
        ["--suite", "basic", "--velocity", "0", "0", "0"],
    ],
)
def test_invalid_inputs_fail_before_starting_gpu(flags):
    with pytest.raises(SystemExit) as caught:
        h1.main(["evaluate", "--run", "missing", "--output", "unused", *flags])
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "suite,corrupt",
    [
        (None, None),
        ("basic", None),
        ("basic", "missing"),
        ("basic", "velocity"),
        ("basic", "duplicate"),
    ],
)
def test_evaluation_retains_reports_and_rejects_bad_suite(
    tmp_path, monkeypatch, suite, corrupt
):
    import argparse

    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    monkeypatch.setattr(h1, "source_identity", lambda repo: {"revision": h1.REVISION})
    monkeypatch.setattr(
        h1,
        "resume_input",
        lambda run: (
            checkpoint,
            {
                "checkpoint_sha256": h1.sha256(checkpoint),
                "checkpoint_iteration": 8,
            },
        ),
    )

    def run(command, **kwargs):
        path = command[command.index("--output") + 1]
        from pathlib import Path

        reports = []
        for case, velocity in (
            BASIC_COMMANDS if suite else {"custom": [0.5, 0, 0]}
        ).items():
            reports.append(
                {
                    "case": case,
                    "seed": 0,
                    "num_envs": 1,
                    "steps_limit": 10,
                    "runtime": {"checkpoint_iteration": 8},
                    "survival_fraction": 0.0,
                    "velocity_command": velocity,
                    "task": "Isaac-Velocity-Flat-H1-Play-v0",
                    "physics": h1.PHYSICS,
                }
            )
        if corrupt == "missing":
            reports.pop()
        elif corrupt == "velocity":
            reports[1]["velocity_command"] = [123, 0, 0]
        elif corrupt == "duplicate":
            reports[1] = reports[0]
        h1.write_json(
            Path(path),
            {"suite": suite, "seed": 0, "cases": reports} if suite else reports[0],
        )

    monkeypatch.setattr(h1, "run_process", run)
    args = argparse.Namespace(
        repo=tmp_path,
        environment=environment,
        output=tmp_path / "output",
        run=tmp_path,
        num_envs=1,
        steps=10,
        seeds=[0],
        velocity=[0.5, 0, 0],
        timeout=10,
        min_survival_fraction=0.8,
        max_planar_rmse=None,
        max_yaw_rmse=None,
        suite=suite,
    )
    if corrupt:
        with pytest.raises(ValueError, match="evaluation suite|requested inputs"):
            h1.evaluate(args)
        manifest = json.loads((args.output / "run.json").read_text())
        assert manifest["status"] == "failed"
        assert not (args.output / "acceptance.json").exists()
        return
    with pytest.raises(SystemExit) as caught:
        h1.evaluate(args)
    assert caught.value.code == 2
    manifest = json.loads((args.output / "run.json").read_text())
    assert manifest["status"] == "rejected"
    assert manifest["reports"] == ["evaluation-seed-0.json"]
    assert (args.output / "acceptance.json").exists()
    assert "finished_at" in manifest
