"""A successful subprocess is insufficient evidence of a complete evaluation."""

import json
from argparse import Namespace
from pathlib import Path

import pytest
from microduck_report_fixture import evaluation_report

from embodiedforge import microduck


# Exercise each invalid report once; batch cases only check failure propagation.
@pytest.mark.parametrize(
    "seeds,invalid",
    [
        (None, invalid)
        for invalid in (
            "seed",
            "steps_per_env",
            "num_envs",
            "checkpoint",
            "task",
            "velocity",
            "pushes",
            "samples",
            "metrics",
            "finite",
            "nan_state",
            "metadata",
            "list",
            "null",
            "nan",
            "infinity",
            "overflow",
        )
    ]
    + [([0, 1], "seed"), ([0, 1], "nan")],
)
def test_invalid_reports_cannot_complete_even_without_acceptance_thresholds(
    tmp_path, monkeypatch, seeds, invalid
):
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    args = Namespace(
        checkpoint=checkpoint,
        onnx=None,
        output=tmp_path / "evaluation",
        num_envs=2,
        steps=4,
        seed=0,
        seeds=seeds,
        velocity=[0.2, 0, 0],
        no_pushes=True,
    )
    calls = []

    def run(command, **kwargs):
        if command[3] != "evaluate":
            return
        calls.append(int(command[7]))
        report = evaluation_report(command)
        if invalid in ("seed", "steps_per_env", "num_envs"):
            report[invalid] += 1
        elif invalid in ("checkpoint", "task"):
            report[invalid] = "wrong"
        elif invalid == "velocity":
            report["conditions"]["velocity_body_frame"] = [0, 0, 0]
        elif invalid == "pushes":
            report["conditions"]["pushes_enabled"] = True
        elif invalid == "samples":
            report["velocity_tracking"]["samples"] -= 1
        elif invalid == "metrics":
            del report["velocity_tracking"]
        elif invalid == "finite":
            report["finite_observations_actions_rewards"] = False
        elif invalid == "nan_state":
            report["termination_counts"]["nan_state"] = 1
        elif invalid == "metadata":
            report["checkpoint_metadata"] = []
        raw = json.dumps(report)
        if invalid in ("list", "null"):
            raw = "[]" if invalid == "list" else "null"
        elif invalid in ("nan", "infinity", "overflow"):
            number = {"nan": "NaN", "infinity": "Infinity", "overflow": "1e999"}[
                invalid
            ]
            raw = raw[:-1] + ', "extra_metric": ' + number + "}"
        Path(command[8]).write_text(raw)

    monkeypatch.setattr(microduck, "run", run)
    with pytest.raises(ValueError):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    assert calls == [0]
    root = args.output / "seed-0" if seeds else args.output
    manifest = json.loads((root / "run.json").read_text())
    assert manifest["status"] == "failed" and manifest["failed_phase"] == "validate"
    assert (root / "evaluation.json").is_file()
    assert not (root / "acceptance.json").exists()
    if seeds:
        batch = json.loads((args.output / "run.json").read_text())
        assert batch["status"] == "failed" and batch["completed_seeds"] == []
        assert not (args.output / "summary.json").exists()
