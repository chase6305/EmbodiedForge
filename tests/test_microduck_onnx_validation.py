"""ONNX comparison evidence must cover the evaluation's requested steps."""

import copy
import json
from pathlib import Path

import pytest
from microduck_report_fixture import evaluation_report

from embodiedforge._microduck_reports import (
    aggregate_evaluations,
    validate_evaluation_report,
)


@pytest.fixture
def evidence(tmp_path):
    checkpoint, onnx = tmp_path / "checkpoint.pt", tmp_path / "policy.onnx"
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"ONNX")
    options = {"onnx": str(onnx), "velocity": None, "no_pushes": False}
    command = [
        "python",
        "-I",
        "worker",
        "evaluate",
        str(checkpoint),
        "2",
        "4",
        "0",
        "evaluation.json",
        json.dumps(options),
    ]
    report = evaluation_report(command)
    manifest = {
        "task": report["task"],
        "commands": [command],
        "evaluation_options": options,
    }
    return report, manifest


@pytest.mark.parametrize(
    "field,value",
    [
        ("samples", None),
        ("samples", 0),
        ("samples", 3),
        ("samples", 5),
        ("samples", 4.0),
        ("samples", True),
        ("max_absolute_error", None),
        ("max_absolute_error", -1),
        ("max_absolute_error", float("nan")),
        ("max_absolute_error", float("inf")),
        ("max_absolute_error", False),
        ("atol", 1e-2),
        ("rtol", 1e-2),
        ("atol", True),
        ("rtol", -1),
        ("provider", "CUDAExecutionProvider"),
        ("sampling", "first environment only"),
    ],
)
def test_incomplete_or_changed_parity_contract_is_rejected(evidence, field, value):
    report, _ = evidence
    report["onnx_parity"][field] = value
    with pytest.raises(ValueError, match="ONNX"):
        aggregate_evaluations([report])


def test_wrong_path_or_precision_is_rejected(evidence):
    report, manifest = evidence
    other = copy.deepcopy(report)
    other["onnx_parity"]["onnx"] = "other.onnx"
    with pytest.raises(ValueError, match="ONNX path"):
        validate_evaluation_report(other, manifest)
    report["conditions"]["policy_matmul_precision"] = "tf32"
    with pytest.raises(ValueError, match="FP32"):
        validate_evaluation_report(report, manifest)


def test_relative_tolerance_does_not_impose_an_absolute_error_ceiling(evidence):
    import numpy as np

    from embodiedforge._microduck_worker import compare_policy_actions

    report, manifest = evidence
    reference = np.full((1, 14), 1000.0)
    error = compare_policy_actions(reference, reference + 0.01)
    assert error > report["onnx_parity"]["atol"]
    report["onnx_parity"]["max_absolute_error"] = error
    validate_evaluation_report(report, manifest)


def test_zero_comparisons_fail_online_before_marking_run_complete(
    evidence, tmp_path, monkeypatch
):
    from argparse import Namespace

    from embodiedforge import microduck

    _, manifest = evidence
    command = manifest["commands"][0]
    args = Namespace(
        checkpoint=Path(command[4]),
        onnx=Path(manifest["evaluation_options"]["onnx"]),
        output=tmp_path / "evaluation",
        num_envs=2,
        steps=4,
        seed=0,
        velocity=None,
        no_pushes=False,
    )

    def run(command, **kwargs):
        if command[3] == "evaluate":
            report = evaluation_report(command)
            report["onnx_parity"]["samples"] = 0
            Path(command[8]).write_text(json.dumps(report))

    monkeypatch.setattr(microduck, "run", run)
    with pytest.raises(ValueError, match="ONNX sample count"):
        microduck.evaluate_run(args, Path("/isolated/python"), {}, {})
    record = json.loads((args.output / "run.json").read_text())
    assert record["status"] == "failed" and record["failed_phase"] == "validate"
