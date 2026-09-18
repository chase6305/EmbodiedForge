"""Freeze evaluation models without importing a training or inference SDK."""

import shutil
from pathlib import Path

from ._microduck_run import checkpoint_digest


def snapshot_inputs(output, checkpoint, options, source_run=None):
    directory = output / "inputs"
    directory.mkdir(exist_ok=False)
    sources = {"checkpoint": (checkpoint, "checkpoint.pt")}
    if options.get("onnx"):
        sources["onnx"] = (Path(options["onnx"]), "policy.onnx")
    result = {}
    for role, (source, filename) in sources.items():
        expected = checkpoint_digest(source)
        if role == "checkpoint" and source_run is not None:
            if expected != source_run["checkpoint_sha256"]:
                raise ValueError("Selected training run checkpoint SHA256 mismatch")
        target = directory / filename
        with source.open("rb") as reader, target.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if checkpoint_digest(target) != expected:
            raise ValueError(f"Evaluation {role} changed while taking snapshot; retry")
        result[role] = {"source": str(source), "path": str(target), "sha256": expected}
    return result


def verify_input_snapshot(snapshot):
    for role, item in snapshot.items():
        if checkpoint_digest(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Evaluation {role} snapshot SHA256 mismatch")


def verify_report_inputs(report, snapshot):
    if (
        report.get("checkpoint_metadata", {}).get("sha256")
        != snapshot["checkpoint"]["sha256"]
    ):
        raise ValueError("Evaluation report checkpoint differs from input snapshot")
    parity = report.get("onnx_parity")
    if "onnx" in snapshot:
        if (
            not isinstance(parity, dict)
            or parity.get("sha256") != snapshot["onnx"]["sha256"]
        ):
            raise ValueError("Evaluation report ONNX differs from input snapshot")
    elif parity is not None:
        raise ValueError("Evaluation report contains an unrequested ONNX comparison")
