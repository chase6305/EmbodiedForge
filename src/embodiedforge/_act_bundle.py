"""Portable ACT inference files; packaging and verification need no ML SDK."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from ._act_checkpoints import inference_fingerprint

MANIFEST = "act-bundle.json"


def write_bundle(checkpoint, dataset, source, output):
    """Copy only inference dependencies, verify the copy, then publish the directory."""
    checkpoint, output = Path(checkpoint), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    fingerprint = inference_fingerprint(checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        root = staging / "bundle"
        model = root / "pretrained_model"
        model.mkdir(parents=True)
        for item in fingerprint["files"]:
            shutil.copyfile(checkpoint / item["path"], model / item["path"])
        if inference_fingerprint(model) != fingerprint:
            raise ValueError("Checkpoint changed while exporting inference bundle")
        metadata = {
            "schema_version": 1,
            "format": "embodiedforge_act_inference",
            "dataset": dataset,
            "source": source,
            "checkpoint_fingerprint": fingerprint,
        }
        (root / MANIFEST).write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        root.rename(output)
    finally:
        shutil.rmtree(staging)
    return {"output": str(output), "checkpoint_fingerprint": fingerprint}


def read_bundle(root):
    """Verify local saved files without accessing source run or dataset paths."""
    root = Path(root).resolve()
    raw = (root / MANIFEST).read_bytes()
    report = json.loads(raw)
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("format") != "embodiedforge_act_inference"
        or not isinstance(report.get("dataset"), dict)
    ):
        raise ValueError("Unsupported ACT inference bundle")
    model = root / "pretrained_model"
    if inference_fingerprint(model) != report.get("checkpoint_fingerprint"):
        raise ValueError("ACT inference bundle fingerprint mismatch")
    return report, model, hashlib.sha256(raw).hexdigest()
