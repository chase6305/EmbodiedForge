"""Inspect ACT checkpoint files without importing Torch or LeRobot."""

import hashlib
import json
import struct
from pathlib import Path


def inference_fingerprint(model):
    """Identify the saved weights, config and processor files used for inference.

    Relative paths keep identity stable when moving a checkpoint. Optimizer and
    training state do not affect inference and are intentionally excluded.
    """
    model = Path(model)
    names = {"config.json", "model.safetensors"}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        names.add(name)
        pipeline = json.loads((model / name).read_text())
        for step in pipeline["steps"]:
            filename = step.get("state_file")
            if filename is not None:
                if (
                    not isinstance(filename, str)
                    or filename in ("", ".", "..")
                    or Path(filename).name != filename
                ):
                    raise ValueError(f"{name}: invalid state_file")
                names.add(filename)
    files = []
    for name in sorted(names):
        path = model / name
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Checkpoint changed while fingerprinting: {path}")
        files.append(
            {"path": name, "size": after.st_size, "sha256": digest.hexdigest()}
        )
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "scope": "saved_inference_files",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "files": files,
    }


def _tensor_layout(path):
    """Check safetensors headers/byte ranges, without loading tensors or their values."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= min(100_000_000, size - 8):
            raise ValueError("invalid safetensors header length")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError("invalid safetensors header")
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError("invalid tensor byte ranges")
        start, end = offsets
        if not 0 <= start <= end:
            raise ValueError("invalid tensor byte ranges")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("checkpoint contains no tensors")
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("overlapping or missing tensor bytes")
        cursor = end
    if cursor != size - 8 - header_size:
        raise ValueError("truncated or excess tensor payload")


def inspect_checkpoint(path):
    """Separate inference readiness from optimizer/RNG restoration readiness."""
    path = Path(path).resolve()
    model = path / "pretrained_model"
    state = path / "training_state"
    inference_errors = []
    resume_errors = []

    def check(file, errors, *, tensor=False, json_type=dict):
        try:
            if tensor:
                _tensor_layout(file)
                return None
            content = json.loads(file.read_text())
            if not isinstance(content, json_type):
                raise ValueError(f"expected JSON {json_type.__name__}")
            return content
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f"{file.relative_to(path)}: {exc}")
            return None

    config = check(model / "config.json", inference_errors)
    if config is not None and config.get("type") != "act":
        inference_errors.append("config.json: policy type must be act")
    check(model / "model.safetensors", inference_errors, tensor=True)
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        pipeline = check(model / name, inference_errors)
        if pipeline is None:
            continue
        steps = pipeline.get("steps")
        if not isinstance(steps, list) or not all(
            isinstance(step, dict) for step in steps
        ):
            inference_errors.append(f"{name}: invalid processor steps")
            continue
        for step in steps:
            filename = step.get("state_file")
            if filename is not None:
                if not isinstance(filename, str) or Path(filename).name != filename:
                    inference_errors.append(f"{name}: invalid state_file")
                else:
                    check(model / filename, inference_errors, tensor=True)
    saved = check(model / "train_config.json", resume_errors)
    if saved is not None:
        policy = saved.get("policy")
        if not isinstance(policy, dict) or policy.get("type") != "act":
            resume_errors.append("train_config.json: policy type must be act")
    metadata = check(state / "training_step.json", resume_errors)
    step = int(path.name) if path.name.isascii() and path.name.isdecimal() else None
    if metadata is not None:
        saved_step = metadata.get("step")
        if type(saved_step) is not int or saved_step < 0:
            resume_errors.append("training_step.json: invalid step")
        elif step is not None and step != saved_step:
            resume_errors.append("training_step.json: step differs from directory name")
        else:
            step = saved_step
    check(state / "optimizer_param_groups.json", resume_errors, json_type=list)
    for name in ("optimizer_state.safetensors", "rng_state.safetensors"):
        check(state / name, resume_errors, tensor=True)
    return {
        "path": str(path),
        "step": step,
        "inference_ready": not inference_errors,
        "resume_ready": not inference_errors and not resume_errors,
        "inference_errors": inference_errors,
        "resume_errors": resume_errors,
        "validation": "json_and_tensor_file_layout_not_tensor_values",
    }


def _checkpoint_paths(run):
    root = (Path(run) / "checkpoints").resolve()
    paths = []
    if root.is_dir():
        for path in root.iterdir():
            if (
                path.name.isascii()
                and path.name.isdecimal()
                and path.is_dir()
                and not path.is_symlink()
            ):
                paths.append(path)
    return sorted(paths, key=lambda path: (int(path.name), path.name), reverse=True)


def checkpoint_inventory(run):
    return [inspect_checkpoint(path) for path in _checkpoint_paths(run)]


def select_checkpoint(run, selector="last", *, for_resume=False):
    """Strict `last`/step selection, or explicit newest usable numeric checkpoint."""
    root = (Path(run) / "checkpoints").resolve()
    selector = str(selector)
    ready_key = "resume_ready" if for_resume else "inference_ready"
    skipped = []
    if selector == "latest":
        candidates = _checkpoint_paths(run)
    elif selector == "last":
        target = (root / "last").resolve()
        if not target.is_dir() or target.parent != root:
            raise FileNotFoundError(
                f"Missing or invalid last checkpoint: {root / 'last'}"
            )
        candidates = [target]
    elif selector.isascii() and selector.isdecimal():
        candidates = [
            path for path in _checkpoint_paths(run) if int(path.name) == int(selector)
        ]
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous checkpoint step {selector}")
    else:
        raise ValueError(
            "checkpoint must be last, latest, or a nonnegative step number"
        )
    for path in candidates:
        entry = inspect_checkpoint(path)
        if entry[ready_key]:
            return Path(entry["path"]), {
                "requested": selector,
                "step": entry["step"],
                "validation": entry["validation"],
                "skipped": skipped,
            }
        errors = entry["inference_errors"] + (
            entry["resume_errors"] if for_resume else []
        )
        skipped.append({"path": entry["path"], "step": entry["step"], "errors": errors})
    raise ValueError(
        f"No usable {ready_key} checkpoint for {selector}: {json.dumps(skipped)}"
    )
