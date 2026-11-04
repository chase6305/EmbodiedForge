"""Inspect SuperMimic prerequisites without downloading or executing its policies."""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .runtime.artifacts import write_json
from .runtime.environment import child_environment
from .runtime.hashing import file_identity as _file_identity
from .runtime.hashing import snapshot_file_hashes as _file_hashes
from .runtime.process import run_process
from .runtime.records import RunRejected, failure_record, managed_run, run_identity
from .workflows.supermimic_catalog import (
    CHECKED_ON,
    PACKAGES,
    PAPER,
    PAPER_CONTRACT,
    REFERENCES,
)

LOGGER = logging.getLogger(__name__)

PROBE = """
import importlib.metadata as metadata
import importlib.util
import json, os, sys
from pathlib import Path
result = {'python_version': list(sys.version_info[:3]), 'executable': sys.executable,
          'isolated': bool(sys.flags.isolated), 'distributions': {},
          'scope': 'distribution metadata and Torch CUDA availability; no policy inference'}
for name in json.loads(sys.argv[2]):
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        result['distributions'][name] = None
        continue
    entry = {'version': dist.version, 'vcs_commit': None, 'module_origin': None}
    direct = dist.read_text('direct_url.json')
    if direct:
        entry['vcs_commit'] = json.loads(direct).get('vcs_info', {}).get('commit_id')
    if name in ('mjlab', 'isaaclab'):
        spec = importlib.util.find_spec(name)
        if spec and spec.origin:
            entry['module_origin'] = str(Path(spec.origin).resolve())
    result['distributions'][name] = entry
try:
    import torch
    torch.set_num_threads(1)
    result['cuda_available'] = bool(torch.cuda.is_available())
except Exception as error:
    result['cuda_available'] = False
    result['torch_import_error'] = type(error).__name__ + ': ' + str(error)
path = Path(sys.argv[1])
temporary = path.with_suffix('.tmp')
with temporary.open('x') as stream:
    json.dump(result, stream, allow_nan=False)
os.replace(temporary, path)
"""


def catalog():
    return {
        "availability_checked_on": CHECKED_ON,
        "availability_is_live": False,
        "paper": PAPER,
        "paper_contract": PAPER_CONTRACT,
        "supported_operations": ["catalog", "preflight"],
        "policy_inference_implemented": False,
        "references": REFERENCES,
    }


def source_identity(root, expected, files=()):
    if root is None:
        raise ValueError("A local checkout path is required")
    root = root.expanduser().resolve(strict=True)
    # A caller's Git environment must not redirect inspection to another tree.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }

    def git(*arguments):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(root), *arguments],
            stderr=subprocess.PIPE,
            text=True,
            errors="surrogateescape",
            timeout=5,
            env=environment,
        ).strip()

    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Source path must be the checkout root")
    revision = git("rev-parse", "HEAD")
    if revision != expected:
        raise ValueError(f"Expected revision {expected}, found {revision}")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Tracked source changes prevent a reproducible preflight")
    # Git status trusts assume-unchanged and skip-worktree flags. Verify present
    # flagged files directly against committed blobs, including sparse checkouts.
    hidden = []
    for entry in git("ls-files", "-v", "-z").split("\0"):
        if entry and (entry[0].islower() or entry[0] == "S"):
            relative = entry[2:]
            if (
                entry[0].upper() != "S"
                or (root / relative).exists()
                or (root / relative).is_symlink()
            ):
                hidden.append(relative)

    paths = list(dict.fromkeys([*(os.fspath(path) for path in files), *hidden]))
    # Bound argument bytes as well as path count, and use literal pathspecs so
    # brackets, newlines and leading dashes retain their filename meaning.
    objects = {}
    batch = []
    batch_bytes = 0

    def load_objects():
        if not batch:
            return
        for entry in git("ls-tree", "-r", "-z", revision, "--", *batch).split("\0"):
            if entry:
                header, _, name = entry.partition("\t")
                mode, kind, object_id = header.split()
                if kind == "blob" and mode in ("100644", "100755"):
                    objects[name] = object_id

    for relative in paths:
        if (
            not Path(relative).parts
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError(f"Missing regular source file: {relative}")
        argument = ":(literal)" + relative
        size = len(os.fsencode(argument)) + 1
        if size > 65536:
            raise ValueError("Source path exceeds Git argument budget")
        if batch and (len(batch) >= 128 or batch_bytes + size > 65536):
            load_objects()
            batch.clear()
            batch_bytes = 0
        batch.append(argument)
        batch_bytes += size
    load_objects()
    git_format = {40: "sha1", 64: "sha256"}.get(len(revision))
    if git_format is None:
        raise ValueError("Unsupported Git object format")
    hashes = {}
    for relative in paths:
        path = root / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root)
        ):
            raise ValueError(f"Missing regular source file: {relative}")
        before = path.stat()
        actual, object_id, opened = _file_hashes(path, git_format)
        committed = objects.get(relative)
        after = path.stat()
        if _file_identity(before) != _file_identity(after) or _file_identity(
            opened
        ) != _file_identity(after):
            raise ValueError(f"Source changed during inspection: {relative}")
        if committed != object_id:
            raise ValueError(f"Source file differs from pinned commit: {relative}")
        hashes[relative] = actual
    if git("rev-parse", "HEAD") != revision or git(
        "status", "--porcelain", "--untracked-files=no"
    ):
        raise ValueError("Source changed during inspection")
    return {"root": str(root), "revision": revision, "files": hashes}


def inspect_motions(
    root, suffix, *, entry_limit=10000, match_limit=129, depth_limit=64
):
    """Bound traversal work as well as matches; never follow entry symlinks."""
    scanned = matched = 0
    depth_limited = False
    stack = []
    try:
        if root is not None and root.is_dir():
            stack.append(os.scandir(root))
        while stack and scanned < entry_limit and matched < match_limit:
            try:
                entry = next(stack[-1])
            except StopIteration:
                stack.pop().close()
                continue
            scanned += 1
            if entry.is_symlink():
                continue
            if entry.is_file(follow_symlinks=False) and entry.name.endswith(suffix):
                matched += 1
            elif entry.is_dir(follow_symlinks=False):
                if len(stack) < depth_limit:
                    stack.append(os.scandir(entry.path))
                else:
                    depth_limited = True
        return {
            "matched_files_at_least": matched,
            "entries_scanned": scanned,
            "entry_limit": entry_limit,
            "match_limit": match_limit,
            "depth_limit": depth_limit,
            "enumeration_capped": bool(stack) or depth_limited,
            "depth_limited": depth_limited,
            "content_verified": False,
        }
    finally:
        for iterator in reversed(stack):
            iterator.close()


def matches_version(value, constraint):
    if not isinstance(value, str):
        return False
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?(?:\+[A-Za-z0-9.]+)?", value)
    if match is None:
        return not constraint  # Unconstrained packages need only be installed.
    version = tuple(int(part or 0) for part in match.groups())
    return (
        "minimum" not in constraint or version >= tuple(constraint["minimum"])
    ) and ("maximum" not in constraint or version < tuple(constraint["maximum"]))


def validate_contract(contract, backend, checkpoint_sha256):
    """Validate declared metadata, without claiming that checkpoint tensors match."""
    if backend not in REFERENCES or backend == "official":
        raise ValueError("No supported exported checkpoint contract for this backend")
    reference = REFERENCES[backend]
    expected = {
        "schema": "ef-supermimic-contract-v1",
        "backend": backend,
        "source_revision": reference["revision"],
        "checkpoint_sha256": checkpoint_sha256,
        "robot": PAPER_CONTRACT["robot"],
        "control_hz": 50,
        "action_dim": 29,
        "history_frames": 10,
        "future_frames": 10,
        "observation_dim": reference["observation_dim"],
        "observation_layout": reference["observation_layout"],
        "goal_dim": reference["goal_dim"],
    }
    anchor = contract.get("anchor_obs", False)
    if not isinstance(anchor, bool) or (anchor and backend != "cuboi-reproduced"):
        raise ValueError("anchor_obs is a boolean option for cuboi-reproduced only")
    if anchor:
        expected["goal_dim"] = reference["anchor_goal_dim"]
    for key, value in expected.items():
        if type(contract.get(key)) is not type(value) or contract.get(key) != value:
            raise ValueError(f"Checkpoint contract mismatch: {key}")
    names = contract.get("joint_names")
    if (
        not isinstance(names, list)
        or len(names) != 29
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != 29
    ):
        raise ValueError(
            "joint_names must declare 29 distinct joint names in policy order"
        )
    return {
        "metadata_matches_profile": True,
        "tensors_verified": False,
        "anchor_obs": anchor,
    }


def preflight(args):
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = REFERENCES[args.backend]
    report = {
        "schema": "ef-supermimic-preflight-v1",
        "backend": args.backend,
        "provenance": reference["provenance"],
        "availability_checked_on": CHECKED_ON,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "prerequisites and declared metadata; no policy or motion deserialization",
        "evaluation_ready": False,
        "inference_executed": False,
        "supported_operations": ["catalog", "preflight"],
        "checks": [],
    }

    def check(name, passed, **details):
        report["checks"].append({"name": name, "passed": bool(passed), **details})

    with managed_run(output, run_identity("supermimic", "preflight", args.backend)):
        try:
            if args.backend == "official":
                check(
                    "authorized_official_policy_release",
                    False,
                    source=reference["license_url"],
                    reason="The reviewed browser demo does not grant local inference/evaluation rights",
                )
                check(
                    "official_export_contract",
                    False,
                    reason="No official local export ABI identified",
                )
            else:
                try:
                    report["source"] = source_identity(
                        args.root, reference["revision"], reference["source_files"]
                    )
                    check("pinned_source", True)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    check("pinned_source", False, reason=str(error))
                python = args.python.expanduser().absolute()
                probe_path = output / "sdk.json"
                run_process(
                    [
                        str(python),
                        "-I",
                        "-X",
                        "faulthandler",
                        "-c",
                        PROBE,
                        str(probe_path),
                        json.dumps(PACKAGES),
                    ],
                    cwd=output,
                    env=child_environment(python.parent.parent),
                    log_path=output / "sdk.log",
                    echo=False,
                    timeout=args.timeout,
                )
                sdk = json.loads(probe_path.read_text())
                report["sdk"] = sdk
                minor = reference["python_minor"]
                check(
                    "python_version",
                    minor is None or sdk["python_version"][:2] == minor,
                    actual=sdk["python_version"],
                    required_minor=minor,
                )
                check("isolated_sdk", sdk["isolated"])
                for name, constraint in reference["dependencies"].items():
                    dist = sdk["distributions"].get(name)
                    actual = dist["version"] if dist else None
                    check(
                        f"package:{name}",
                        matches_version(actual, constraint),
                        actual=actual,
                        required=constraint,
                    )
                check("cuda_available", sdk["cuda_available"])
                framework = reference["framework"]
                installed = sdk["distributions"].get(framework["distribution"]) or {}
                framework_ok = installed.get("vcs_commit") == framework["revision"]
                if args.framework_root is not None:
                    try:
                        identity = source_identity(
                            args.framework_root, framework["revision"]
                        )
                        origin = installed.get("module_origin")
                        framework_ok = bool(
                            origin and Path(origin).is_relative_to(identity["root"])
                        )
                        report["framework_source"] = identity
                    except (OSError, ValueError, subprocess.SubprocessError) as error:
                        framework_ok = False
                        report["framework_source_error"] = str(error)
                check(
                    "framework_revision",
                    framework_ok,
                    expected=framework["revision"],
                    installed_commit=installed.get("vcs_commit"),
                )
                checkpoint_hash = None
                if args.checkpoint is not None:
                    checkpoint = args.checkpoint.expanduser().resolve()
                    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
                        before = checkpoint.stat()
                        checkpoint_hash, _, opened = _file_hashes(checkpoint)
                        after = checkpoint.stat()
                        if _file_identity(before) != _file_identity(
                            after
                        ) or _file_identity(opened) != _file_identity(after):
                            raise ValueError("Checkpoint changed while hashing")
                        report["checkpoint"] = {
                            "path": str(checkpoint),
                            "sha256": checkpoint_hash,
                            "bytes": after.st_size,
                            "loaded": False,
                        }
                check("checkpoint_present", checkpoint_hash is not None)
                try:
                    if args.contract is None or checkpoint_hash is None:
                        raise ValueError(
                            "A checkpoint and its JSON contract are required"
                        )
                    contract = json.loads(args.contract.expanduser().read_text())
                    if not isinstance(contract, dict):
                        raise ValueError("Checkpoint contract must be a JSON object")
                    report["contract"] = validate_contract(
                        contract, args.backend, checkpoint_hash
                    )
                    check("declared_checkpoint_contract", True)
                except (OSError, ValueError) as error:
                    check("declared_checkpoint_contract", False, reason=str(error))
                motions = (
                    args.motions.expanduser() if args.motions is not None else None
                )
                motion_inventory = inspect_motions(motions, reference["motion_suffix"])
                check(
                    "motion_data_present",
                    motion_inventory["matched_files_at_least"] > 0,
                    **motion_inventory,
                )
            report["prerequisites_passed"] = all(
                item["passed"] for item in report["checks"]
            )
            report["status"] = (
                "complete" if report["prerequisites_passed"] else "rejected"
            )
            write_json(output / "result.json", report)
            if not report["prerequisites_passed"]:
                raise RunRejected(
                    "SuperMimic prerequisites are incomplete; see result.json"
                )
        except RunRejected:
            raise
        except BaseException as error:
            report.update(failure_record(error))
            try:
                write_json(output / "result.json", report)
            except BaseException:
                LOGGER.exception(
                    "Could not publish preflight failure; preserving original error"
                )
            raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "catalog", help="Show reviewed sources and compatibility boundaries"
    )
    probe = commands.add_parser(
        "preflight", help="Check prerequisites; does not execute policy inference"
    )
    probe.add_argument("--backend", choices=REFERENCES, default="official")
    probe.add_argument("--root", type=Path, help="Pinned reference checkout")
    probe.add_argument(
        "--framework-root", type=Path, help="Pinned editable mjlab/IsaacLab checkout"
    )
    probe.add_argument("--python", type=Path, default=Path(sys.executable))
    probe.add_argument("--checkpoint", type=Path)
    probe.add_argument(
        "--contract", type=Path, help="Declared checkpoint contract JSON"
    )
    probe.add_argument("--motions", type=Path)
    probe.add_argument("--timeout", type=float, default=30)
    probe.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "catalog":
        print(json.dumps(catalog(), indent=2))
        return
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    try:
        report = preflight(args)
    except RunRejected as error:
        parser.exit(2, f"{error}\n")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, f"SuperMimic preflight failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
