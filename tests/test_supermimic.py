"""Preflight checks distinguish declared compatibility from actual inference."""

import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiedforge import supermimic as sm
from embodiedforge.runtime.records import RunRejected


def arguments(tmp_path, backend="cuboi-reproduced"):
    return SimpleNamespace(
        backend=backend,
        output=tmp_path / "output",
        root=tmp_path / "source",
        framework_root=None,
        python=Path(sys.executable),
        checkpoint=None,
        contract=None,
        motions=None,
        timeout=10,
    )


def contract(backend="cuboi-reproduced", digest="abc"):
    ref = sm.REFERENCES[backend]
    return {
        "schema": "ef-supermimic-contract-v1",
        "backend": backend,
        "source_revision": ref["revision"],
        "checkpoint_sha256": digest,
        "robot": "unitree-g1-29dof",
        "control_hz": 50,
        "action_dim": 29,
        "history_frames": 10,
        "future_frames": 10,
        "observation_dim": ref["observation_dim"],
        "observation_layout": ref["observation_layout"],
        "goal_dim": ref["goal_dim"],
        "joint_names": [f"joint_{i}" for i in range(29)],
    }


@pytest.fixture
def compatible_sdk(monkeypatch):
    def source(root, revision, files=()):
        return {"root": str(root), "revision": revision, "files": {}}

    def run(command, **kwargs):
        packages = {}
        for ref in sm.REFERENCES.values():
            for name, spec in ref.get("dependencies", {}).items():
                version = ".".join(map(str, spec.get("minimum", [1, 0, 0])))
                packages.setdefault(name, {"version": version, "vcs_commit": None})
        for ref in sm.REFERENCES.values():
            framework = ref.get("framework")
            if framework:
                packages[framework["distribution"]]["vcs_commit"] = framework[
                    "revision"
                ]
        sdk = {
            "python_version": [3, 12, 14],
            "isolated": True,
            "distributions": packages,
            "cuda_available": True,
        }
        Path(command[-2]).write_text(json.dumps(sdk))

    monkeypatch.setattr(sm, "source_identity", source)
    monkeypatch.setattr(sm, "run_process", run)


@pytest.mark.parametrize("backend", ["cuboi-reproduced", "vibed"])
def test_declared_contracts_are_separate_and_do_not_verify_tensors(backend):
    result = sm.validate_contract(contract(backend), backend, "abc")
    assert result == {
        "metadata_matches_profile": True,
        "tensors_verified": False,
        "anchor_obs": False,
    }
    other = "vibed" if backend == "cuboi-reproduced" else "cuboi-reproduced"
    with pytest.raises(ValueError):
        sm.validate_contract(contract(other), backend, "abc")


@pytest.mark.parametrize(
    "key,value",
    [
        ("checkpoint_sha256", "wrong"),
        ("observation_dim", 99),
        ("observation_layout", "time-major"),
        ("goal_dim", 730),
        ("action_dim", True),
        ("action_dim", 29.0),
        ("history_frames", 9),
        ("joint_names", ["same"] * 29),
        ("joint_names", [""] * 29),
        ("anchor_obs", "yes"),
    ],
)
def test_reject_incompatible_declared_contract(key, value):
    data = contract()
    data[key] = value
    with pytest.raises(ValueError):
        sm.validate_contract(data, "cuboi-reproduced", "abc")


def test_anchor_observations_require_explicit_goal_dimension():
    data = contract()
    data["anchor_obs"] = True
    with pytest.raises(ValueError):
        sm.validate_contract(data, "cuboi-reproduced", "abc")
    data["goal_dim"] = 689
    assert sm.validate_contract(data, "cuboi-reproduced", "abc")["anchor_obs"]
    with pytest.raises(ValueError):
        sm.validate_contract(contract(), "official", "abc")


def test_official_preflight_records_rejection_without_running_sdk(
    tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("Official preflight must not run an SDK or download a policy")

    monkeypatch.setattr(sm, "run_process", forbidden)
    args = arguments(tmp_path, "official")
    with pytest.raises(RunRejected):
        sm.preflight(args)
    result = json.loads((args.output / "result.json").read_text())
    assert result["status"] == "rejected"
    assert not result["evaluation_ready"] and not result["inference_executed"]
    assert json.loads((args.output / "run.json").read_text())["status"] == "rejected"


@pytest.mark.parametrize("backend", ["cuboi-reproduced", "vibed"])
def test_complete_metadata_does_not_claim_inference(tmp_path, compatible_sdk, backend):
    args = arguments(tmp_path, backend)
    args.checkpoint = tmp_path / "deliberately-not-a-model.pt"
    payload = b"No pickle, torch.load, or policy execution should occur."
    args.checkpoint.write_bytes(payload)
    args.contract = tmp_path / "contract.json"
    args.contract.write_text(
        json.dumps(contract(backend, hashlib.sha256(payload).hexdigest()))
    )
    args.motions = tmp_path / "motions"
    args.motions.mkdir()
    (args.motions / ("clip" + sm.REFERENCES[backend]["motion_suffix"])).write_bytes(
        b"presence only"
    )
    result = sm.preflight(args)
    assert result["prerequisites_passed"]
    assert not result["evaluation_ready"] and not result["inference_executed"]
    assert result["checkpoint"]["loaded"] is False
    assert result["contract"]["tensors_verified"] is False
    with pytest.raises(FileExistsError):
        sm.preflight(args)


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_empty_checkpoint_is_rejected(tmp_path, compatible_sdk, missing):
    args = arguments(tmp_path)
    args.checkpoint = tmp_path / "checkpoint.pt"
    if not missing:
        args.checkpoint.touch()
    with pytest.raises(RunRejected):
        sm.preflight(args)
    result = json.loads((args.output / "result.json").read_text())
    assert not next(
        c["passed"] for c in result["checks"] if c["name"] == "checkpoint_present"
    )


def test_checkpoint_rewrite_with_restored_mtime_is_not_a_stable_hash(
    tmp_path, compatible_sdk, monkeypatch
):
    args = arguments(tmp_path)
    args.checkpoint = tmp_path / "checkpoint.pt"
    payload = b"old-checkpoint"
    args.checkpoint.write_bytes(payload)
    args.contract = tmp_path / "contract.json"
    args.contract.write_text(
        json.dumps(contract(digest=hashlib.sha256(payload).hexdigest()))
    )
    args.motions = tmp_path / "motions"
    args.motions.mkdir()
    (args.motions / "clip.npz").write_bytes(b"presence only")
    # This hook runs after a real hash, then simulates an in-place replacement
    # whose size and mtime are preserved. The earlier implementation misses it.
    name = "_file_hashes" if hasattr(sm, "_file_hashes") else "file_sha256"
    original = getattr(sm, name)

    def rewrite(path, *extra):
        result = original(path, *extra)
        before = path.stat()
        # Linux may reuse the same ctime within one filesystem clock tick.
        time.sleep(0.02)
        path.write_bytes(b"new-checkpoint")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert path.stat().st_ctime_ns != before.st_ctime_ns
        return result

    monkeypatch.setattr(sm, name, rewrite)
    with pytest.raises(ValueError, match="Checkpoint changed while hashing"):
        sm.preflight(args)
    assert json.loads((args.output / "run.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "fault,check_name",
    [
        ("python", "python_version"),
        ("cuda", "cuda_available"),
        ("torch", "package:torch"),
        ("framework", "framework_revision"),
    ],
)
def test_sdk_incompatibilities_are_reported(
    tmp_path, monkeypatch, compatible_sdk, fault, check_name
):
    original = sm.run_process

    def probe(command, **kwargs):
        original(command, **kwargs)
        path = Path(command[-2])
        data = json.loads(path.read_text())
        if fault == "python":
            data["python_version"] = [3, 10, 0]
        elif fault == "cuda":
            data["cuda_available"] = False
        elif fault == "torch":
            data["distributions"]["torch"]["version"] = "2.9.0"
        else:
            data["distributions"]["mjlab"]["vcs_commit"] = "wrong"
        path.write_text(json.dumps(data))

    monkeypatch.setattr(sm, "run_process", probe)
    args = arguments(tmp_path)
    with pytest.raises(RunRejected):
        sm.preflight(args)
    data = json.loads((args.output / "result.json").read_text())
    assert not next(
        check["passed"] for check in data["checks"] if check["name"] == check_name
    )


def test_missing_source_is_a_recorded_prerequisite_rejection(
    tmp_path, monkeypatch, compatible_sdk
):
    def missing(*args, **kwargs):
        raise ValueError("Expected revision does not match")

    monkeypatch.setattr(sm, "source_identity", missing)
    args = arguments(tmp_path)
    with pytest.raises(RunRejected):
        sm.preflight(args)
    data = json.loads((args.output / "result.json").read_text())
    check = next(check for check in data["checks"] if check["name"] == "pinned_source")
    assert not check["passed"] and "Expected revision" in check["reason"]


@pytest.mark.parametrize(
    "error,status",
    [
        (subprocess.CalledProcessError(7, ["sdk"]), "failed"),
        (subprocess.TimeoutExpired(["sdk"], 1), "timed_out"),
        (KeyboardInterrupt(), "interrupted"),
    ],
)
@pytest.mark.parametrize("publication_fails", [False, True])
def test_probe_error_survives_result_publication(
    tmp_path, monkeypatch, error, status, publication_fails
):
    args = arguments(tmp_path)
    monkeypatch.setattr(sm, "source_identity", lambda *a: {})

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(sm, "run_process", fail)
    if publication_fails:
        monkeypatch.setattr(
            sm, "write_json", lambda *a: (_ for _ in ()).throw(OSError("disk full"))
        )
    with pytest.raises(type(error)) as caught:
        sm.preflight(args)
    assert caught.value is error
    assert json.loads((args.output / "run.json").read_text())["status"] == status
    if not publication_fails:
        assert json.loads((args.output / "result.json").read_text())["status"] == status


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_bad_timeout_does_not_claim_output(tmp_path, timeout):
    output = tmp_path / "new"
    with pytest.raises(SystemExit) as caught:
        sm.main(["preflight", "--timeout", timeout, "--output", str(output)])
    assert caught.value.code == 2
    assert not output.exists()


def test_dependency_minimum_does_not_accept_prerelease_or_old_version():
    spec = {"minimum": [2, 14, 0]}
    assert sm.matches_version("2.14.1+cu130", spec)
    assert not sm.matches_version("2.9.0", spec)
    assert not sm.matches_version("2.14.0rc1", spec)
    assert not sm.matches_version(None, {})
    assert sm.matches_version("3.0.0.dev1", {})


def test_source_revision_and_dirty_files_are_checked(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "README").write_text("source")
    subprocess.run(["git", "-C", str(root), "add", "README"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert sm.source_identity(root, revision, ["README"])["revision"] == revision
    with pytest.raises(ValueError, match="Expected revision"):
        sm.source_identity(root, "0" * 40)
    (root / "README").write_text("modified")
    with pytest.raises(ValueError, match="Tracked source changes"):
        sm.source_identity(root, revision)


def test_catalog_is_metadata_only():
    data = copy.deepcopy(sm.catalog())
    assert not data["availability_is_live"]
    assert not data["policy_inference_implemented"]
    assert data["supported_operations"] == ["catalog", "preflight"]
