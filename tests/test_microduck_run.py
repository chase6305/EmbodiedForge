"""Training-directory inputs choose recorded checkpoints, never guessed latest files."""

import json
import shutil
from argparse import Namespace

import pytest
from microduck_report_fixture import evaluation_report

from embodiedforge import microduck
from embodiedforge._microduck_run import checkpoint_digest, resolve_run_checkpoint

IDENTITY = {"revision": microduck.REVISION, "uv_lock_sha256": "lock-hash"}


@pytest.fixture
def training(tmp_path):
    root = tmp_path / "training with spaces"
    model = root / "logs/rsl_rl/microduck/session/model_10.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"selected weights")
    (model.parent / "model_999.pt").write_bytes(b"unrecorded newer file")
    report = {
        "schema": 1,
        "workflow": "train",
        "task": microduck.TASK,
        "status": "complete",
        "source": IDENTITY,
        "checkpoint": str(model),
        "checkpoint_relative": model.relative_to(root).as_posix(),
        "checkpoint_sha256": checkpoint_digest(model),
    }
    (root / "run.json").write_text(json.dumps(report))
    return root, model


def test_recorded_checkpoint_is_selected_and_can_move(training, tmp_path):
    root, model = training
    selected, info = resolve_run_checkpoint(root, IDENTITY, microduck.TASK)
    assert selected == model and info["verification"] == "verified"
    copied = tmp_path / "copied run"
    shutil.copytree(root, copied)
    model.write_bytes(b"changed original must not be used")
    selected, copied_info = resolve_run_checkpoint(copied, IDENTITY, microduck.TASK)
    assert selected == copied / info["checkpoint_relative"]
    assert copied_info["checkpoint_sha256"] == info["checkpoint_sha256"]


def test_legacy_run_can_move_but_does_not_claim_historical_hash(training, tmp_path):
    root, _ = training
    path = root / "run.json"
    data = json.loads(path.read_text())
    del data["checkpoint_sha256"], data["checkpoint_relative"]
    path.write_text(json.dumps(data))
    moved = tmp_path / "moved"
    root.rename(moved)
    selected, info = resolve_run_checkpoint(moved, IDENTITY, microduck.TASK)
    assert selected.is_relative_to(moved)
    assert info["verification"] == "unverified_legacy"


@pytest.fixture
def interrupted(training):
    root, model = training
    path = root / "run.json"
    data = json.loads(path.read_text())
    data.update(
        status="interrupted",
        finished_at="2026-09-17T00:00:00Z",
        checkpoints=[str(model)],
    )
    for name in ("checkpoint", "checkpoint_relative", "checkpoint_sha256"):
        data.pop(name)
    path.write_text(json.dumps(data))
    return root, model


@pytest.mark.parametrize("status", ["interrupted", "failed"])
def test_resume_selects_latest_recorded_model_and_supports_relocation(
    interrupted, tmp_path, status
):
    root, model = interrupted
    path = root / "run.json"
    data = json.loads(path.read_text())
    earlier = model.with_name("model_2.pt")
    earlier.write_bytes(b"older checkpoint")
    data.update(status=status, checkpoints=[str(model), str(earlier)])
    path.write_text(json.dumps(data))
    copied = tmp_path / "relocated"
    shutil.copytree(root, copied)
    selected, info = resolve_run_checkpoint(
        copied, IDENTITY, microduck.TASK, allow_incomplete=True
    )
    assert selected == copied / model.relative_to(root)
    assert selected.name == "model_10.pt"  # Unrecorded model_999.pt is ignored.
    assert info["verification"] == "recorded_at_recovery"
    assert info["checkpoint_sha256"] == checkpoint_digest(selected)
    assert info["selection"] == "latest_recorded_recovery"
    assert info["run_status"] == status
    with pytest.raises(ValueError, match="completed"):
        resolve_run_checkpoint(copied, IDENTITY, microduck.TASK)


@pytest.mark.parametrize(
    "invalid",
    [
        "running",
        "unfinished",
        "live",
        "missing",
        "empty",
        "temporary",
        "ambiguous",
        "duplicate",
        "escape",
        "hash",
    ],
)
def test_unsafe_automatic_recovery_is_rejected(interrupted, tmp_path, invalid):
    from embodiedforge._microduck_worker import launcher_identity

    root, model = interrupted
    path = root / "run.json"
    data = json.loads(path.read_text())
    if invalid == "running":
        data["status"] = "running"
    elif invalid == "unfinished":
        del data["finished_at"]
    elif invalid == "live":
        data["launcher"] = launcher_identity()
    elif invalid == "missing":
        model.unlink()
    elif invalid == "empty":
        data["checkpoints"] = []
    elif invalid == "temporary":
        data["checkpoints"] = [str(model) + ".tmp"]
    elif invalid == "ambiguous":
        data["checkpoints"].append(str(model.parent.parent / "other/model_2.pt"))
    elif invalid == "duplicate":
        data["checkpoints"].append(str(model.with_name("model_010.pt")))
    elif invalid == "escape":
        other = tmp_path / "outside.pt"
        model.rename(other)
        model.symlink_to(other)
    elif invalid == "hash":
        data.update(checkpoint=str(model), checkpoint_sha256="different")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        resolve_run_checkpoint(root, IDENTITY, microduck.TASK, allow_incomplete=True)


def test_train_cli_accepts_recorded_interrupted_run(interrupted, tmp_path, monkeypatch):
    root, model = interrupted
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    (environment / "embodiedforge-source.json").write_text(json.dumps(IDENTITY))
    monkeypatch.setattr(microduck, "source_identity", lambda repo: IDENTITY)
    calls = []
    monkeypatch.setattr(microduck, "train_run", lambda args, *rest: calls.append(args))
    microduck.main(
        [
            "train",
            "--repo",
            str(tmp_path),
            "--env-dir",
            str(environment),
            "--resume-run",
            str(root),
            "--iterations",
            "2",
            "--output",
            str(tmp_path / "new"),
        ]
    )
    assert calls[0].resume == model
    assert calls[0].source_run["verification"] == "recorded_at_recovery"


@pytest.mark.parametrize(
    "case, message",
    [
        ("running", "completed"),
        ("failed", "completed"),
        ("task", "completed"),
        ("workflow", "completed"),
        ("source", "source differs"),
        ("hash", "SHA256 mismatch"),
        ("missing", "missing"),
        ("escape", "outside the run"),
        ("path", "paths disagree"),
    ],
)
def test_invalid_run_is_rejected(training, tmp_path, case, message):
    root, model = training
    path = root / "run.json"
    data = json.loads(path.read_text())
    if case in ("running", "failed"):
        data["status"] = case
    elif case == "task":
        data["task"] = "unrelated"
    elif case == "workflow":
        data["workflow"] = "evaluate"
    elif case == "source":
        data["source"]["uv_lock_sha256"] = "other"
    elif case == "hash":
        model.write_bytes(b"replaced")
    elif case == "missing":
        model.unlink()  # model_999 still exists; it must not be selected instead.
    elif case == "escape":
        outside = tmp_path / "external.pt"
        model.rename(outside)
        model.symlink_to(outside)
    elif case == "path":
        data["checkpoint_relative"] = "../external.pt"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        resolve_run_checkpoint(root, IDENTITY, microduck.TASK)


@pytest.mark.parametrize("command", ["evaluate", "export", "play", "train"])
def test_cli_forwards_selected_model(training, tmp_path, monkeypatch, command):
    root, model = training
    environment = tmp_path / "env"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    (environment / "embodiedforge-source.json").write_text(json.dumps(IDENTITY))
    monkeypatch.setattr(microduck, "source_identity", lambda repo: IDENTITY)
    calls = []
    monkeypatch.setattr(
        microduck, "run", lambda command, **kwargs: calls.append(command)
    )
    monkeypatch.setattr(
        microduck, "evaluate_run", lambda args, *rest: calls.append(args)
    )
    monkeypatch.setattr(microduck, "train_run", lambda args, *rest: calls.append(args))
    monkeypatch.setattr(microduck, "export_run", lambda args, *rest: calls.append(args))
    argv = [command, "--repo", str(tmp_path), "--env-dir", str(environment)]
    argv += ["--resume-run" if command == "train" else "--run", str(root)]
    if command != "play":
        argv += ["--output", str(tmp_path / "output")]
    if command == "train":
        argv += ["--iterations", "2"]
    microduck.main(argv)
    if command in ("evaluate", "train", "export"):
        args = calls[0]
        assert (args.resume if command == "train" else args.checkpoint) == model
        assert args.source_run["verification"] == "verified"
    else:
        assert str(model) in calls[0]


@pytest.mark.parametrize(
    "command, switches",
    [
        ("play", ["--run", "run", "--checkpoint", "model.pt"]),
        (
            "train",
            [
                "--resume-run",
                "run",
                "--resume",
                "model.pt",
                "--iterations",
                "2",
                "--output",
                "out",
            ],
        ),
    ],
)
def test_source_arguments_are_mutually_exclusive(command, switches):
    with pytest.raises(SystemExit) as exc:
        microduck.main([command, "--repo", "repo", *switches])
    assert exc.value.code == 2


@pytest.mark.parametrize("changed", [False, True])
def test_evaluation_records_run_source_and_rejects_changed_loaded_model(
    training, tmp_path, monkeypatch, changed
):
    root, model = training
    _, source = resolve_run_checkpoint(root, IDENTITY, microduck.TASK)
    output = tmp_path / "evaluation"
    args = Namespace(
        checkpoint=model,
        source_run=source,
        steps=10,
        num_envs=2,
        seed=0,
        output=output,
        velocity=None,
        no_pushes=False,
        onnx=None,
    )

    def execute(command, **kwargs):
        if "evaluate" in command:
            report = evaluation_report(command)
            if changed:
                report["checkpoint_metadata"]["sha256"] = "changed"
            (output / "evaluation.json").write_text(json.dumps(report))

    monkeypatch.setattr(microduck, "run", execute)
    if changed:
        with pytest.raises(ValueError, match="checkpoint differs from input snapshot"):
            microduck.evaluate_once(args, model, IDENTITY, {})
    else:
        microduck.evaluate_once(args, model, IDENTITY, {})
    record = json.loads((output / "run.json").read_text())
    assert record["source_run"] == source
    assert record["status"] == ("failed" if changed else "complete")
