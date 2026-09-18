"""Validate actual TensorBoard event streams, including resumed runs."""

import pytest

from embodiedforge._microduck_worker import checkpoint_metadata, validate_metrics


@pytest.fixture
def events(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    def write(steps, *, loss=0.1, nan_state=0, extra_value=1.0, extra_scalars=None):
        writer = EventFileWriter(str(tmp_path))
        try:
            for step in steps:
                writer.add_event(
                    Event(
                        step=step,
                        summary=Summary(
                            value=[
                                Summary.Value(tag="Loss/value", simple_value=loss),
                                Summary.Value(
                                    tag="Episode_Termination/nan_state",
                                    simple_value=nan_state,
                                ),
                                Summary.Value(
                                    tag="Other/metric", simple_value=extra_value
                                ),
                            ]
                            + [
                                Summary.Value(tag=tag, simple_value=value)
                                for tag, value in (extra_scalars or {}).items()
                            ]
                        ),
                    )
                )
        finally:
            writer.close()
        return tmp_path / "model.pt"

    return write


def test_resumed_metrics_use_checkpoint_iteration(events):
    checkpoint = events([42, 43])
    report = validate_metrics(checkpoint, 2, start_iteration=42)
    assert report["iterations"] == 2
    assert report["start_iteration"] == 42
    assert report["all_scalars_finite"]
    with pytest.raises(RuntimeError, match="PPO updates"):
        validate_metrics(checkpoint, 2)


@pytest.mark.parametrize("steps", [[42], [42, 44], list(range(1000))])
def test_missing_or_unexpected_updates_rejected(events, steps):
    with pytest.raises(RuntimeError, match="PPO updates") as error:
        validate_metrics(events(steps), 2, start_iteration=42)
    message = str(error.value)
    assert "steps 42..43" in message
    assert f"missing {len({42, 43} - set(steps))}" in message
    assert f"unexpected {len(set(steps) - {42, 43})}" in message
    assert len(message) < 400


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"loss": float("nan")}, "Nonfinite training metric"),
        ({"extra_value": float("inf")}, "Nonfinite training metric"),
        ({"nan_state": 1}, "simulation reported NaN"),
    ],
)
def test_invalid_metrics_rejected(events, kwargs, match):
    with pytest.raises(RuntimeError, match=match):
        validate_metrics(events([0, 1], **kwargs), 2)


def test_checkpoint_learning_rate_and_provenance(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "model.pt"
    state = {
        "iter": 42,
        "infos": {"env_state": {"common_step_counter": 1032}},
        "actor_state_dict": {"weight": torch.ones(1)},
        "critic_state_dict": {"weight": torch.ones(1)},
        "optimizer_state_dict": {"param_groups": [{"lr": 3e-5}]},
    }
    torch.save(state, path)
    before = checkpoint_metadata(path)
    assert before["iteration"] == 42
    assert before["common_step_counter"] == 1032
    assert before["learning_rate"] == 3e-5
    state["optimizer_state_dict"]["param_groups"][0]["lr"] = 1e-5
    replacement = tmp_path / "replacement.pt"
    torch.save(state, replacement)
    load = torch.load

    def load_then_publish(*args, **kwargs):
        checkpoint = load(*args, **kwargs)
        replacement.replace(path)
        return checkpoint

    # Publishing a new checkpoint during inspection must not mix two versions.
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", load_then_publish)
        assert checkpoint_metadata(path) == before
    after = checkpoint_metadata(path)
    assert after["learning_rate"] == 1e-5
    assert after["sha256"] != before["sha256"]
    state["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
    torch.save(state, path)
    with pytest.raises(ValueError, match="positive PPO learning rate"):
        checkpoint_metadata(path)
    for payload, message in (
        ({"actor_state_dict": state["actor_state_dict"]}, "critic_state_dict"),
        ({**state, "infos": {}}, "curriculum counter"),
        (torch.ones(1), "full PPO training checkpoint dictionary"),
    ):
        torch.save(payload, path)
        with pytest.raises(ValueError, match=message):
            checkpoint_metadata(path)


def test_training_progress_resumed_offsets_and_eta(events):
    import json

    from embodiedforge._microduck_worker import training_progress

    checkpoint = events(
        [42, 43], extra_scalars={"Perf/collection_time": 0.8, "Perf/learning_time": 0.2}
    )
    root = checkpoint.parent
    log = root / "logs/rsl_rl/microduck/current"
    log.mkdir(parents=True)
    for path in root.glob("events.out.tfevents.*"):
        path.rename(log / path.name)
    (log / "model_42.pt").touch()
    manifest = {
        "workflow": "train",
        "status": "running",
        "resume": {"iteration": 42},
        "commands": [["python", "--agent.max-iterations", "5"]],
    }
    (root / "run.json").write_text(json.dumps(manifest))
    report = training_progress(root)
    assert report["start_iteration"] == 42
    assert report["observed_updates"] == 2
    assert report["latest_iteration"] == 43
    assert report["estimated_remaining_seconds"] == pytest.approx(3)
    assert report["latest_checkpoint"].endswith("model_42.pt")
    assert report["all_recorded_scalars_finite"]
    assert report["nan_states_seen"] is False
    manifest["status"] = "interrupted"
    (root / "run.json").write_text(json.dumps(manifest))
    assert training_progress(root)["estimated_remaining_seconds"] is None


def test_training_progress_before_first_event_is_unknown(events):
    import json

    from embodiedforge._microduck_worker import training_progress

    root = events([]).parent
    log = root / "logs/rsl_rl/microduck/current"
    log.mkdir(parents=True)
    for path in root.glob("events.out.tfevents.*"):
        path.rename(log / path.name)
    (root / "run.json").write_text(
        json.dumps({"workflow": "train", "status": "running", "commands": []})
    )
    report = training_progress(root)
    assert report["observed_updates"] == 0
    assert report["latest_iteration"] is None
    assert report["estimated_remaining_seconds"] is None
    assert report["nan_states_seen"] is None
    assert report["all_recorded_scalars_finite"] is None


def test_training_progress_reports_nonfinite_metrics_as_valid_json(events):
    import json

    from embodiedforge._microduck_worker import training_progress

    root = events([0], loss=float("nan"), nan_state=1).parent
    log = root / "logs/rsl_rl/microduck/current"
    log.mkdir(parents=True)
    for path in root.glob("events.out.tfevents.*"):
        path.rename(log / path.name)
    (root / "run.json").write_text(
        json.dumps(
            {
                "workflow": "train",
                "status": "failed",
                "commands": [["--agent.max-iterations", "10"]],
            }
        )
    )
    report = training_progress(root)
    assert not report["all_recorded_scalars_finite"]
    assert report["nan_states_seen"]
    assert report["latest_metrics"]["Loss/value"] is None
    json.dumps(report, allow_nan=False)
