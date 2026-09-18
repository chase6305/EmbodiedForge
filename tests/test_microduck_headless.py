"""Training must not inherit a desktop dependency from the invoking shell."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from embodiedforge import microduck


def test_training_child_is_offscreen_and_preserves_gpu_selection():
    parent = {
        **microduck.child_environment(),
        "DISPLAY": ":99",
        "WAYLAND_DISPLAY": "wayland-1",
        "MUJOCO_GL": "glfw",
        "PYOPENGL_PLATFORM": "glx",
        "CUDA_VISIBLE_DEVICES": "2",
    }
    child = microduck.training_environment(parent)
    assert "DISPLAY" not in child and "WAYLAND_DISPLAY" not in child
    assert child["MUJOCO_GL"] == child["PYOPENGL_PLATFORM"] == "egl"
    assert child["CUDA_VISIBLE_DEVICES"] == "2"
    assert parent["DISPLAY"] == ":99" and parent["MUJOCO_GL"] == "glfw"


@pytest.mark.parametrize("workflow", ["train", "evaluate", "export"])
@pytest.mark.parametrize("flag", [[], ["--headless"]])
def test_headless_cli_defaults_and_explicit_flag(tmp_path, monkeypatch, workflow, flag):
    from embodiedforge import _microduck_native as native

    environment = tmp_path / "environment"
    (environment / "bin").mkdir(parents=True)
    (environment / "bin/python").touch()
    (environment / "embodiedforge-source.json").write_text(
        json.dumps(native.dependency_identity())
    )
    calls = []
    monkeypatch.setattr(microduck, "train_run", lambda args, *rest: calls.append(args))
    monkeypatch.setattr(
        microduck, "evaluate_run", lambda args, *rest: calls.append(args)
    )
    monkeypatch.setattr(microduck, "export_run", lambda args, *rest: calls.append(args))
    args = [
        workflow,
        "--env-dir",
        str(environment),
        "--output",
        str(tmp_path / "run"),
        *flag,
    ]
    if workflow == "train":
        args += ["--iterations", "2"]
    if workflow in ("evaluate", "export"):
        checkpoint = tmp_path / "checkpoint.pt"
        checkpoint.touch()
        args += ["--checkpoint", str(checkpoint)]
    microduck.main(args)
    assert len(calls) == 1 and calls[0].headless is True


@pytest.mark.parametrize(
    "native,video,seeds",
    [
        (True, False, None),
        (True, True, [0, 1]),
        (False, True, None),
        (False, False, [0, 1]),
    ],
)
def test_evaluation_selects_egl_before_check_and_task_imports(
    tmp_path, monkeypatch, native, video, seeds
):
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    output = tmp_path / "evaluation"
    args = Namespace(
        checkpoint=checkpoint,
        output=output,
        num_envs=2,
        steps=2,
        seed=0,
        seeds=seeds,
        velocity=None,
        no_pushes=False,
        onnx=None,
        video=video,
    )
    parent = {
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": "wayland-0",
        "MUJOCO_GL": "glfw",
        "PYOPENGL_PLATFORM": "glx",
        "CUDA_VISIBLE_DEVICES": "2",
        "EF_MICRODUCK_NATIVE": "1" if native else "0",
    }
    calls = []

    def execute(command, *, cwd, env, **kwargs):
        assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env
        assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        calls.append(command)
        if command[3] == "evaluate":
            assert bool(json.loads(command[-1]).get("video")) is video
            raise RuntimeError("stop before simulation")

    monkeypatch.setattr(microduck, "run", execute)
    with pytest.raises(RuntimeError, match="stop before simulation"):
        microduck.evaluate_run(args, Path(sys.executable), {}, parent)
    assert [command[3] for command in calls] == ["check", "evaluate"]
    assert parent["MUJOCO_GL"] == "glfw" and parent["DISPLAY"] == ":0"
    expected = {"headless": True, "viewer": None, "video": video, "mujoco_gl": "egl"}
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["execution"] == expected
    if seeds:
        manifest = json.loads((output / "seed-0/run.json").read_text())
        assert manifest["execution"] == expected
    assert manifest["failed_phase"] == "evaluate"


@pytest.mark.parametrize("native", [False, True])
def test_training_disables_video_and_records_headless_before_start(
    tmp_path, monkeypatch, native
):
    output = tmp_path / "run"
    command = microduck.training_command(
        Path(sys.executable), num_envs=64, iterations=5, native=native
    )
    assert command[command.index("--video") + 1] == "False"

    def fail(command, *, cwd, env, log_path=None, echo=True):
        assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env
        assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
        raise RuntimeError("stop before CUDA check")

    monkeypatch.setattr(microduck, "run", fail)
    with pytest.raises(RuntimeError, match="stop before CUDA"):
        microduck.train_run(
            Namespace(command="train", num_envs=64, iterations=5, output=output),
            Path(sys.executable),
            {},
            {
                "DISPLAY": ":0",
                "WAYLAND_DISPLAY": "wayland-0",
                "EF_MICRODUCK_NATIVE": "1" if native else "0",
            },
        )
    report = json.loads((output / "run.json").read_text())
    assert report["execution"] == {
        "headless": True,
        "viewer": None,
        "video": False,
        "mujoco_gl": "egl",
    }
    assert report["status"] == "failed"
    assert report["failed_phase"] == "check"
    assert report["active_log"] == "00-check.log"


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_training_seed_rejected_before_output_creation(tmp_path, seed):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="--seed"):
        microduck.train_run(
            Namespace(
                command="train", output=output, num_envs=64, iterations=5, seed=seed
            ),
            Path(sys.executable),
            {},
            {},
        )
    assert not output.exists()


def test_explicit_training_seed_is_forwarded():
    command = microduck.training_command(
        Path(sys.executable), num_envs=64, iterations=5, seed=42, native=True
    )
    assert command[command.index("--agent.seed") + 1] == "42"
