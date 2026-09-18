"""Use installed Go1 dependencies without preparing an external SDK checkout."""

import json
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("mjbatch")
pytest.importorskip("mujoco_menagerie")
pytest.importorskip("torch")

from embodiedforge import recipes


@pytest.mark.parametrize("backend", ["native", "light-loco"])
def test_train_resume_evaluate_and_live_without_checkout(
    tmp_path, monkeypatch, backend
):
    if backend == "light-loco":
        pytest.importorskip("light_loco_parkour")
    cache = tmp_path / "absent-sdk-cache"
    train, resume, evaluation = [
        tmp_path / name for name in ("train", "resume", "eval")
    ]
    common = [
        "--task",
        "go1-joystick",
        "--standalone",
        "--cache",
        str(cache),
        "--num-envs",
        "4",
        "--threads",
        "1",
        "--timeout",
        "60",
    ]
    recipes.main(
        [
            "train",
            *common,
            "--go1-learner",
            backend,
            "--updates",
            "2",
            "--horizon",
            "4",
            "--output",
            str(train),
        ]
    )
    assets = json.loads((train / "assets.json").read_text())
    # A viewer or resumed process need not inherit the original shell's cache.
    monkeypatch.delenv("MENAGERIE_CACHE_DIR", raising=False)
    recipes.main(
        [
            "train",
            *common,
            "--updates",
            "1",
            "--horizon",
            "4",
            "--resume-run",
            str(train),
            "--seed",
            "7",
            "--go1-learning-rate",
            "0.0005",
            "--output",
            str(resume),
        ]
    )
    recipes.main(
        [
            "evaluate",
            *common,
            "--run",
            str(resume),
            "--steps",
            "5",
            "--record-motion",
            "--output",
            str(evaluation),
        ]
    )
    assert not cache.exists(), "standalone must not prepare/read a checkout"
    for run in (train, resume, evaluation):
        data = json.loads((run / "run.json").read_text())
        assert data["status"] == "complete"
        assert data["source"] == recipes.GO1_STANDALONE_SOURCE
        assert data["command"][0] == sys.executable
        assert data["request"]["project"] is None
        assert data["result"]["runtime"]["launch_mode"] == "installed_packages"
        assert data["result"]["assets"] == assets
    data = json.loads((resume / "run.json").read_text())
    assert data["result"]["learner_backend"] == backend
    assert data["request"]["go1_learner"] == backend
    with pytest.raises(ValueError, match="cannot switch learner"):
        recipes.main(
            [
                "train",
                *common,
                "--resume-run",
                str(train),
                "--go1-learner",
                "native" if backend == "light-loco" else "light-loco",
                "--output",
                str(tmp_path / "wrong-backend"),
            ]
        )
    assert not (tmp_path / "wrong-backend").exists()
    report = json.loads((resume / "resume-report.json").read_text())
    origin = json.loads((resume / "input-run.json").read_text())
    assert origin == json.loads((train / "run.json").read_text())
    assert report["source_manifest"]["sha256"] == recipes.sha256(
        resume / "input-run.json"
    )
    assert data["result"]["resume_report"]["sha256"] == recipes.sha256(
        resume / "resume-report.json"
    )
    assert report["code"]["status"] == "unchanged"
    assert report["configuration"]["seed"] == {
        "before": 0,
        "after": 7,
        "status": "changed",
    }
    assert report["configuration"]["learning_rate_override"] == {
        "before": None,
        "after": 0.0005,
        "status": "changed",
    }
    assert all(v["status"] == "unchanged" for v in report["dependencies"].values())
    assert report["checkpoint_sha256"] == origin["result"]["checkpoint_sha256"]
    assert report["start_iteration"] == 2
    assert "optimizer_state" in report["continuation"]["restored"]
    assert not (train / "resume-report.json").exists()
    evaluation_data = json.loads((evaluation / "run.json").read_text())
    evaluation_impl = evaluation_data["result"]["implementation"]
    assert evaluation_impl["mode"] == "training_snapshot"
    assert (
        evaluation_data["input_implementation"]["files"]
        == data["implementation"]["files"]
    )
    assert (
        evaluation_impl["task"]["sha256"]
        == data["implementation"]["files"]["locomotion/go1.py"]
    )
    assert evaluation_impl["task"]["path"].startswith(
        str(evaluation / "input-implementation") + "/"
    )
    assert data["result"]["checkpoint_iteration"] == 2
    assert data["result"]["cumulative_updates"] == 3
    assert list(evaluation.glob("motion-*.npz"))
    runtime = json.loads((resume / "training-runtime.json").read_text())
    assert runtime["environment"]["module"] == "embodiedforge.locomotion.go1"
    module = "go1_light_loco" if backend == "light-loco" else "go1_ppo"
    assert runtime["learner"]["module"] == "embodiedforge.locomotion." + module
    assert "/implementation/embodiedforge/" in runtime["environment"]["file"]
    from embodiedforge.go1_live import LivePolicyProcess

    live = LivePolicyProcess(resume, num_envs=4, threads=1, episode_steps=6)
    try:
        assert live.metadata["assets"] == assets
        implementation = live.metadata["implementation"]
        assert implementation["mode"] == "training_snapshot"
        assert (
            implementation["task"]["sha256"]
            == data["implementation"]["files"]["locomotion/go1.py"]
        )
        assert (
            implementation["learner"]["sha256"]
            == data["implementation"]["files"]["locomotion/go1_ppo.py"]
        )
        assert implementation["task"]["path"].startswith(live.directory.name + "/")
        for env_id in range(4):
            live.request("velocity", env_id=env_id, value=[0.5, 0, 0])
        with np.load(
            next(evaluation.glob("motion-*.npz")), allow_pickle=False
        ) as motion:
            for frame in range(5):
                state = live.request("step")
                assert state["step"] == [frame + 1] * 4
                assert state["command"] == [[0.5, 0, 0]] * 4
                # Motion archives intentionally store poses as float32.
                np.testing.assert_array_equal(
                    np.asarray(state["qpos"][0][7:], dtype=np.float32),
                    motion["joints"][frame],
                )
        assert live.model_path.is_file()
    finally:
        live.close()

    # A broken training snapshot must not silently switch to current task code.
    (resume / "implementation/embodiedforge/locomotion/go1_scene.xml").write_text(
        "<mujoco/>"
    )
    with pytest.raises(ValueError, match="Implementation snapshot differs"):
        LivePolicyProcess(resume, num_envs=1, threads=1)
    failed = tmp_path / "eval-damaged"
    with pytest.raises(ValueError, match="Implementation snapshot differs"):
        recipes.main(
            [
                "evaluate",
                *common,
                "--run",
                str(resume),
                "--steps",
                "1",
                "--output",
                str(failed),
            ]
        )
    assert json.loads((failed / "run.json").read_text())["status"] == "failed"
    assert not (failed / "runtime.json").exists(), (
        "damaged source must fail before worker launch"
    )

    # The completed evaluation keeps its own valid copy after the input changes.
    recipes.validate_snapshot(
        evaluation / "input-implementation", evaluation_data["input_implementation"]
    )

    # Old runs that never recorded source are explicit about using current code.
    data.pop("implementation")
    recipes.write_json(resume / "run.json", data)
    legacy = tmp_path / "eval-legacy"
    recipes.main(
        [
            "evaluate",
            *common,
            "--run",
            str(resume),
            "--steps",
            "1",
            "--output",
            str(legacy),
        ]
    )
    legacy_data = json.loads((legacy / "run.json").read_text())
    assert legacy_data["result"]["implementation"]["mode"] == "current_code_legacy"

    data["result"]["runtime"]["versions"]["numpy"] = "0.0.0"
    recipes.write_json(resume / "run.json", data)
    mismatch = tmp_path / "eval-version-mismatch"
    with pytest.raises(subprocess.CalledProcessError):
        recipes.main(
            [
                "evaluate",
                *common,
                "--run",
                str(resume),
                "--steps",
                "1",
                "--output",
                str(mismatch),
            ]
        )
    assert (
        "Training dependency mismatch for numpy"
        in (mismatch / "console.log").read_text()
    )
