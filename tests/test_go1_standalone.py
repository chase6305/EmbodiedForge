"""Use installed Go1 dependencies without preparing an external SDK checkout."""

import json
import sys

import pytest

pytest.importorskip("mjbatch")
pytest.importorskip("mujoco_menagerie")
pytest.importorskip("torch")

from embodiedforge import recipes


def test_train_resume_evaluate_and_live_without_checkout(tmp_path, monkeypatch):
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
        ["train", *common, "--updates", "2", "--horizon", "4", "--output", str(train)]
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
    assert data["result"]["checkpoint_iteration"] == 2
    assert data["result"]["cumulative_updates"] == 3
    assert list(evaluation.glob("motion-*.npz"))
    runtime = json.loads((resume / "training-runtime.json").read_text())
    assert runtime["environment"]["module"] == "embodiedforge.locomotion.go1"
    assert runtime["learner"]["module"] == "embodiedforge.locomotion.go1_ppo"
    assert "/implementation/embodiedforge/" in runtime["environment"]["file"]
    from embodiedforge.go1_live import LivePolicyProcess

    live = LivePolicyProcess(resume, num_envs=1, threads=1)
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
        live.request("velocity", env_id=0, value=[0.2, 0, 0])
        state = live.request("step")
        assert state["step"] == [1]
        assert state["command"] == [[0.2, 0, 0]]
        assert live.model_path.is_file()
    finally:
        live.close()

    # A broken training snapshot must not silently switch to current task code.
    (resume / "implementation/embodiedforge/locomotion/go1_scene.xml").write_text(
        "<mujoco/>"
    )
    with pytest.raises(ValueError, match="Implementation snapshot differs"):
        LivePolicyProcess(resume, num_envs=1, threads=1)
