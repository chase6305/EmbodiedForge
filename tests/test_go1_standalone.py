"""Use installed Go1 dependencies without preparing an external SDK checkout."""

import json
import sys

import pytest

pytest.importorskip("mjbatch")
pytest.importorskip("mujoco_menagerie")
pytest.importorskip("torch")

from embodiedforge import recipes


def test_train_resume_evaluate_and_live_without_checkout(tmp_path):
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
        live.request("velocity", env_id=0, value=[0.2, 0, 0])
        state = live.request("step")
        assert state["step"] == [1]
        assert state["command"] == [[0.2, 0, 0]]
        assert live.model_path.is_file()
    finally:
        live.close()
