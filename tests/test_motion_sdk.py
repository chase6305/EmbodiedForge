"""Motion SDK contracts that do not require the optional upstream environments."""

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge._gmr_worker import bvh_timing
from embodiedforge._sonic_worker import episode_result, validate_clip


@pytest.mark.parametrize("interrupt", [False, True], ids=["worker-error", "sigterm"])
def test_motion_launch_finalizes_partial_results(tmp_path, monkeypatch, interrupt):
    from embodiedforge import _motion_sdk as sdk
    from embodiedforge._microduck_process import ProcessInterrupted

    source = tmp_path / "embodiedforge"
    source.mkdir()
    (source / "__init__.py").write_text("")
    clips = [{"name": "finished", "status": "success", "joint_mae_rad": 0.1}]
    partial = {"status": "running", "clips": clips}
    (source / "_sonic_worker.py").write_text(
        "import os, signal, sys, time\n"
        "from pathlib import Path\n"
        f"Path('result.json').write_text({json.dumps(partial)!r})\n"
        + (
            "os.kill(os.getppid(), signal.SIGTERM)\ntime.sleep(30)\n"
            if interrupt
            else "sys.exit(3)\n"
        )
    )
    monkeypatch.setattr(sdk, "__file__", str(source / "_motion_sdk.py"))
    monkeypatch.setattr(sdk, "checkout", lambda *_: None)
    output = tmp_path / "run"
    expected = ProcessInterrupted if interrupt else subprocess.CalledProcessError
    with pytest.raises(expected) as error:
        sdk.launch(
            "sonic",
            SimpleNamespace(root=tmp_path, python=Path(sys.executable), output=output),
            "pinned-revision",
            {},
            {},
        )
    if interrupt:
        assert error.value.signum == signal.SIGTERM
    else:
        assert error.value.returncode == 3
    state = "interrupted" if interrupt else "failed"
    run = json.loads((output / "run.json").read_text())
    result = json.loads((output / "result.json").read_text())
    assert run["status"] == result["status"] == state
    assert result["clips"] == clips
    assert result["error"] == run["error"]
    assert "success_rate" not in result


def test_motion_launch_keeps_code_snapshot_and_rejects_tampering(tmp_path, monkeypatch):
    from embodiedforge import _motion_sdk as sdk

    source = tmp_path / "source" / "embodiedforge"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "helper.py").write_text("VALUE = 'original'\n")
    (source / "_gmr_worker.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "from .helper import VALUE\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "(Path(request['output']) / 'result.json').write_text(\n"
        "    json.dumps({'status': 'complete', 'value': VALUE}))\n"
    )
    monkeypatch.setattr(sdk, "__file__", str(source / "_motion_sdk.py"))
    monkeypatch.setattr(sdk, "checkout", lambda *_: None)
    run_process = sdk.run_process

    def edit_before_running(*args, **kwargs):
        (source / "helper.py").write_text("VALUE = 'edited'\n")
        return run_process(*args, **kwargs)

    monkeypatch.setattr(sdk, "run_process", edit_before_running)
    output = sdk.launch(
        "gmr",
        SimpleNamespace(
            root=tmp_path, python=Path(sys.executable), output=tmp_path / "run"
        ),
        "pinned-revision",
        {},
        {},
    )
    assert json.loads((output / "result.json").read_text())["value"] == "original"
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    package = output / "implementation" / "embodiedforge"
    assert not package.is_symlink()
    request_path = output / "request.json"
    with pytest.raises(ValueError, match="must load its implementation snapshot"):
        sdk.load_request(request_path)
    monkeypatch.setattr(sdk, "__file__", str(package / "_motion_sdk.py"))
    assert sdk.load_request(request_path)["implementation"]["files"]["helper.py"]
    (package / "helper.py").write_text("VALUE = 'tampered'\n")
    with pytest.raises(ValueError, match="Motion SDK implementation differs"):
        sdk.load_request(request_path)


def test_bvh_keeps_declared_timing_and_rejects_missing_interval(tmp_path):
    source = tmp_path / "motion.bvh"
    source.write_text("HIERARCHY\nMOTION\nFrames: 4249\nFrame Time: 0.008333\n")
    assert bvh_timing(source) == (4249, 0.008333)
    source.write_text("MOTION\nFrames: 3\nFrame Time: nan\n")
    with pytest.raises(ValueError, match="Frame Time"):
        bvh_timing(source)


def test_sonic_terminal_report_distinguishes_fall_completion_and_cap():
    def line(reason):
        return (
            f"[end] ep=0 ran 9.22s, reason: {reason} | "
            "joint MAE 0.1234 rad (max 0.532) | pelvis-z MAE 0.042 m"
        )

    assert episode_result(line("motion_end"))["status"] == "success"
    assert episode_result(line("pelvis_z=0.300 < 0.40"))["status"] == "failed"
    assert episode_result(line("reached --max-episode=9.2s"))["status"] == "truncated"
    for text in (
        "",
        line("motion_end") * 2,
        line("motion_end").replace("0.1234", "nan"),
        line("motion_end") + "\n" + line("motion_end").replace("0.1234", "nan"),
        line("motion_end").replace("ep=0", "ep=1"),
        line("motion_end") + " trailing garbage",
        line("motion_end").replace("0.1234", "9" * 400),
    ):
        with pytest.raises(ValueError, match="terminal episode"):
            episode_result(text)


def test_sonic_checks_frame_extent_and_quaternion_before_launching_player():
    clip = {
        "fps": 30.0,
        "dof": np.zeros((4, 31)),
        "root_trans_offset": np.zeros((4, 3)),
        "root_rot": np.tile([0.0, 0.0, 0.0, 1.0], (4, 1)),
    }
    assert validate_clip("idle", clip, 2) == (4, 30.0)
    with pytest.raises(ValueError, match="two frames"):
        validate_clip("idle", clip, 3)
    clip["root_rot"][1] = 0
    with pytest.raises(ValueError, match="Nonunit"):
        validate_clip("idle", clip, 0)
