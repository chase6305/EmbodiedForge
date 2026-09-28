"""Reference-motion interoperability and first-episode scoring, without SDKs."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from embodiedforge._motion_clips import MotionClips, import_weave
from embodiedforge.motion import compare_clips, main
from embodiedforge.weave import prepare_inputs


@pytest.fixture
def motion(tmp_path):
    layout = {
        "robot": "fixture",
        "joint_names": ["hip", "knee"],
        "body_names": ["pelvis", "foot"],
        "quaternion_order": "wxyz",
        "coordinate_frame": "world",
    }
    arrays = {
        "fps": np.array(50.0),
        "motion_lengths": np.array([2, 3]),
        "motion_names": np.array(["lift", "carry"]),
        "object_names": np.array(["box", "box"]),
        "joint_pos": np.zeros((5, 2)),
        "joint_vel": np.zeros((5, 2)),
        "body_pos_w": np.zeros((5, 2, 3)),
        "body_quat_w": np.tile([1.0, 0, 0, 0], (5, 2, 1)),
        "body_lin_vel_w": np.zeros((5, 2, 3)),
        "body_ang_vel_w": np.zeros((5, 2, 3)),
        "object_pos_w": np.zeros((5, 3)),
        "object_quat_w": np.tile([1.0, 0, 0, 0], (5, 1)),
        "object_lin_vel_w": np.zeros((5, 3)),
        "object_ang_vel_w": np.zeros((5, 3)),
        "contact_label": np.zeros((5, 2)),
    }
    source = tmp_path / "weave.npz"
    np.savez(source, **arrays)
    return source, layout, arrays


def test_import_sampling_and_windows_keep_clip_boundaries(motion, tmp_path):
    source, layout, arrays = motion
    arrays["joint_pos"][:, 0] = np.arange(5)
    arrays["body_quat_w"] = np.roll(arrays["body_quat_w"], -1, axis=-1)
    arrays["object_quat_w"] = np.roll(arrays["object_quat_w"], -1, axis=-1)
    np.savez(source, **arrays)
    library = import_weave(source, {**layout, "quaternion_order": "xyzw"})
    target = tmp_path / "library.npz"
    library.save(target)
    loaded = MotionClips.load(target)
    frames = loaded.frames([0, 1], [1, 0], [0, 1, 999])
    assert frames["joint_pos"][..., 0].tolist() == [[1, 1, 1], [2, 3, 4]]
    assert np.all(frames["body_quat_w"][..., 0] == 1)
    a = loaded.sample(["box"] * 20, np.random.default_rng(3))
    b = loaded.sample(["box"] * 20, np.random.default_rng(3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert (a[1] < loaded.lengths[a[0]]).all()
    mixed = MotionClips(
        {**loaded.arrays, "object_names": np.array(["box", "chair"])},
        loaded.metadata,
    )
    ids, steps = mixed.sample(
        iter(["chair", "box", "chair", "box"]), np.random.default_rng(3)
    )
    assert ids.tolist() == [1, 0, 1, 0]
    assert ((steps >= 0) & (steps < mixed.lengths[ids])).all()
    empty_ids, empty_steps = mixed.sample([], np.random.default_rng(3))
    assert empty_ids.shape == empty_steps.shape == (0,)
    with pytest.raises(ValueError, match="Unknown object"):
        loaded.sample(["chair"], np.random.default_rng(3))
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        loaded.save(target)
    assert target.read_bytes() == before


def test_invalid_motion_data_is_rejected(motion):
    source, layout, original = motion
    cases = {
        "fps": np.array(float("nan")),
        "motion_lengths": np.array([2, 4]),
        "motion_names": np.array(["duplicate", "duplicate"]),
        "joint_pos": np.full((5, 2), np.inf),
        "body_quat_w": np.zeros((5, 2, 4)),
        "contact_label": np.full((5, 2), 2),
        "object_names": np.array(["box", "box"], dtype=object),
    }
    for key, value in cases.items():
        np.savez(source, **{**original, key: value})
        with pytest.raises(ValueError):
            import_weave(source, layout)


def test_failed_snapshot_cleanup_preserves_error_and_interruption(
    motion, tmp_path, monkeypatch, caplog
):
    source, layout, _ = motion
    library = import_weave(source, layout)

    def denied(*args, **kwargs):
        raise PermissionError("snapshot cleanup denied")

    monkeypatch.setattr(Path, "unlink", denied)
    for error in (RuntimeError("snapshot write failed"), KeyboardInterrupt()):
        destination = tmp_path / f"{type(error).__name__}.npz"

        def interrupted_save(stream, failure=error, **arrays):
            stream.write(b"partial snapshot")
            raise failure

        monkeypatch.setattr(np, "savez_compressed", interrupted_save)
        with pytest.raises(type(error)) as caught:
            library.save(destination)
        assert caught.value is error
        assert destination.read_bytes() == b"partial snapshot"
        assert str(destination) in caplog.text and "cleanup denied" in caplog.text


def test_weave_training_snapshots_and_control_rate(motion, tmp_path):
    source, layout, arrays = motion
    destination = tmp_path / "inputs"
    destination.mkdir()
    arrays["joint_pos"] = arrays["joint_pos"].astype(">f8")
    arrays["joint_pos"][0, 0] = 0.125
    np.savez(source, **arrays)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="--layout"):
        prepare_inputs([source], None, destination, "evaluate")
    result = prepare_inputs([source], layout, destination, "evaluate")
    snapshot = MotionClips.load(result["files"][0]["path"])
    assert snapshot.sha256 == result["files"][0]["sha256"]
    assert result["names"] == ["lift", "carry"]
    assert result["files"][0]["source_sha256"] == source_hash
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert snapshot.arrays["joint_pos"][0, 0] == 0.125
    assert all(
        snapshot.arrays[key].dtype == np.dtype("float32")
        for key in snapshot.frame_fields
    )
    with np.load(source, allow_pickle=False) as original:
        assert original["joint_pos"].dtype == np.dtype(">f8")
    arrays["fps"] = np.array(60.0)
    np.savez(source, **arrays)
    with pytest.raises(ValueError, match="50 Hz"):
        prepare_inputs([source], layout, destination, "train")


def test_weave_rejects_values_that_overflow_sdk_precision(motion, tmp_path):
    source, layout, arrays = motion
    arrays["joint_vel"][0, 0] = 1e100
    np.savez(source, **arrays)
    # Valid in the offline format, but cannot be passed to Weave's float32 loader.
    import_weave(source, layout)
    destination = tmp_path / "inputs"
    destination.mkdir()
    with pytest.raises(ValueError, match="joint_vel overflows Weave float32"):
        prepare_inputs([source], layout, destination, "train")
    assert not list(destination.iterdir())


@pytest.fixture
def paired(motion, tmp_path):
    source, layout, arrays = motion
    path = tmp_path / "reference.npz"
    import_weave(source, layout).save(path)
    reference = MotionClips.load(path)
    for name in ("terminated", "truncated", "clip_end"):
        arrays[name] = np.zeros(5, dtype=bool)
    arrays["clip_end"][[1, 4]] = True

    def rollout():
        path = tmp_path / "rollout.npz"
        metadata = {**reference.metadata, "reference_sha256": reference.sha256}
        # A producer writes a new recording before the reader validates it.
        np.savez(path, metadata=np.array(json.dumps(metadata)), **arrays)
        return MotionClips.load(path)

    return reference, arrays, rollout


def test_failed_terminal_has_priority_and_metrics_use_equal_clip_weight(paired):
    reference, arrays, read = paired
    arrays["terminated"][1] = arrays["truncated"][1] = True
    arrays["object_pos_w"][:2, 0] = 2
    # q and -q represent the same rotation.
    arrays["object_quat_w"] *= -1
    report = compare_clips(reference, read())
    assert [row["status"] for row in report["clips"]] == ["failed", "success"]
    assert report["success_rate"] == 0.5
    assert report["metrics_all"]["object_position_rmse_m"] == 1
    assert report["metrics_success"]["object_rotation_rmse_rad"] == 0
    arrays["terminated"][4] = True
    report = compare_clips(reference, read())
    assert report["success_rate"] == 0
    assert all(value is None for value in report["metrics_success"].values())
    json.dumps(report, allow_nan=False)


def test_incomplete_clips_and_post_terminal_frames_are_not_success(paired):
    reference, arrays, read = paired
    arrays["clip_end"][4] = False
    assert compare_clips(reference, read())["status"] == "incomplete"
    arrays["truncated"][4] = True
    report = compare_clips(reference, read())
    assert (
        report["status"] == "complete" and report["clips"][1]["status"] == "truncated"
    )
    arrays["terminated"][0] = True
    with pytest.raises(ValueError, match="after a terminal"):
        read()


def test_rotation_metrics_preserve_small_angles_and_quaternion_sign(paired):
    reference, arrays, read = paired
    for angle in (0.0, 1e-9, 1e-5, np.pi - 1e-9, np.pi):
        rotation = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        for sign in (1, -1):
            for field in ("body_quat_w", "object_quat_w"):
                arrays[field][:] = sign * rotation
            report = compare_clips(reference, read())
            for row in report["clips"]:
                for metric in ("body_rotation_rmse_rad", "object_rotation_rmse_rad"):
                    assert row[metric] == pytest.approx(angle, rel=1e-12, abs=1e-15)
    # Identical non-axis-aligned poses must have exactly zero error as well.
    rng = np.random.default_rng(42)
    for field in ("body_quat_w", "object_quat_w"):
        rotations = rng.normal(size=arrays[field].shape)
        rotations /= np.linalg.norm(rotations, axis=-1, keepdims=True)
        reference.arrays[field][:] = rotations
        arrays[field][:] = -rotations
    report = compare_clips(reference, read())
    assert report["metrics_all"]["body_rotation_rmse_rad"] == 0
    assert report["metrics_all"]["object_rotation_rmse_rad"] == 0


def test_layout_fps_and_reference_identity_are_checked(paired):
    reference, _, read = paired
    for key, value, message in (
        ("joint_names", ["knee", "hip"], "joint_names"),
        ("reference_sha256", "changed", "reference_sha256"),
    ):
        rollout = read()
        rollout.metadata[key] = value
        with pytest.raises(ValueError, match=message):
            compare_clips(reference, rollout)
    rollout = read()
    rollout.fps = 30
    with pytest.raises(ValueError, match="frame rates"):
        compare_clips(reference, rollout)


def test_short_failed_rollout_matches_by_name_and_rejects_early_completion(paired):
    reference, arrays, read = paired
    # Reorder clips; carry fails after its first frame, lift completes normally.
    for key in (*reference.frame_fields, "terminated", "truncated", "clip_end"):
        arrays[key] = arrays[key][[2, 0, 1]]
    arrays["motion_lengths"] = np.array([1, 2])
    arrays["motion_names"] = np.array(["carry", "lift"])
    arrays["terminated"][0] = True
    report = compare_clips(reference, read())
    assert [row["name"] for row in report["clips"]] == ["lift", "carry"]
    assert report["clips"][1]["status"] == "failed"
    assert report["clips"][1]["progress"] == pytest.approx(1 / 3)
    arrays["clip_end"][0] = True
    with pytest.raises(ValueError, match="before the reference ends"):
        compare_clips(reference, read())


def test_cli_import_inspect_and_incomplete_report(motion, paired, tmp_path, capsys):
    source, layout, _ = motion
    layout_file = tmp_path / "layout.json"
    layout_file.write_text(json.dumps(layout))
    output = tmp_path / "converted.npz"
    main(
        [
            "import-weave",
            "--input",
            str(source),
            "--layout",
            str(layout_file),
            "--output",
            str(output),
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodiedforge",
            "motion",
            "inspect",
            "--input",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["num_clips"] == 2
    _, arrays, read = paired
    arrays["clip_end"][4] = False
    read()
    report = tmp_path / "report.json"
    command = [
        "compare",
        "--reference",
        str(tmp_path / "reference.npz"),
        "--rollout",
        str(tmp_path / "rollout.npz"),
        "--output",
        str(report),
    ]
    with pytest.raises(SystemExit) as error:
        main(command)
    assert error.value.code == 1
    assert json.loads(report.read_text())["status"] == "incomplete"
    before = report.read_bytes()
    with pytest.raises(SystemExit):
        main(command)
    assert report.read_bytes() == before
