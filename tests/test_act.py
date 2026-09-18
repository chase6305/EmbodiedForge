"""ACT bridge contracts; SDK tests run in the optional LeRobot environment."""

import json
import subprocess
import sys

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.act import ACTAdapter, export_dataset
from embodiedforge.data import EpisodeRecorder, read_episodes
from embodiedforge.rollout import DemonstrationPolicy, run_rollout


@pytest.fixture
def recording(tmp_path):
    root = tmp_path / "recording"
    with VectorEnv(
        Config(num_envs=2, max_steps=4, render="raster", channels=("rgb",))
    ) as env:
        with EpisodeRecorder(root, env) as writer:
            run_rollout(
                env, DemonstrationPolicy(env.task.expert_action), steps=8, writer=writer
            )
    return root


def test_core_cli_does_not_require_lerobot():
    result = subprocess.run(
        [sys.executable, "-m", "embodiedforge", "act", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "export" in result.stdout and "evaluate" in result.stdout


def test_doctor_reports_missing_optional_sdk(monkeypatch):
    from embodiedforge import act

    def missing():
        raise ModuleNotFoundError("No module named 'lerobot'")

    monkeypatch.setattr(act, "_require_lerobot", missing)
    report = act.doctor()
    assert not report["ok"]
    assert report["native_act_implementation"] is False
    assert "lerobot" in report["error"]
    assert "--python" in report["hint"]


def test_runtime_report_identifies_actual_implementation():
    pytest.importorskip("lerobot")
    import hashlib
    from pathlib import Path

    from lerobot.policies.act.modeling_act import ACTPolicy

    from embodiedforge.act import runtime_report

    report = runtime_report()
    assert report["ok"]
    assert report["implementation"] == "lerobot"
    assert not report["native_act_implementation"]
    assert report["components"]["policy"]["module"] == ACTPolicy.__module__
    assert report["adapter"]["project_namespace"]
    for item in report["components"].values():
        assert not item["project_namespace"]
        assert (
            item["python_source_sha256"]
            == hashlib.sha256(Path(item["file"]).read_bytes()).hexdigest()
        )


def test_export_rejects_existing_output(recording):
    with pytest.raises(FileExistsError):
        export_dataset(recording, recording, "local/reach")


@pytest.mark.parametrize("status", ["recording", "failed"])
def test_export_rejects_unfinished_recording(recording, tmp_path, status):
    path = recording / "run.json"
    manifest = json.loads(path.read_text())
    manifest["status"] = status
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="closed"):
        export_dataset(recording, tmp_path / "output", "local/reach")
    assert not (tmp_path / "output").exists()


def test_export_rejects_empty_and_missing_images(recording, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "run.json").write_bytes((recording / "run.json").read_bytes())
    with pytest.raises(ValueError, match="No complete"):
        export_dataset(empty, tmp_path / "output", "local/reach")
    source = tmp_path / "state-only"
    with VectorEnv(Config(num_envs=1, max_steps=2)) as env:
        with EpisodeRecorder(source, env) as writer:
            run_rollout(
                env, DemonstrationPolicy(env.task.expert_action), steps=2, writer=writer
            )
    with pytest.raises(ValueError, match="RGB"):
        export_dataset(source, tmp_path / "output", "local/reach", images=True)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("images", [False, True])
def test_real_lerobot_roundtrip_and_episode_padding(recording, tmp_path, images):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = tmp_path / "dataset"
    metadata = export_dataset(recording, output, "local/reach", images=images)
    dataset = LeRobotDataset(
        "local/reach",
        root=output,
        delta_timestamps={"action": [0, 0.02, 0.04]},
    )
    episodes = list(read_episodes(recording))
    assert metadata["episodes"] == len(episodes) == 4
    assert len(dataset) == metadata["frames"] == 16
    offset = 0
    for episode in episodes:
        for t in range(len(episode["action"])):
            row = dataset[offset + t]
            np.testing.assert_allclose(
                row[metadata["state_key"]], episode["obs/proprio"][t]
            )
            np.testing.assert_allclose(row["action"][0], episode["action"][t])
            if images:
                decoded = row[metadata["camera_keys"][0]].permute(1, 2, 0).numpy()
                np.testing.assert_allclose(
                    decoded, episode["obs/rgb"][t, 0] / 255, atol=1e-7
                )
        terminal = dataset[offset + len(episode["action"]) - 1]
        assert terminal["action_is_pad"].tolist() == [False, True, True]
        np.testing.assert_allclose(terminal["action"][2], episode["action"][-1])
        offset += len(episode["action"])


def test_failed_export_is_not_published(recording, tmp_path, monkeypatch):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    def fail(*args, **kwargs):
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(LeRobotDataset, "add_frame", fail)
    with pytest.raises(RuntimeError, match="simulated write failure"):
        export_dataset(recording, tmp_path / "output", "local/reach")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output-*"))


@pytest.mark.parametrize("images", [False, True])
def test_real_checkpoint_adapter_normalization_and_subset_rows(
    recording, tmp_path, images, monkeypatch
):
    pytest.importorskip("lerobot")
    import torch
    from lerobot.configs import FeatureType
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.policies.act import ACTConfig, ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    torch.set_num_threads(1)
    output = tmp_path / "dataset"
    mapping = export_dataset(recording, output, "local/reach", images=images)
    metadata = LeRobotDatasetMetadata("local/reach", root=output)
    features = dataset_to_policy_features(metadata.features)
    config = ACTConfig(
        input_features={
            k: v for k, v in features.items() if v.type != FeatureType.ACTION
        },
        output_features={"action": features["action"]},
        device="cpu",
        chunk_size=3,
        n_action_steps=2,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_vae_encoder_layers=1,
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(config).eval()
    pre, post = make_pre_post_processors(config, dataset_stats=metadata.stats)
    checkpoint = tmp_path / "checkpoint"
    policy.save_pretrained(checkpoint)
    pre.save_pretrained(checkpoint)
    post.save_pretrained(checkpoint)
    adapter = ACTAdapter(checkpoint, mapping)
    with VectorEnv(Config(**mapping["config"])) as env:
        obs = env.observe()
        batch = {mapping["state_key"]: torch.from_numpy(obs["proprio"])}
        if mapping["env_state_key"]:
            batch[mapping["env_state_key"]] = batch[mapping["state_key"]].clone()
        if images:
            batch[mapping["camera_keys"][0]] = (
                torch.from_numpy(obs["rgb"][:, 0]).permute(0, 3, 1, 2).float() / 255
            )
        expected = post(policy.predict_action_chunk(pre(batch)))[:, :2].numpy()
        low, high = mapping["env_spec"]["action_range"]
        actual = adapter.act(obs)
        np.testing.assert_allclose(actual, np.clip(expected, low, high), atol=1e-6)
        subset = adapter.act({k: v[1:] for k, v in obs.items()})
        np.testing.assert_allclose(subset, actual[1:], atol=1e-5)
        assert actual.shape == (2, 2, 2)
        # Same weights with different output normalization are different policies.
        import shutil

        from safetensors.torch import load_file, save_file

        changed_path = tmp_path / "changed-normalization"
        shutil.copytree(checkpoint, changed_path)
        post_config = json.loads(
            (changed_path / "policy_postprocessor.json").read_text()
        )
        stats_path = changed_path / post_config["steps"][0]["state_file"]
        stats = load_file(str(stats_path))
        stats["action.mean"].fill_(-1000 if actual[0, 0, 0] >= 0 else 1000)
        save_file(stats, str(stats_path))
        changed = ACTAdapter(changed_path, mapping)
        assert changed.checkpoint_sha256 == adapter.checkpoint_sha256
        assert changed.checkpoint_fingerprint != adapter.checkpoint_fingerprint
        assert not np.allclose(changed.act(obs), actual)
        np.testing.assert_array_equal(adapter.act(obs), actual)

    # An edit during loading must not be reported as a stable saved policy.
    from lerobot.policies import factory

    make_processors = factory.make_pre_post_processors

    def edit_during_load(*args, **kwargs):
        processors = make_processors(*args, **kwargs)
        path = checkpoint / "config.json"
        path.write_text(path.read_text() + "\n")
        return processors

    with monkeypatch.context() as patch:
        patch.setattr(factory, "make_pre_post_processors", edit_during_load)
        with pytest.raises(ValueError, match="Checkpoint changed while loading"):
            ACTAdapter(checkpoint, mapping)
    config.temporal_ensemble_coeff = 0.01
    config.n_action_steps = 1
    config.save_pretrained(checkpoint)
    with pytest.raises(ValueError, match="temporal ensembling"):
        ACTAdapter(checkpoint, mapping)


def test_evaluation_refuses_failed_run(tmp_path):
    from argparse import Namespace

    from embodiedforge.act import evaluate

    (tmp_path / "act-run.json").write_text(
        json.dumps({"schema_version": 1, "status": "failed"})
    )
    with pytest.raises(ValueError, match="completed"):
        evaluate(Namespace(run=tmp_path))


@pytest.mark.parametrize("images", [False, True])
def test_real_training_and_closed_loop_evaluation(recording, tmp_path, images):
    pytest.importorskip("lerobot")
    from argparse import Namespace

    from safetensors.torch import load_file

    from embodiedforge.act import benchmark, bundle, evaluate, resume, train

    dataset = tmp_path / "dataset"
    export_dataset(recording, dataset, "local/reach", images=images)
    output = tmp_path / "train"
    train(
        Namespace(
            dataset=dataset,
            output=output,
            action_steps=2,
            chunk_size=3,
            device="cpu",
            steps=2,
            batch_size=2,
            seed=1001,
            small_model=True,
            save_freq=1,
            learning_rate=2e-5,
        )
    )
    manifest = json.loads((output / "act-run.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["runtime"]["implementation"] == "lerobot"
    assert manifest["dataset_fingerprint"]["algorithm"] == "sha256"
    assert "--policy.push_to_hub=false" in manifest["command"]
    assert "--save_freq=1" in manifest["command"]
    assert (output / "checkpoints/000001/pretrained_model/model.safetensors").is_file()
    evaluation = Namespace(
        run=output,
        num_envs=2,
        steps=8,
        seed=2001,
        device="cpu",
        output=tmp_path / "metrics.json",
    )
    result = evaluate(evaluation)
    assert json.loads(evaluation.output.read_text()) == result
    with pytest.raises(FileExistsError):
        evaluate(evaluation)
    assert result["completed_episodes"] == 4
    assert np.isfinite(result["mean_step_reward"])
    assert len(result["checkpoint_sha256"]) == 64
    assert result["environment_config"]["seed"] == 2001
    assert result["inference_settings"]["device"] == "cpu"
    assert result["inference_settings"]["action_steps"] == 2
    assert len(result["checkpoint_fingerprint"]["files"]) == 6
    assert (
        result["runtime"]["components"]["policy"]
        == manifest["runtime"]["components"]["policy"]
    )
    comparison_args = Namespace(
        run=output,
        num_envs=2,
        seeds=[2001, 2002],
        device="cpu",
        output=tmp_path / "benchmark.json",
    )
    comparison = benchmark(comparison_args)
    assert json.loads(comparison_args.output.read_text()) == comparison
    assert comparison["checkpoint_fingerprint"] == result["checkpoint_fingerprint"]
    repeated = benchmark(Namespace(**{**vars(comparison_args), "output": None}))
    assert repeated == comparison
    for name in ("act", "expert"):
        assert comparison["summary"][name]["episodes"] == 4
        assert comparison["summary"][name]["successes"] == sum(
            sum(row[name]["episode_success"]) for row in comparison["per_seed"]
        )
    for row in comparison["per_seed"]:
        assert row["environment_config"]["seed"] == row["seed"]
        assert (
            row["act"]["initial_observation_sha256"]
            == row["expert"]["initial_observation_sha256"]
        )
        assert max(row["act"]["episode_length"]) <= 4
    with pytest.raises(FileExistsError):
        benchmark(comparison_args)
    resumed = tmp_path / "resumed"
    with pytest.raises(ValueError, match="total target"):
        resume(Namespace(run=output, output=resumed, steps=2))
    data_path = dataset / "embodiedforge.json"
    original_data = data_path.read_text()
    changed = json.loads(original_data)
    changed["frames"] += 1
    data_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Dataset manifest changed"):
        resume(Namespace(run=output, output=resumed, steps=4))
    data_path.write_text(original_data)
    # Changing payloads without changing the manifest must fail before launching
    # training or publishing any startup record (including for RGB datasets).
    from embodiedforge.act import _attempt_paths

    for payload in (
        dataset / "meta/stats.json",
        next((dataset / "data").rglob("*.parquet")),
    ):
        original = payload.read_bytes()
        payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        try:
            with pytest.raises(ValueError, match="Dataset content changed") as error:
                resume(Namespace(run=output, output=resumed, steps=4))
            assert payload.relative_to(dataset).as_posix() in str(error.value)
            assert not resumed.exists()
            assert not _attempt_paths(resumed)[0].exists()
        finally:
            payload.write_bytes(original)
    resume(Namespace(run=output, output=resumed, steps=4))
    resumed_manifest = json.loads((resumed / "act-run.json").read_text())
    assert resumed_manifest["resumed_from"]["step"] == 2
    assert (
        resumed_manifest["resumed_from"]["dataset_verification"]["status"] == "verified"
    )
    assert resumed_manifest["dataset_fingerprint"] == manifest["dataset_fingerprint"]
    assert resumed_manifest["status"] == "complete"
    assert json.loads((output / "act-run.json").read_text()) == manifest
    source_step = output / "checkpoints/last/training_state/training_step.json"
    target_step = resumed / "checkpoints/last/training_state/training_step.json"
    assert json.loads(source_step.read_text())["step"] == 2
    assert json.loads(target_step.read_text())["step"] == 4
    optimizer = load_file(
        str(resumed / "checkpoints/last/training_state/optimizer_state.safetensors")
    )
    optimizer_steps = [
        float(value) for key, value in optimizer.items() if key.endswith("/step")
    ]
    assert optimizer_steps and set(optimizer_steps) == {4.0}
    resumed_eval = evaluate(
        Namespace(run=resumed, num_envs=2, steps=8, seed=2001, device="cpu")
    )
    assert resumed_eval["completed_episodes"] == 4
    assert resumed_eval["checkpoint_sha256"] != result["checkpoint_sha256"]
    source_eval = evaluate(
        Namespace(run=output, num_envs=2, steps=8, seed=2001, device="cpu")
    )
    assert source_eval["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert source_eval == result
    portable = tmp_path / "portable"
    bundle(Namespace(run=output, output=portable))
    moved_bundle = tmp_path / "moved-portable"
    portable.rename(moved_bundle)
    # Hide original paths while evaluating the copied policy and processors.
    hidden_run, hidden_dataset = tmp_path / "hidden-run", tmp_path / "hidden-data"
    output.rename(hidden_run)
    dataset.rename(hidden_dataset)
    try:
        portable_eval = evaluate(
            Namespace(bundle=moved_bundle, num_envs=2, steps=8, seed=2001, device="cpu")
        )
        for key in (
            "completed_episodes",
            "success_rate",
            "mean_step_reward",
            "clipped_action_fraction",
            "checkpoint_sha256",
            "checkpoint_fingerprint",
            "environment_config",
        ):
            assert portable_eval[key] == result[key]
        assert portable_eval["checkpoint_selection"]["requested"] == "bundle"
        portable_comparison = benchmark(
            Namespace(bundle=moved_bundle, num_envs=2, seeds=[2001, 2002], device="cpu")
        )
        assert portable_comparison["per_seed"] == comparison["per_seed"]
        assert portable_comparison["summary"] == comparison["summary"]
    finally:
        hidden_run.rename(output)
        hidden_dataset.rename(dataset)
    earlier = evaluate(
        Namespace(
            run=output, checkpoint="1", num_envs=2, steps=8, seed=2001, device="cpu"
        )
    )
    assert earlier["checkpoint_selection"]["step"] == 1
    assert earlier["checkpoint_sha256"] != result["checkpoint_sha256"]
    branch = tmp_path / "from-step-one"
    resume(Namespace(run=output, output=branch, checkpoint="1", steps=3))
    assert (
        json.loads((branch / "act-run.json").read_text())["resumed_from"]["step"] == 1
    )
    assert (
        json.loads(
            (branch / "checkpoints/last/training_state/training_step.json").read_text()
        )["step"]
        == 3
    )
    with pytest.raises(FileExistsError):
        resume(Namespace(run=output, output=resumed, steps=6))
    manifest["dataset"]["env_spec"]["action_units"] = "radians"
    (output / "act-run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="action_units"):
        evaluate(Namespace(run=output, num_envs=2, steps=8, seed=2001, device="cpu"))


@pytest.mark.parametrize("seeds", [[], [1, 1], [-1], [True]])
def test_benchmark_rejects_invalid_seeds_before_loading_model(seeds):
    from argparse import Namespace

    from embodiedforge.act import benchmark

    with pytest.raises(ValueError, match="nonnegative and unique"):
        benchmark(Namespace(seeds=seeds))


def test_startup_record_recovers_only_after_lock_released(tmp_path):
    from embodiedforge.act import _attempt_paths, _resume_report, _run_lock

    run = tmp_path / "interrupted"
    startup, _ = _attempt_paths(run)
    with _run_lock(run):
        startup.write_text(json.dumps({"schema_version": 1, "status": "running"}))
        with pytest.raises(ValueError, match="still active"):
            _resume_report(run)
    recovered = _resume_report(run)
    assert recovered["status"] == "interrupted"
    assert json.loads(startup.read_text())["status"] == "running"
    assert not run.exists()


def test_running_record_without_lock_is_not_assumed_dead(tmp_path):
    from embodiedforge.act import _attempt_paths, _resume_report

    run = tmp_path / "unknown"
    startup, _ = _attempt_paths(run)
    startup.write_text(json.dumps({"schema_version": 1, "status": "running"}))
    with pytest.raises(ValueError, match="missing its lock"):
        _resume_report(run)


def test_training_failure_before_output_creation_keeps_startup_record(tmp_path):
    pytest.importorskip("lerobot")
    from embodiedforge.act import _attempt_paths, _execute_training

    output = tmp_path / "early-failure"
    with pytest.raises(subprocess.CalledProcessError):
        _execute_training(output, {}, [sys.executable, "-c", "raise SystemExit(3)"])
    startup, lock = _attempt_paths(output)
    assert not output.exists()
    assert lock.is_file()
    report = json.loads(startup.read_text())
    assert report["status"] == "failed"
    with pytest.raises(FileExistsError, match="startup record"):
        _execute_training(output, {}, [sys.executable, "-c", "pass"])
    assert json.loads(startup.read_text()) == report


def test_forced_launcher_exit_preserves_checkpoint_recovery(recording, tmp_path):
    pytest.importorskip("lerobot")
    import os
    import signal
    import time
    from argparse import Namespace

    from safetensors.torch import load_file

    from embodiedforge.act import _attempt_paths, _resume_report, evaluate, resume

    dataset = tmp_path / "dataset"
    export_dataset(recording, dataset, "local/interruption")
    run = tmp_path / "interrupted"
    recovered = tmp_path / "recovered"
    command = [
        sys.executable,
        "-m",
        "embodiedforge.act",
        "train",
        "--dataset",
        str(dataset),
        "--output",
        str(run),
        "--steps",
        "100000",
        "--small-model",
        "--chunk-size",
        "3",
        "--action-steps",
        "2",
        "--save-freq",
        "20",
    ]
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    with (tmp_path / "interrupted.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 45
            last_step = run / "checkpoints/last/training_state/training_step.json"
            while not last_step.is_file():
                assert process.poll() is None, (
                    tmp_path / "interrupted.log"
                ).read_text()
                assert time.monotonic() < deadline, "Trainer did not checkpoint in time"
                time.sleep(0.02)
            # Kill only our launcher. Its upstream trainer remains alive and must
            # retain the inherited lock, even though no parent manifest is written.
            process.kill()
            process.wait(timeout=5)
            assert not (run / "act-run.json").exists()
            startup, _ = _attempt_paths(run)
            assert json.loads(startup.read_text())["status"] == "running"
            with pytest.raises(ValueError, match="still active"):
                resume(Namespace(run=run, output=recovered, steps=100002))
            os.killpg(process.pid, signal.SIGKILL)
            deadline = time.monotonic() + 5
            while True:
                try:
                    assert _resume_report(run)["status"] == "interrupted"
                    break
                except ValueError as exc:
                    assert "still active" in str(exc)
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
            saved_step = json.loads(last_step.read_text())["step"]
            evaluated = evaluate(
                Namespace(
                    run=run,
                    checkpoint="latest",
                    num_envs=2,
                    steps=8,
                    seed=2001,
                    device="cpu",
                )
            )
            assert evaluated["checkpoint_selection"]["source_status"] == "interrupted"
            assert evaluated["checkpoint_selection"]["step"] == saved_step
            resume(Namespace(run=run, output=recovered, steps=saved_step + 2))
            metadata = json.loads((recovered / "act-run.json").read_text())
            assert metadata["status"] == "complete"
            assert metadata["resumed_from"]["source_status"] == "interrupted"
            assert (
                metadata["resumed_from"]["dataset_verification"]["status"] == "verified"
            )
            state = load_file(
                str(
                    recovered
                    / "checkpoints/last/training_state/optimizer_state.safetensors"
                )
            )
            steps = {
                float(value) for key, value in state.items() if key.endswith("/step")
            }
            assert steps == {float(saved_step + 2)}
            assert json.loads(last_step.read_text())["step"] == saved_step
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
