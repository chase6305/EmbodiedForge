"""RLinf adapter contracts; these tests do not launch training or simulators."""

import json
import os
import random
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiedforge import _rlinf_worker as worker
from embodiedforge import rlinf
from embodiedforge._microduck_process import ProcessInterrupted


def checkpoint(root, *, seed=1234, num_envs=32):
    path = root / "global_step_200"
    for relative in (
        rlinf.WEIGHTS,
        "actor/dcp_checkpoint/.metadata",
        "actor/dcp_checkpoint/__0_0.distcp",
        "actor/sac_components/alpha/dcp_checkpoint/.metadata",
        "actor/sac_components/alpha/dcp_checkpoint/__0_0.distcp",
        "actor/sac_components/target_model/checkpoint_rank_0.pt",
        "actor/sac_components/replay_buffer/rank_0/metadata.json",
        "actor/sac_components/replay_buffer/rank_0/trajectory_index.json",
    ):
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    replay = path / "actor/sac_components/replay_buffer/rank_0"
    rlinf.write_json(
        replay / "metadata.json",
        {
            "trajectory_format": "pt",
            "seed": seed,
            "size": 1,
            "trajectory_counter": 1,
            "total_samples": 2 * num_envs,
        },
    )
    rlinf.write_json(
        replay / "trajectory_index.json",
        {
            "trajectory_id_list": [0],
            "trajectory_index": {
                "0": {
                    "model_weights_id": "199",
                    "shape": [2, num_envs, 1],
                    "num_samples": 2 * num_envs,
                    "trajectory_id": 0,
                }
            },
        },
    )
    (replay / "trajectory_0_199.pt").write_bytes(b"fixture")
    return path


@pytest.mark.parametrize(
    "outcome",
    ["complete", "missing_result", "failed", "interrupted", "preparing_interrupt"],
)
def test_launcher_snapshot_environment_and_status(tmp_path, monkeypatch, outcome):
    source = checkpoint(tmp_path / "source", seed=37, num_envs=8)
    python = tmp_path / "SDK env/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    output = tmp_path / "run with spaces"
    args = rlinf.parser().parse_args(
        [
            "train",
            "--repo",
            str(tmp_path / "repo"),
            "--python",
            str(python),
            "--output",
            str(output),
            "--iterations",
            "300",
            "--gpu",
            "3",
            "--resume",
            str(source),
            "--quiet",
        ]
    )
    monkeypatch.setattr(rlinf, "check_checkout", lambda _: None)
    monkeypatch.setenv("PYTHONPATH", "unrelated")
    monkeypatch.setenv("RAY_ADDRESS", "other-cluster:1234")
    monkeypatch.setenv("RLINF_NODE_RANK", "7")
    monkeypatch.setenv("RLINF_EXT_MODULE", "unrelated.extension")
    monkeypatch.setenv("EF_RLINF_SEED", "999")
    monkeypatch.setenv("MASTER_PORT", "9876")
    monkeypatch.setenv("WORLD_SIZE", "1")

    def run(command, *, cwd, env, log_path, echo):
        assert echo is False
        assert command[0] == str(python)
        assert cwd == output and log_path == output / "worker.log"
        assert env["CUDA_VISIBLE_DEVICES"] == "3"
        assert env["RLINF_EXT_MODULE"] == "embodiedforge._rlinf_worker"
        assert env["EF_RLINF_SEED"] == "37"
        assert "unrelated" not in env["PYTHONPATH"]
        assert env["PYTHONPATH"].split(os.pathsep) == [
            str(output / "python"),
            str(args.repo.resolve()),
        ]
        assert (output / "python/embodiedforge").resolve() == Path(
            rlinf.__file__
        ).parent
        for key in ("RAY_ADDRESS", "RLINF_NODE_RANK", "MASTER_PORT", "WORLD_SIZE"):
            assert key not in env
        request = json.loads((output / "request.json").read_text())
        assert request["quiet"] is True
        copied = Path(request["checkpoint"])
        assert copied != source and rlinf.checkpoint_step(copied) == 200
        assert request["start_step"] == 200 and request["iterations"] == 300
        assert request["seed"] == 37 and request["num_envs"] == 8
        assert request["input_files"] == rlinf.inventory(output / "inputs")
        (source / rlinf.WEIGHTS).write_bytes(b"source changed after snapshot")
        assert (copied / rlinf.WEIGHTS).read_bytes() == b"fixture"
        checkpoints = output / rlinf.CONFIG / "checkpoints"
        saved = checkpoint(checkpoints).rename(checkpoints / "global_step_400")
        assert rlinf.checkpoint_step(saved) == 400
        # A later interrupted upstream save must not hide the earlier save.
        partial = checkpoints / "global_step_500/actor"
        partial.mkdir(parents=True)
        (partial / "partial.pt").write_bytes(b"unfinished")
        if outcome == "failed":
            raise RuntimeError("worker failed")
        if outcome == "interrupted":
            raise KeyboardInterrupt
        if outcome == "complete":
            rlinf.write_json(output / "result.json", {"status": "complete"})

    monkeypatch.setattr(rlinf, "run_process", run)
    if outcome == "complete":
        for name, value in (("seed", 1234), ("num_envs", 32)):
            conflicting = SimpleNamespace(**{**vars(args), name: value})
            with pytest.raises(ValueError, match="differs from the checkpoint"):
                rlinf.launch(conflicting)
            assert not output.exists()
    previous_handler = signal.getsignal(signal.SIGTERM)
    if outcome == "preparing_interrupt":

        def interrupt_copy(*args, **kwargs):
            assert (
                json.loads((output / "run.json").read_text())["status"] == "preparing"
            )
            assert json.loads((output / "request.json").read_text())[
                "input_source"
            ] == str(source)
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)

        monkeypatch.setattr(rlinf.shutil, "copytree", interrupt_copy)
    if outcome == "complete":
        assert rlinf.launch(args) == output
    else:
        error = {
            "missing_result": FileNotFoundError,
            "failed": RuntimeError,
            "interrupted": KeyboardInterrupt,
            "preparing_interrupt": ProcessInterrupted,
        }[outcome]
        with pytest.raises(error):
            rlinf.launch(args)
    report = json.loads((output / "run.json").read_text())
    status = report["status"]
    expected = {"missing_result": "failed", "preparing_interrupt": "interrupted"}.get(
        outcome, outcome
    )
    assert status == expected
    if outcome == "preparing_interrupt":
        assert report["latest_checkpoint"] is None
        assert report["last_saved_step"] is None
    else:
        assert Path(report["latest_checkpoint"]).name == "global_step_400"
        assert report["last_saved_step"] == 400
        assert report["checkpoint_validation"] == "layout_only"
    assert signal.getsignal(signal.SIGTERM) is previous_handler
    with pytest.raises(FileExistsError):
        rlinf.launch(args)


@pytest.mark.parametrize(
    "missing",
    [
        "actor/sac_components/replay_buffer/rank_0/trajectory_index.json",
        "actor/sac_components/replay_buffer/rank_0/trajectory_0_199.pt",
        "actor/dcp_checkpoint/__0_0.distcp",
    ],
)
def test_incomplete_resume_rejected(tmp_path, missing):
    source = checkpoint(tmp_path)
    (source / missing).unlink()
    with pytest.raises(ValueError, match="SAC.*checkpoint"):
        rlinf.checkpoint_step(source)
    with pytest.raises(ValueError, match="global_step_N"):
        rlinf.checkpoint_step(source / "actor")


def test_resume_only_requires_active_replay_window(tmp_path, monkeypatch):
    source = checkpoint(tmp_path)
    replay = source / "actor/sac_components/replay_buffer/rank_0"
    # Evicted trajectories remain in upstream's historical index but have no file.
    monkeypatch.setattr(rlinf, "REPLAY_WINDOW", 1)
    rlinf.write_json(
        replay / "metadata.json",
        {
            "trajectory_format": "pt",
            "seed": 1234,
            "size": 2,
            "trajectory_counter": 2,
            "total_samples": 128,
        },
    )
    rlinf.write_json(
        replay / "trajectory_index.json",
        {
            "trajectory_id_list": [0, 1],
            "trajectory_index": {
                str(i): {
                    "model_weights_id": str(199 + i),
                    "shape": [2, 32, 1],
                    "num_samples": 64,
                    "trajectory_id": i,
                }
                for i in range(2)
            },
        },
    )
    (replay / "trajectory_0_199.pt").unlink()
    (replay / "trajectory_1_200.pt").write_bytes(b"fixture")
    assert rlinf.checkpoint_step(source) == 200
    (replay / "trajectory_1_200.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="missing active replay trajectory"):
        rlinf.checkpoint_step(source)
    (replay / "trajectory_1_200.pt").write_bytes(b"fixture")
    metadata_path = replay / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for name in ("size", "trajectory_counter", "total_samples"):
        rlinf.write_json(metadata_path, {**metadata, name: 0})
        with pytest.raises(ValueError, match=f"replay {name} differs"):
            rlinf.checkpoint_step(source)


@pytest.mark.parametrize("failure", [False, True, "worker"])
def test_ray_local_retry_address_propagation_and_cleanup(monkeypatch, failure):
    calls = []
    previous_failure_handler = signal.getsignal(signal.SIGUSR1)

    def init(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("first local startup failed")
        return SimpleNamespace(address_info={"gcs_address": "127.0.0.1:12345"})

    ray = SimpleNamespace(init=init, shutdown=lambda: calls.append("shutdown"))
    monkeypatch.setitem(sys.modules, "ray", ray)
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setattr(worker.os, "cpu_count", lambda: 32)

    def cluster():
        with pytest.raises(ConnectionError):
            ray.init(address="auto", namespace="RLinf")
        assert not calls  # Never ask Ray to discover or connect to an existing job.
        with pytest.raises(RuntimeError, match="local startup failed"):
            ray.init(namespace="RLinf")
        ray.shutdown()  # Pinned Cluster._start_local_ray cleans up before retry.
        ray.init(namespace="RLinf")
        assert os.environ["RAY_ADDRESS"] == "127.0.0.1:12345"
        assert ray.init is init
        return "owned cluster"

    def entry(cfg):
        assert cfg == "configuration"
        assert globals()["Cluster"]() == "owned cluster"
        if failure == "worker":
            signal.raise_signal(signal.SIGUSR1)
        elif failure:
            raise KeyboardInterrupt

    monkeypatch.setitem(entry.__globals__, "Cluster", cluster)

    monkeypatch.setattr(
        worker.runpy,
        "run_path",
        lambda *a, **kw: {
            "main": SimpleNamespace(__wrapped__=entry),
        },
    )
    if failure == "worker":
        with pytest.raises(RuntimeError, match="RLinf worker failed"):
            worker.execute("configuration", Path("entry.py"))
    elif failure:
        with pytest.raises(KeyboardInterrupt):
            worker.execute("configuration", Path("entry.py"))
    else:
        worker.execute("configuration", Path("entry.py"))
    assert (
        calls
        == [
            {
                "address": "local",
                "namespace": "RLinf",
                "num_cpus": 4,
            },
            "shutdown",
        ]
        * 2
    )
    assert ray.init is init
    assert entry.__globals__["Cluster"] is cluster
    assert signal.getsignal(signal.SIGUSR1) is previous_failure_handler


def test_pinned_upstream_config_train_resume_and_eval(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    root = os.environ.get("RLINF_ROOT")
    if not root:
        pytest.skip(
            "Set RLINF_ROOT to the pinned checkout for the upstream config contract"
        )
    rlinf.check_checkout(Path(root))
    monkeypatch.setenv("EMBODIED_PATH", "will be replaced by selected checkout")
    request = dict(
        repo=root,
        output=str(tmp_path / "space in path"),
        seed=37,
        gpu=3,
        command="train",
        num_envs=8,
        start_step=8000,
        iterations=250,
        save_interval=100,
        checkpoint=str(tmp_path / "global_step_8000"),
        eval_epochs=3,
    )
    cfg = worker.compose_config(request)
    assert cfg.runner.max_steps == cfg.runner.max_epochs == 8250
    assert cfg.runner.save_interval == cfg.runner.val_check_interval == 100
    assert cfg.runner.resume_dir == request["checkpoint"]
    assert cfg.env.train.total_num_envs == 8 and cfg.env.eval.total_num_envs == 16
    assert cfg.actor.seed == cfg.env.train.seed == cfg.env.eval.seed == 37
    assert dict(cfg.cluster.component_placement) == {
        name: "3-3" for name in ("actor", "env", "rollout")
    }
    assert (
        not cfg.env.train.video_cfg.save_video and not cfg.env.eval.video_cfg.save_video
    )
    assert cfg.actor.model.action_dim == 4 and cfg.actor.model.obs_dim == 42
    assert cfg.actor.fsdp_config.use_orig_params is False
    assert cfg.algorithm.loss_type == "embodied_sac"
    assert cfg.algorithm.replay_buffer.sample_window_size == rlinf.REPLAY_WINDOW
    assert cfg.algorithm.replay_buffer.auto_save is False
    request.update(command="evaluate", checkpoint=str(tmp_path / "weights.pt"))
    cfg = worker.compose_config(request)
    assert cfg.runner.only_eval and cfg.runner.task_type == "embodied_eval"
    assert cfg.runner.ckpt_path == request["checkpoint"]
    assert cfg.rollout.model == cfg.actor.model
    assert cfg.env.eval.total_num_envs == 8 and cfg.env.eval.rollout_epoch == 3
    assert (
        cfg.cluster.component_placement.rollout
        == cfg.cluster.component_placement.env
        == "3-3"
    )
    request.update(policy_format="onnx", checkpoint=str(tmp_path / "policy.onnx"))
    cfg = worker.compose_config(request)
    assert cfg.env.eval.init_params.control_mode == "pd_ee_delta_pos"
    assert cfg.env.eval.total_num_envs == 8 and cfg.env.eval.rollout_epoch == 3


def test_weights_and_metrics_describe_saved_artifacts(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tensorboard")
    from torch.utils.tensorboard import SummaryWriter

    weights = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(3, 4), "action_scale": torch.tensor(1.0)}, weights)
    summary = worker.weight_summary(weights)
    assert summary["state_tensor_elements"] == 13
    assert summary["state_tensor_bytes"] == 52
    assert summary["file_bytes"] == weights.stat().st_size
    assert summary["sha256"] == rlinf.sha256(weights)
    original_load = torch.load

    def replace_after_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        weights.write_bytes(b"replaced after deserialization")
        return loaded

    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", replace_after_load)
        assert worker.weight_summary(weights) == summary
    with SummaryWriter(tmp_path / "tensorboard") as writer:
        writer.add_scalar("eval/success_once", 0.25, 0)
        writer.add_scalar("eval/success_once", 0.75, 2)
        writer.add_scalar("eval/num_trajectories", 16, 2)
    assert worker.last_metrics(tmp_path, expected_step=2) == {
        "eval/success_once": {"step": 2, "value": 0.75},
        "eval/num_trajectories": {"step": 2, "value": 16},
    }
    with pytest.raises(RuntimeError, match="final log step 3"):
        worker.last_metrics(tmp_path, expected_step=3)
    with SummaryWriter(tmp_path / "tensorboard") as writer:
        writer.add_scalar("eval/success_once", 0, 3)
        writer.add_scalar("eval/num_trajectories", 0, 3)
    with pytest.raises(RuntimeError, match="completed no trajectories"):
        worker.last_metrics(tmp_path, expected_step=3)
    torch.save({"weight": torch.tensor(float("nan"))}, weights)
    with pytest.raises(ValueError, match="nonfinite"):
        worker.weight_summary(weights)


def test_pinned_upstream_mlp_weights_match_architecture(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    root = os.environ.get("RLINF_ROOT")
    if not root:
        pytest.skip("Set RLINF_ROOT to check the actual upstream CPU MLP")
    rlinf.check_checkout(Path(root))
    monkeypatch.syspath_prepend(root)
    previous = set(sys.modules)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        import numpy as np
        from rlinf.models.embodiment.mlp_policy import get_model
        from rlinf.scheduler.cluster import load_user_extension_module

        cfg = OmegaConf.create(
            dict(
                model_type="mlp_policy",
                obs_dim=42,
                action_dim=4,
                num_action_chunks=1,
                add_value_head=False,
                add_q_head=True,
            )
        )
        monkeypatch.setenv("RLINF_EXT_MODULE", "embodiedforge._rlinf_worker")
        monkeypatch.setenv("EF_RLINF_SEED", "37")
        monkeypatch.setenv("RANK", "0")
        # The actual worker hook must seed model initialization and exploration,
        # regardless of random draws made while importing its dependencies.
        samples = []
        for unrelated_seed in (10, 20):
            torch.manual_seed(unrelated_seed)
            random.seed(unrelated_seed)
            np.random.seed(unrelated_seed)
            load_user_extension_module()
            assert torch.initial_seed() == 37
            model = get_model(cfg)
            action, _ = model.predict_action_batch(
                {"states": torch.ones(8, 42)}, mode="train"
            )
            samples.append(
                (model.state_dict(), action, random.random(), np.random.rand())
            )
        for name, value in samples[0][0].items():
            torch.testing.assert_close(value, samples[1][0][name], atol=0, rtol=0)
        np.testing.assert_array_equal(samples[0][1], samples[1][1])
        assert samples[0][2:] == samples[1][2:]
        monkeypatch.setenv("RANK", "1")
        load_user_extension_module()
        assert torch.initial_seed() == 38
        weights = tmp_path / "full_weights.pt"
        torch.save(model.state_dict(), weights)
        rng = torch.get_rng_state().clone()
        summary = worker.weight_summary(weights, cfg)
        assert torch.equal(torch.get_rng_state(), rng)
        assert summary["parameters"]["total"] == sum(
            p.numel() for p in model.parameters()
        )
        assert summary["parameters"]["critic"] == sum(
            p.numel() for p in model.q_head.parameters()
        )
        damaged = model.state_dict()
        damaged["backbone.0.weight"] = damaged["backbone.0.weight"][:, :-1]
        torch.save(damaged, weights)
        with pytest.raises(RuntimeError, match="size mismatch"):
            worker.weight_summary(weights, cfg)
        damaged = {key: value.double() for key, value in model.state_dict().items()}
        torch.save(damaged, weights)
        with pytest.raises(ValueError, match="Wrong model tensor dtype"):
            worker.weight_summary(weights, cfg)
        damaged = model.state_dict()
        damaged["action_scale"] = torch.tensor(0.0)
        torch.save(damaged, weights)
        with pytest.raises(ValueError, match="Wrong fixed action transform"):
            worker.weight_summary(weights, cfg)
    finally:
        torch.set_num_threads(threads)
        for name in set(sys.modules) - previous:
            if name == "rlinf" or name.startswith("rlinf."):
                sys.modules.pop(name, None)


def test_dcp_shards_checked_against_real_metadata(tmp_path):
    torch = pytest.importorskip("torch")
    from torch.distributed.checkpoint import save

    for relative in (
        "actor/dcp_checkpoint",
        "actor/sac_components/alpha/dcp_checkpoint",
    ):
        save(
            {"model": {"weight": torch.arange(12).reshape(3, 4)}},
            checkpoint_id=tmp_path / relative,
        )
    worker.validate_dcp_files(tmp_path)
    shard = next((tmp_path / "actor/dcp_checkpoint").glob("*.distcp"))
    with shard.open("r+b") as stream:
        stream.truncate(shard.stat().st_size - 1)
    with pytest.raises(ValueError, match="Truncated or invalid DCP"):
        worker.validate_dcp_files(tmp_path)


@pytest.mark.parametrize("weight_storage", ["float32", "float16", "int8"])
def test_actor_export_prunes_training_heads_and_supports_dynamic_batches(
    tmp_path, monkeypatch, weight_storage
):
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    OmegaConf = pytest.importorskip("omegaconf").OmegaConf
    root = os.environ.get("RLINF_ROOT")
    if not root:
        pytest.skip("Set RLINF_ROOT to check export against the pinned CPU policy")
    rlinf.check_checkout(Path(root))
    monkeypatch.syspath_prepend(root)
    from embodiedforge._rlinf_export import export_actor
    from embodiedforge._rlinf_onnx import load_policy

    cfg = OmegaConf.create(
        dict(
            model_type="mlp_policy",
            obs_dim=42,
            action_dim=4,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=True,
        )
    )
    previous = set(sys.modules)
    threads = torch.get_num_threads()
    try:
        from rlinf.models.embodiment.mlp_policy import get_model

        torch.set_num_threads(1)
        model = get_model(cfg).eval()
        path = tmp_path / "weights.pt"
        torch.save(model.state_dict(), path)
        weights = worker.weight_summary(path, cfg)
        report = export_actor(
            path, tmp_path, cfg, weights["sha256"], weight_storage=weight_storage
        )
        assert report["weight_storage"] == weight_storage
        assert report["parameters"] == 143620 < weights["parameters"]["actor"]
        assert report["sha256"] == rlinf.sha256(tmp_path / "policy.onnx")
        assert not (tmp_path / "policy.pending.onnx").exists()
        graph = onnx.load(str(tmp_path / "policy.onnx"))
        assert sum(
            item.data_type == onnx.TensorProto.FLOAT16
            for item in graph.graph.initializer
        ) == (4 if weight_storage == "float16" else 0)
        assert sum(
            torch.tensor(item.dims).prod().item() for item in graph.graph.initializer
        ) <= 143620 + (772 if weight_storage == "int8" else 0)
        assert sum(
            item.data_type == onnx.TensorProto.INT8 for item in graph.graph.initializer
        ) == (4 if weight_storage == "int8" else 0)
        session, loaded = load_policy(tmp_path / "policy.onnx", report["sha256"])
        assert isinstance(session, ort.InferenceSession)
        assert loaded["sha256"] == report["sha256"]
        assert loaded["metadata"] == report["metadata"]
        states = torch.linspace(-3, 3, 13 * 42).reshape(13, 42)
        if weight_storage == "float16":
            with torch.no_grad():
                for layer in (model.backbone, model.actor_mean):
                    for parameter in layer.parameters():
                        if parameter.ndim == 2:
                            parameter.copy_(parameter.half().float())
        elif weight_storage == "int8":
            with torch.no_grad():
                for layer in (model.backbone, model.actor_mean):
                    for parameter in layer.parameters():
                        if parameter.ndim == 2:
                            scale = (
                                parameter.abs().amax(dim=1, keepdim=True) / 127
                            ).clamp_min(torch.finfo(torch.float32).tiny)
                            parameter.copy_(
                                (parameter / scale).round().clamp(-127, 127) * scale
                            )
        expected, _ = model.predict_action_batch({"states": states}, mode="eval")
        actual = session.run(["actions"], {"states": states.numpy()})[0]
        torch.testing.assert_close(
            torch.from_numpy(actual), expected[:, 0], atol=1e-5, rtol=1e-5
        )
        with pytest.raises(ValueError, match="changed before actor export"):
            export_actor(path, tmp_path, cfg, "0" * 64)
        assert report["sha256"] == rlinf.sha256(tmp_path / "policy.onnx")
        with pytest.raises(ValueError, match="changed before loading"):
            load_policy(tmp_path / "policy.onnx", "0" * 64)
        incompatible = tmp_path / "incompatible.onnx"
        onnx.helper.set_model_props(graph, {**report["metadata"], "task": "Other-v1"})
        onnx.save(graph, incompatible)
        with pytest.raises(ValueError, match="Incompatible RLinf ONNX task"):
            load_policy(incompatible, rlinf.sha256(incompatible))
        onnx.helper.set_model_props(graph, report["metadata"])
        graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 13
        onnx.save(graph, incompatible)
        with pytest.raises(ValueError, match=r"requires float32 states\[batch, 42\]"):
            load_policy(incompatible, rlinf.sha256(incompatible))
        if weight_storage == "float16":
            bad_state = model.state_dict()
            bad_state["actor_mean.weight"] = torch.full_like(
                bad_state["actor_mean.weight"], 1e6
            )
            bad_path = tmp_path / "out_of_range.pt"
            torch.save(bad_state, bad_path)
            with pytest.raises(ValueError, match="finite float16 storage"):
                export_actor(
                    bad_path,
                    tmp_path,
                    cfg,
                    rlinf.sha256(bad_path),
                    weight_storage="float16",
                )
            assert report["sha256"] == rlinf.sha256(tmp_path / "policy.onnx")
    finally:
        torch.set_num_threads(threads)
        for name in set(sys.modules) - previous:
            if name == "rlinf" or name.startswith("rlinf."):
                sys.modules.pop(name, None)


def test_incompatible_sdk_keeps_dependency_diagnostics(tmp_path, monkeypatch):
    versions = {
        name: "1.0"
        for name in (
            "omegaconf",
            "numpy",
            "mani_skill",
            "sapien",
            "tensorboard",
            "packaging",
        )
    }
    versions.update({"torch": "2.10.0+cu128", "ray": "2.57.0", "hydra-core": "1.3.2"})

    def version(name):
        if name not in versions:
            raise worker.importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(worker.importlib.metadata, "version", version)
    worker.record_environment(tmp_path)
    assert json.loads((tmp_path / "environment.json").read_text())["errors"] == []
    versions["hydra-core"] = "1.4.0.dev8"
    del versions["sapien"]
    with pytest.raises(RuntimeError, match="not a compatible RLinf SDK"):
        worker.record_environment(tmp_path)
    report = json.loads((tmp_path / "environment.json").read_text())
    assert report["packages"]["sapien"] is None
    assert report["errors"] == [
        "missing sapien",
        "hydra-core<1.4.0.dev8 required, found 1.4.0.dev8",
    ]
    del versions["packaging"]
    with pytest.raises(RuntimeError, match="missing packaging"):
        worker.record_environment(tmp_path)


@pytest.mark.parametrize("mutation", [None, "source", "snapshot"])
@pytest.mark.parametrize("policy_format", ["pytorch", "onnx"])
def test_evaluation_snapshot_must_match_stable_source(
    tmp_path, monkeypatch, mutation, policy_format
):
    source = tmp_path / "weights.pt"
    source.write_bytes(b"original weights")
    output = tmp_path / "evaluation"
    args = rlinf.parser().parse_args(
        [
            "evaluate",
            "--repo",
            str(tmp_path / "repo"),
            "--python",
            sys.executable,
            "--output",
            str(output),
            "--onnx" if policy_format == "onnx" else "--checkpoint",
            str(source),
        ]
    )
    monkeypatch.setattr(rlinf, "check_checkout", lambda _: None)
    original_copy = rlinf.shutil.copyfile

    def copy(src, dst):
        original_copy(src, dst)
        if mutation:
            (src if mutation == "source" else dst).write_bytes(b"changed weights!")

    monkeypatch.setattr(rlinf.shutil, "copyfile", copy)
    launched = []

    def run(*args, **kwargs):
        launched.append(True)
        assert kwargs["echo"] is True
        request = json.loads((output / "request.json").read_text())
        copied = Path(request["checkpoint"])
        assert request["policy_format"] == policy_format
        assert copied.name == (
            "policy.onnx" if policy_format == "onnx" else "full_weights.pt"
        )
        assert request["input_files"] == rlinf.inventory(output / "inputs")
        assert copied.read_bytes() == source.read_bytes() == b"original weights"
        source.write_bytes(b"changed after snapshot")
        assert copied.read_bytes() == b"original weights"
        rlinf.write_json(output / "result.json", {"status": "complete"})

    monkeypatch.setattr(rlinf, "run_process", run)
    if mutation:
        with pytest.raises(ValueError, match="changed during snapshot"):
            rlinf.launch(args)
        assert not launched
        assert not (output / "result.json").exists()
    else:
        assert rlinf.launch(args) == output
        assert launched
    status = json.loads((output / "run.json").read_text())
    assert status["status"] == ("failed" if mutation else "complete")


@pytest.mark.parametrize("activate_ok", [True, False])
def test_venv_activation_is_isolated_and_required(tmp_path, monkeypatch, activate_ok):
    sdk = tmp_path / "SDK with spaces"
    (sdk / "bin").mkdir(parents=True)
    (sdk / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    (sdk / "bin/activate").write_text(
        "export EF_TEST_SDK_MARKER=activated\n" if activate_ok else "return 7\n"
    )
    python = sdk / "bin/python"
    # This executable stands in for the SDK worker, recording its environment.
    python.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
request = json.loads(Path(sys.argv[-1]).read_text())
Path(request["output"], "result.json").write_text(json.dumps({
    "status": "complete", "activation": os.environ.get("EF_TEST_SDK_MARKER"),
}))
"""
    )
    python.chmod(0o755)
    monkeypatch.setattr(rlinf, "check_checkout", lambda _: None)
    monkeypatch.delenv("EF_TEST_SDK_MARKER", raising=False)
    output = tmp_path / "output"
    args = rlinf.parser().parse_args(
        [
            "train",
            "--repo",
            str(tmp_path / "repo"),
            "--python",
            str(python),
            "--output",
            str(output),
            "--iterations",
            "200",
        ]
    )
    if activate_ok:
        rlinf.launch(args)
        assert (
            json.loads((output / "result.json").read_text())["activation"]
            == "activated"
        )
    else:
        with pytest.raises(subprocess.CalledProcessError) as error:
            rlinf.launch(args)
        assert error.value.returncode == 7
        assert not (output / "result.json").exists()
    assert "EF_TEST_SDK_MARKER" not in os.environ
