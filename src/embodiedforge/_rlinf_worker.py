"""RLinf SDK-side configuration and execution; never imported by the core CLI."""

import hashlib
import importlib.metadata
import io
import json
import math
import os
import runpy
import signal
import sys
import tempfile
from pathlib import Path

from .h1 import write_json
from .rlinf import CONFIG, WEIGHTS, check_checkout, checkpoint_step, inventory


def register():
    """RLinf's worker extension hook runs before model construction in each process."""
    from rlinf.utils.utils import seed_everything

    seed_everything(int(os.environ["EF_RLINF_SEED"]) + int(os.environ["RANK"]))


def compose_config(request):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    config_dir = Path(request["repo"]) / "examples/embodiment/config"
    os.environ["EMBODIED_PATH"] = str(config_dir.parent)
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=CONFIG)
    with open_dict(cfg):
        # RLinf enumerates physical GPUs via NVML and overwrites worker CUDA
        # visibility from placements. The driver's CUDA mask alone is insufficient.
        if request["command"] != "export":
            cfg.cluster.component_placement = {
                component: f"{request['gpu']}-{request['gpu']}"
                for component in ("actor", "env", "rollout")
            }
        cfg.runner.logger.log_path = request["output"]
        cfg.runner.logger.experiment_name = CONFIG
        cfg.runner.logger.logger_backends = ["tensorboard"]
        cfg.actor.seed = request["seed"]
        cfg.env.train.seed = request["seed"]
        cfg.env.eval.seed = request["seed"]
        cfg.env.train.video_cfg.save_video = False
        cfg.env.eval.video_cfg.save_video = False
        if request["command"] == "train":
            final = request["start_step"] + request["iterations"]
            cfg.runner.max_steps = final
            cfg.runner.max_epochs = final
            cfg.runner.save_interval = request["save_interval"]
            cfg.runner.val_check_interval = request["save_interval"]
            cfg.runner.resume_dir = request["checkpoint"]
            cfg.env.train.total_num_envs = request["num_envs"]
        else:
            cfg.runner.task_type = "embodied_eval"
            cfg.runner.only_eval = True
            cfg.runner.ckpt_path = request["checkpoint"]
            # Training's rollout.model is only a stub. The eval worker builds
            # directly from it and needs the actor's architecture and Q heads.
            cfg.rollout.model = OmegaConf.create(
                OmegaConf.to_container(cfg.actor.model, resolve=True)
            )
            cfg.env.eval.total_num_envs = request["num_envs"]
            cfg.env.eval.rollout_epoch = request["eval_epochs"]
            if request.get("policy_format") == "onnx":
                # The Ray entry normally resolves panda-ee-dpos to this mode.
                # Direct ONNX evaluation does not start its Cluster validator.
                cfg.env.eval.init_params.control_mode = "pd_ee_delta_pos"
    OmegaConf.resolve(cfg)
    return cfg


def weight_summary(path, model_cfg=None):
    import torch

    payload = path.read_bytes()
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a nonempty RLinf model state dictionary")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid or nonfinite model tensor: {name}")
    summary = {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_bytes": len(payload),
        "state_tensor_elements": sum(v.numel() for v in state.values()),
        "state_tensor_bytes": sum(v.numel() * v.element_size() for v in state.values()),
    }
    if model_cfg is not None:
        from rlinf.models.embodiment.mlp_policy import get_model

        # Build only the upstream structure, avoiding random initialization and
        # extra parameter storage. Loading remains on CPU and must not cast.
        with torch.device("meta"):
            model = get_model(model_cfg)
        expected_state = model.state_dict()
        model.load_state_dict(state, strict=True, assign=True)
        for name, expected in expected_state.items():
            if state[name].dtype != expected.dtype:
                raise ValueError(
                    f"Wrong model tensor dtype for {name}: {state[name].dtype}, expected {expected.dtype}"
                )
        for name, expected in (("action_scale", 1.0), ("action_bias", 0.0)):
            if state[name].item() != expected:
                raise ValueError(
                    f"Wrong fixed action transform: {name} must equal {expected}"
                )
        parameters = dict(model.named_parameters())
        total = sum(p.numel() for p in parameters.values())
        critic = sum(
            p.numel() for name, p in parameters.items() if name.startswith("q_head.")
        )
        summary["parameters"] = {
            "total": total,
            "actor": total - critic,
            "critic": critic,
        }
    return summary


def validate_dcp_files(checkpoint):
    """Check metadata-referenced byte ranges before starting Ray or loading FSDP."""
    from torch.distributed.checkpoint import FileSystemReader

    for relative in (
        "actor/dcp_checkpoint",
        "actor/sac_components/alpha/dcp_checkpoint",
    ):
        directory = checkpoint / relative
        metadata = FileSystemReader(directory).read_metadata()
        if not metadata.storage_data:
            raise ValueError(f"Empty DCP checkpoint: {directory}")
        sizes = {}
        for item in metadata.storage_data.values():
            path = (directory / item.relative_path).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError(f"DCP data shard is outside its checkpoint: {path}")
            if path not in sizes:
                sizes[path] = path.stat().st_size
            if (
                item.offset < 0
                or item.length <= 0
                or item.offset + item.length > sizes[path]
            ):
                raise ValueError(f"Truncated or invalid DCP data shard: {path}")


def checkpoint_summary(checkpoint, model_cfg):
    step = checkpoint_step(checkpoint)
    validate_dcp_files(checkpoint)
    return {
        "final_step": step,
        "weights": weight_summary(checkpoint / WEIGHTS, model_cfg),
        "target_weights": weight_summary(
            checkpoint / "actor/sac_components/target_model/checkpoint_rank_0.pt",
            model_cfg,
        ),
    }


def last_metrics(output, expected_step=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = EventAccumulator(str(output / "tensorboard"), size_guidance={"scalars": 1})
    events.Reload()
    result = {}
    for tag in events.Tags()["scalars"]:
        last = events.Scalars(tag)[-1]
        if not math.isfinite(last.value):
            raise ValueError(f"Nonfinite RLinf metric: {tag}")
        result[tag] = {"step": last.step, "value": last.value}
    evaluation = {
        tag: value for tag, value in result.items() if tag.startswith("eval/")
    }
    if not evaluation:
        raise RuntimeError("RLinf produced no evaluation metrics")
    if not {"eval/success_once", "eval/num_trajectories"} <= evaluation.keys():
        raise RuntimeError("RLinf produced no complete PickCube evaluation metrics")
    if evaluation["eval/num_trajectories"]["value"] <= 0:
        raise RuntimeError("RLinf evaluation completed no trajectories")
    if not 0 <= evaluation["eval/success_once"]["value"] <= 1:
        raise ValueError("RLinf evaluation success rate is outside [0, 1]")
    if expected_step is not None and any(
        value["step"] != expected_step for value in evaluation.values()
    ):
        raise RuntimeError(
            f"RLinf evaluation metrics do not cover final log step {expected_step}"
        )
    return result


def execute(cfg, entry):
    """Own one local Ray runtime, including when another Ray cluster is running."""
    import ray

    original_init = ray.init
    previous_failure_handler = signal.getsignal(signal.SIGUSR1)
    entry_globals = None

    def worker_failed(signum, frame):
        # WorkerGroup prints the original exception before signalling the driver.
        # Further failed tasks may send the same signal during Ray teardown.
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        raise RuntimeError("RLinf worker failed; see the original error in worker.log")

    def init_local(*args, **kwargs):
        # Let RLinf enter its normal local-start/retry path instead of connecting
        # to another job. Converting its first 'auto' call directly to 'local'
        # would bypass the upstream retry for non-ConnectionError startup failures.
        if kwargs.get("address") == "auto":
            raise ConnectionError(
                "EmbodiedForge requires a separate local Ray instance"
            )
        kwargs["address"] = "local"
        # This adapter places one actor, rollout and env group on one GPU.
        # Advertising every host core makes Ray prestart many unused workers,
        # adding avoidable RAM pressure on shared training machines.
        kwargs["num_cpus"] = min(4, os.cpu_count() or 1)
        context = original_init(*args, **kwargs)
        os.environ["RAY_ADDRESS"] = context.address_info["gcs_address"]
        ray.init = original_init
        return context

    ray.init = init_local
    try:
        namespace = runpy.run_path(str(entry), run_name="_ef_rlinf_entry")
        main = namespace["main"].__wrapped__
        original_cluster = main.__globals__["Cluster"]
        entry_globals = main.__globals__

        def owned_cluster(*args, **kwargs):
            cluster = original_cluster(*args, **kwargs)
            # Replace only this entry point's post-start failure handler. Its
            # local Ray instance is already owned by the finally block below;
            # listing actors through the optional dashboard API is unnecessary.
            signal.signal(signal.SIGUSR1, worker_failed)
            return cluster

        entry_globals["Cluster"] = owned_cluster
        main(cfg)
    finally:
        ray.init = original_init
        if entry_globals is not None:
            entry_globals["Cluster"] = original_cluster
        active_error = sys.exc_info()[0] is not None
        signal.signal(signal.SIGUSR1, signal.SIG_IGN)
        try:
            ray.shutdown()
        except Exception as exc:
            if not active_error:
                raise
            print(f"Failed to shut down RLinf Ray runtime: {exc}", file=sys.stderr)
        finally:
            signal.signal(signal.SIGUSR1, previous_failure_handler)


def record_environment(output, *, export_onnx=False, evaluate_onnx=False):
    """Keep SDK diagnostics even when a selected Python cannot start RLinf."""
    versions, errors = {}, []
    packages = (
        "torch",
        "ray",
        "hydra-core",
        "omegaconf",
        "numpy",
        "mani_skill",
        "sapien",
        "tensorboard",
        "packaging",
    )
    if export_onnx:
        packages += ("onnx", "onnxruntime")
    elif evaluate_onnx:
        packages += ("onnxruntime",)
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            errors.append(f"missing {name}")
    # Read installed metadata without importing CUDA or starting Ray.
    if versions["packaging"]:
        from packaging.version import Version

        for name, minimum in (("torch", "2.5"), ("ray", "2.47")):
            if versions[name] and Version(versions[name]) < Version(minimum):
                errors.append(f"{name}>={minimum} required, found {versions[name]}")
        hydra = versions["hydra-core"]
        if hydra and Version(hydra) >= Version("1.4.0.dev8"):
            errors.append(f"hydra-core<1.4.0.dev8 required, found {hydra}")
    write_json(
        output / "environment.json",
        {
            "python": sys.version,
            "executable": sys.executable,
            "packages": versions,
            "errors": errors,
        },
    )
    if errors:
        raise RuntimeError(
            "Selected --python is not a compatible RLinf SDK: " + "; ".join(errors)
        )


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    repo, output = Path(request["repo"]), Path(request["output"])
    check_checkout(repo)
    evaluate_onnx = request.get("policy_format") == "onnx"
    record_environment(
        output,
        export_onnx=request["command"] == "export",
        evaluate_onnx=evaluate_onnx,
    )
    if request["checkpoint"] and inventory(output / "inputs") != request["input_files"]:
        raise ValueError("RLinf input snapshot differs from request.json")
    from omegaconf import OmegaConf

    cfg = compose_config(request)
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    if evaluate_onnx:
        from ._rlinf_onnx import evaluate

        result = evaluate(
            cfg, Path(request["checkpoint"]), request["input_files"][0]["sha256"]
        )
        write_json(output / "result.json", result)
        return
    if request["command"] in ("evaluate", "export"):
        weights = weight_summary(Path(request["checkpoint"]), cfg.actor.model)
    elif request["checkpoint"]:
        checkpoint = Path(request["checkpoint"])
        checkpoint_summary(checkpoint, cfg.actor.model)
    if request["command"] == "export":
        from ._rlinf_export import export_actor

        exported = export_actor(
            Path(request["checkpoint"]),
            output,
            cfg.actor.model,
            weights["sha256"],
            weight_storage=request.get("weight_storage", "float32"),
        )
        write_json(
            output / "result.json",
            {"status": "complete", "weights": weights, "onnx": exported},
        )
        return
    entry = repo / (
        "examples/embodiment/train_embodied_agent.py"
        if request["command"] == "train"
        else "evaluations/eval_embodied_agent.py"
    )
    # Short paths are required for Ray's Unix sockets. Ray shuts down before
    # temporary runtime files are removed; persistent training logs stay in output.
    with tempfile.TemporaryDirectory(prefix="ef-rlinf-") as runtime:
        os.environ["RAY_TMPDIR"] = runtime
        execute(cfg, entry)
    final = (
        request["start_step"] + request["iterations"]
        if request["command"] == "train"
        else 0
    )
    result = {
        "status": "complete",
        "metrics": last_metrics(output, final - 1 if final else 0),
    }
    if request["command"] == "train":
        checkpoint = output / CONFIG / "checkpoints" / f"global_step_{final}"
        result.update(checkpoint_summary(checkpoint, cfg.actor.model))
        files = inventory(checkpoint)
        result.update(
            checkpoint=str(checkpoint),
            checkpoint_files=files,
            checkpoint_bytes=sum(entry["bytes"] for entry in files),
        )
    else:
        result["weights"] = weights
    write_json(output / "result.json", result)


if __name__ == "__main__":
    main()
