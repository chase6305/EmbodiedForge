"""Optional LeRobot ACT bridge for the core VectorEnv's recorded episodes.

Run in a separate LeRobot environment; importing this module only requires NumPy.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np

from .core import Config
from .data import read_episodes

DATA_MANIFEST = "embodiedforge.json"
RUN_MANIFEST = "act-run.json"


def _attempt_paths(run):
    """Keep startup metadata outside the directory LeRobot requires to be new."""
    return (
        run.parent / f".{run.name}.act-run.json",
        run.parent / f".{run.name}.act.lock",
    )


@contextmanager
def _run_lock(run):
    """Linux advisory lock; inherited by the trainer until both processes exit.

    Close rather than explicitly unlock: an orphan trainer may still own the
    same open file description after its launcher has exited.
    """
    import fcntl

    _, lock_path = _attempt_paths(run)
    run.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"ACT run is still active: {run}") from exc
        yield lock


def _resume_report(run):
    startup, lock_path = _attempt_paths(run)
    manifest = run / RUN_MANIFEST
    path = manifest if manifest.is_file() else startup
    report = json.loads(path.read_text())
    if report.get("status") == "running" and not lock_path.exists():
        raise ValueError(
            "Interrupted run is missing its lock file; cannot verify trainer exit"
        )
    with _run_lock(run):
        # Re-read after acquiring the lock: the launcher may have just finalized.
        report = json.loads((manifest if manifest.is_file() else startup).read_text())
        if report.get("schema_version") != 1 or report.get("status") not in (
            "complete",
            "failed",
            "running",
        ):
            raise ValueError("Resume requires a managed ACT run")
        if report["status"] == "running":
            report["status"] = "interrupted"
        return report


def _write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _require_lerobot():
    try:
        import lerobot  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ACT requires a separate Python 3.12+ environment with LeRobot[training]. "
            "See docs/act.md; pass --python /path/to/environment/bin/python."
        ) from exc


def runtime_report():
    """Identify imported implementations and whether they live in the installed SDK.

    This checks entry points, not every transitive import or checkpoint quality.
    Local installation URLs describe build inputs, not runtime dependencies.
    """
    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.scripts import lerobot_train

    from ._training_runtime import component

    package = distribution("lerobot")
    origin = json.loads(package.read_text("direct_url.json") or "null")
    installed_root = Path(package.locate_file("lerobot")).resolve()
    components = {
        "policy": component(ACTPolicy),
        "trainer": component(lerobot_train),
        "dataset": component(LeRobotDataset),
        "processors": component(make_pre_post_processors),
    }
    installed = all(
        item["file"] is not None and Path(item["file"]).is_relative_to(installed_root)
        for item in components.values()
    )
    return {
        "schema_version": 1,
        "ok": True,
        "integration": "external_package_adapter",
        "implementation": "lerobot",
        "native_act_implementation": False,
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "scope": "imported_entry_points_not_full_dependency_graph",
        "installation": {
            "version": package.version,
            "mode": "installed_package" if installed else "external_source",
            "editable": bool((origin or {}).get("dir_info", {}).get("editable")),
            "build_origin": origin,
            "package_directory": str(installed_root),
        },
        "components": components,
        "adapter": component(ACTAdapter),
    }


def doctor():
    """Return actionable diagnostics even when the optional SDK is absent."""
    try:
        return runtime_report()
    except (ImportError, RuntimeError) as exc:
        return {
            "ok": False,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "implementation": "lerobot",
            "native_act_implementation": False,
            "error": str(exc),
            "hint": "Use act --python /path/to/act-environment/bin/python doctor; see docs/act.md",
        }


def export_dataset(source: Path, output: Path, repo_id: str, *, images=False):
    """Export complete episodes, preserving obs[t]/action[t] alignment.

    Publication is atomic. PNG image features avoid video encoder dependencies.
    Low-rate camera frames remain sample-and-hold at the control frequency.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(repo_id.split("/")) != 2 or not all(repo_id.split("/")):
        raise ValueError("repo_id must have the form namespace/dataset")
    manifest = json.loads((source / "run.json").read_text())
    if manifest.get("status") != "closed":
        raise ValueError("Export requires a closed, successful recording run")
    config = Config(**manifest["config"])
    state_key = "observation.state"
    # This LeRobot revision still reads OBS_STATE unconditionally inside ACT.
    # Nonvisual ACT additionally requires ENV for feature validation/conditioning.
    env_state_key = None if images else "observation.environment_state"
    first = next(read_episodes(source), None)
    if first is None:
        raise ValueError("No complete episodes to export")
    features = {
        state_key: {
            "dtype": "float32",
            "shape": first["obs/proprio"].shape[1:],
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": first["action"].shape[1:],
            "names": None,
        },
    }
    if env_state_key:
        features[env_state_key] = dict(features[state_key])
    camera_keys = []
    if images:
        rgb = first.get("obs/rgb")
        if rgb is None or rgb.ndim != 5 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("--images requires recorded uint8 RGB [T,C,H,W,3]")
        for index in range(rgb.shape[1]):
            key = f"observation.images.camera_{index}"
            camera_keys.append(key)
            features[key] = {
                "dtype": "image",
                "shape": rgb.shape[2:],
                "names": ["height", "width", "channels"],
            }
    # Validate every episode before creating the destination or importing the SDK.
    episodes = frames = 0
    for episode in read_episodes(source):
        if (
            episode["obs/proprio"].shape[1:] != features[state_key]["shape"]
            or episode["action"].shape[1:] != features["action"]["shape"]
        ):
            raise ValueError("Episode feature dimensions changed")
        if images and (
            "obs/rgb" not in episode
            or episode["obs/rgb"].shape[1:] != first["obs/rgb"].shape[1:]
        ):
            raise ValueError("Episode camera layout changed")
        episodes += 1
        frames += len(episode["action"])
    _require_lerobot()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        root = staging / "dataset"
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=config.control_hz,
            root=root,
            features=features,
            robot_type=f"embodiedforge_{config.task}",
            use_videos=False,
        )
        try:
            for episode in read_episodes(source):
                for t, action in enumerate(episode["action"]):
                    frame = {
                        state_key: episode["obs/proprio"][t].astype(np.float32),
                        "action": action.astype(np.float32),
                        "task": manifest["env_spec"]["instruction"],
                    }
                    if env_state_key:
                        frame[env_state_key] = frame[state_key].copy()
                    for index, key in enumerate(camera_keys):
                        frame[key] = episode["obs/rgb"][t, index].copy()
                    dataset.add_frame(frame)
                dataset.save_episode()
        finally:
            dataset.finalize()
        report = {
            "schema_version": 1,
            "repo_id": repo_id,
            "source": str(source),
            "source_run_id": manifest["run_id"],
            "config": config.to_dict(),
            "env_spec": manifest["env_spec"],
            "state_key": state_key,
            "env_state_key": env_state_key,
            "camera_keys": camera_keys,
            "episodes": episodes,
            "frames": frames,
            "lerobot_version": version("lerobot"),
            "camera_sampling": "sample_and_hold_at_control_hz",
        }
        _write_json(root / DATA_MANIFEST, report)
        if output.exists():
            raise FileExistsError(output)
        root.rename(output)
    finally:
        shutil.rmtree(staging)
    return report


def train(args):
    """Delegate optimization/checkpointing to the installed LeRobot trainer."""
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    dataset = args.dataset.resolve()
    data = json.loads((dataset / DATA_MANIFEST).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported ACT dataset manifest")
    if not 1 <= args.action_steps <= args.chunk_size:
        raise ValueError("Require 1 <= action-steps <= chunk-size")
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={data['repo_id']}",
        f"--dataset.root={dataset}",
        "--policy.type=act",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        f"--output_dir={output}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        f"--seed={args.seed}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.action_steps}",
        "--num_workers=0",
        "--env_eval_freq=0",
        f"--save_freq={getattr(args, 'save_freq', 1000)}",
        f"--log_freq={min(args.steps, 100)}",
    ]
    learning_rate = getattr(args, "learning_rate", None)
    if learning_rate is not None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning-rate must be finite and positive")
        command.append(f"--policy.optimizer_lr={learning_rate}")
    if args.small_model:
        command += [
            "--policy.dim_model=64",
            "--policy.n_heads=4",
            "--policy.dim_feedforward=128",
            "--policy.n_encoder_layers=1",
            "--policy.n_vae_encoder_layers=1",
            "--policy.latent_dim=8",
            "--policy.pretrained_backbone_weights=null",
        ]
    from ._act_dataset import fingerprint_dataset

    return _execute_training(
        output, data, command, dataset_fingerprint=fingerprint_dataset(dataset)
    )


def resume(args):
    """Continue a saved optimizer/RNG state into a new run directory."""
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    run = args.run.resolve()
    report = _resume_report(run)
    from ._act_checkpoints import select_checkpoint

    checkpoint, selection = select_checkpoint(
        run, getattr(args, "checkpoint", "last"), for_resume=True
    )
    model = checkpoint / "pretrained_model"
    saved = json.loads((model / "train_config.json").read_text())
    step = selection["step"]
    if type(step) is not int or step < 0 or args.steps <= step:
        raise ValueError(
            f"--steps must exceed saved step {step}; it is the total target"
        )
    dataset = Path(saved["dataset"]["root"])
    current_data = json.loads((dataset / DATA_MANIFEST).read_text())
    if current_data != report["dataset"]:
        raise ValueError("Dataset manifest changed since the source training run")
    from ._act_dataset import fingerprint_dataset, verify_dataset_fingerprint

    fingerprint = fingerprint_dataset(dataset)
    verification = verify_dataset_fingerprint(
        report.get("dataset_fingerprint"), fingerprint
    )
    if verification["status"] == "unverified_legacy":
        print(
            "Source run has no dataset fingerprint; historical data identity cannot "
            "be verified. Recording the current dataset for subsequent resumes.",
            file=sys.stderr,
        )
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--config_path={model / 'train_config.json'}",
        "--resume=true",
        f"--output_dir={output}",
        f"--steps={args.steps}",
        "--policy.push_to_hub=false",
        "--save_checkpoint_to_hub=false",
        "--wandb.enable=false",
        "--job.target=null",
        "--policy.pretrained_backbone_weights=null",
        f"--log_freq={min(args.steps - step, 100)}",
    ]
    if getattr(args, "device", None) is not None:
        command.append(f"--policy.device={args.device}")
    return _execute_training(
        output,
        current_data,
        command,
        dataset_fingerprint=fingerprint,
        resumed_from={
            "run": str(run),
            "checkpoint": str(checkpoint),
            "step": step,
            "source_status": report["status"],
            "selection": selection,
            "dataset_verification": verification,
            "checkpoint_sha256": hashlib.sha256(
                (model / "model.safetensors").read_bytes()
            ).hexdigest(),
        },
    )


def _execute_training(
    output, data, command, *, resumed_from=None, dataset_fingerprint=None
):
    """Record each training attempt separately; never overwrite the source run."""
    _require_lerobot()
    from lerobot.policies.act.modeling_act import ACTPolicy

    from ._training_runtime import component

    report = {
        "schema_version": 1,
        "dataset": data,
        "dataset_fingerprint": dataset_fingerprint,
        "command": command,
        "python": sys.executable,
        "policy": component(ACTPolicy),
        "runtime": runtime_report(),
        "versions": {
            name: version(name)
            for name in (
                "lerobot",
                "torch",
                "torchvision",
                "numpy",
                "datasets",
                "multiprocess",
            )
        },
        "status": "running",
        "resumed_from": resumed_from,
    }
    with _run_lock(output) as lock:
        startup, _ = _attempt_paths(output)
        if output.exists() or startup.exists():
            raise FileExistsError(
                f"ACT output or prior startup record already exists: {output}"
            )
        _write_json(startup, report)
        print(json.dumps({"command": command}), flush=True)
        try:
            subprocess.run(command, check=True, pass_fds=(lock.fileno(),))
            report["status"] = "complete"
        except BaseException:
            report["status"] = "failed"
            raise
        finally:
            _write_json(startup, report)
            if output.is_dir():
                _write_json(output / RUN_MANIFEST, report)
    return {"output": str(output), "status": report["status"]}


class ACTAdapter:
    """Stateless chunk prediction; ActionExecutor owns per-environment cursors.

    Temporal ensembling is deliberately rejected: it needs per-environment state.
    Predictions are explicitly clipped to the task's physical action bounds.
    """

    def __init__(self, checkpoint, dataset_manifest, *, device="cpu"):
        _require_lerobot()
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from ._act_checkpoints import inference_fingerprint

        self.metadata = dataset_manifest
        self.checkpoint_fingerprint = inference_fingerprint(checkpoint)
        config = ACTConfig.from_pretrained(str(checkpoint))
        if config.temporal_ensemble_coeff is not None:
            raise ValueError("ACTAdapter does not support temporal ensembling")
        config.device = device
        # Loading saved weights must not download an ImageNet initialization.
        config.pretrained_backbone_weights = None
        self.policy = ACTPolicy.from_pretrained(str(checkpoint), config=config)
        self.policy.to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.horizon = config.n_action_steps
        expected = {self.metadata["state_key"], *self.metadata["camera_keys"]}
        if self.metadata.get("env_state_key"):
            expected.add(self.metadata["env_state_key"])
        if set(config.input_features) != expected:
            raise ValueError("Checkpoint input features do not match dataset mapping")
        for key in (self.metadata["state_key"], self.metadata.get("env_state_key")):
            if key and tuple(config.input_features[key].shape) != tuple(
                self.metadata["env_spec"]["proprio_shape"]
            ):
                raise ValueError("Checkpoint state shape does not match environment")
        size = self.metadata["config"]["image_size"]
        for key in self.metadata["camera_keys"]:
            if tuple(config.input_features[key].shape) != (3, size, size):
                raise ValueError("Checkpoint image shape does not match environment")
        if tuple(config.output_features["action"].shape) != tuple(
            self.metadata["env_spec"]["action_shape"]
        ):
            raise ValueError("Checkpoint action shape does not match environment")
        if inference_fingerprint(checkpoint) != self.checkpoint_fingerprint:
            raise ValueError("Checkpoint changed while loading ACT inference files")
        self.checkpoint_sha256 = next(
            item["sha256"]
            for item in self.checkpoint_fingerprint["files"]
            if item["path"] == "model.safetensors"
        )
        self.clipped_values = 0
        self.total_values = 0

    def inference_settings(self):
        """Describe effective overrides and numerical settings; not a replay guarantee."""
        import torch

        config = self.policy.config
        return {
            "device": str(next(self.policy.parameters()).device),
            "dtype": str(next(self.policy.parameters()).dtype),
            "chunk_size": config.chunk_size,
            "action_steps": self.horizon,
            "temporal_ensemble_coeff": config.temporal_ensemble_coeff,
            "pretrained_backbone_weights": config.pretrained_backbone_weights,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        }

    def act(self, observation):
        import torch

        batch = {
            self.metadata["state_key"]: torch.from_numpy(
                np.asarray(observation["proprio"], dtype=np.float32)
            ),
        }
        if self.metadata.get("env_state_key"):
            batch[self.metadata["env_state_key"]] = batch[
                self.metadata["state_key"]
            ].clone()
        for index, key in enumerate(self.metadata["camera_keys"]):
            rgb = observation["rgb"][:, index]
            batch[key] = torch.from_numpy(rgb.copy()).permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            actions = self.policy.predict_action_chunk(self.preprocess(batch))
            actions = self.postprocess(actions)[:, : self.horizon].cpu().numpy()
        if not np.isfinite(actions).all():
            raise ValueError("ACT produced non-finite actions")
        low, high = self.metadata["env_spec"]["action_range"]
        self.clipped_values += int(((actions < low) | (actions > high)).sum())
        self.total_values += actions.size
        return np.clip(actions, low, high).astype(np.float32)


def _evaluation_checkpoint(run, selector="last"):
    from ._act_checkpoints import select_checkpoint

    run = run.resolve()
    report = _resume_report(run)
    if report.get("status") != "complete" and selector == "last":
        raise ValueError("Evaluation requires a completed ACT training run")
    data = report["dataset"]
    checkpoint, selection = select_checkpoint(run, selector)
    selection["source_status"] = report["status"]
    return data, checkpoint / "pretrained_model", selection


def checkpoints(args):
    """List structural checkpoint readiness in a stopped managed run."""
    from ._act_checkpoints import checkpoint_inventory

    run = args.run.resolve()
    report = _resume_report(run)
    return {
        "run": str(run),
        "status": report["status"],
        "checkpoints": checkpoint_inventory(run),
    }


def bundle(args):
    """Export a selected stopped-run checkpoint for portable local inference."""
    from ._act_bundle import write_bundle

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    data, checkpoint, selection = _evaluation_checkpoint(
        args.run, getattr(args, "checkpoint", "last")
    )
    return write_bundle(
        checkpoint,
        data,
        {
            "run": str(args.run.resolve()),
            "checkpoint": str(checkpoint),
            "selection": selection,
        },
        args.output,
    )


def _evaluation_policy(args):
    root = getattr(args, "bundle", None)
    expected = None
    if root is not None:
        from ._act_bundle import read_bundle

        if getattr(args, "checkpoint", "last") != "last":
            raise ValueError("--checkpoint applies only to --run, not --bundle")
        report, checkpoint, manifest_hash = read_bundle(root)
        data = report["dataset"]
        expected = report["checkpoint_fingerprint"]
        selection = {
            "requested": "bundle",
            "bundle": str(root.resolve()),
            "manifest_sha256": manifest_hash,
            "source": report.get("source"),
        }
    else:
        data, checkpoint, selection = _evaluation_checkpoint(
            args.run, getattr(args, "checkpoint", "last")
        )
    policy = ACTAdapter(checkpoint, data, device=args.device)
    if expected is not None and policy.checkpoint_fingerprint != expected:
        raise ValueError("ACT inference bundle changed while loading")
    return data, checkpoint, selection, policy


def _validate_evaluation_environment(env, data):
    spec = json.loads(json.dumps(env.spec))
    for key in (
        "task",
        "scene",
        "action_shape",
        "action_range",
        "action_units",
        "proprio_shape",
        "control_dt",
    ):
        if spec[key] != data["env_spec"][key]:
            raise ValueError(f"Environment contract mismatch: {key}")


def evaluate(args):
    """Evaluate the selected checkpoint against its recorded simulation contract."""
    from .env import VectorEnv
    from .rollout import run_rollout

    output = getattr(args, "output", None)
    if output is not None and output.exists():
        raise FileExistsError(output)
    data, checkpoint, selection, policy = _evaluation_policy(args)
    config = replace(Config(**data["config"]), num_envs=args.num_envs, seed=args.seed)
    with VectorEnv(config) as env:
        _validate_evaluation_environment(env, data)
        summary = run_rollout(
            env, policy, steps=args.steps, chunk_horizon=policy.horizon
        )
    result = {
        **asdict(summary),
        "seed": args.seed,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "environment_config": json.loads(json.dumps(config.to_dict())),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_selection": selection,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "checkpoint_fingerprint": policy.checkpoint_fingerprint,
        "inference_settings": policy.inference_settings(),
        "clipped_action_fraction": policy.clipped_values / max(policy.total_values, 1),
        "runtime": runtime_report(),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, result)
    return result


def benchmark(args):
    """Compare ACT and the task expert on paired, complete episodes."""
    from .env import VectorEnv
    from .rollout import DemonstrationPolicy, run_episode_batch

    seeds = args.seeds
    if (
        not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Benchmark seeds must be nonnegative and unique")
    output = getattr(args, "output", None)
    if output is not None and output.exists():
        raise FileExistsError(output)
    data, checkpoint, selection, policy = _evaluation_policy(args)
    results = []
    for seed in seeds:
        row = {"seed": seed}
        config = replace(Config(**data["config"]), num_envs=args.num_envs, seed=seed)
        row["environment_config"] = json.loads(json.dumps(config.to_dict()))
        for name in ("act", "expert"):
            with VectorEnv(config) as env:
                _validate_evaluation_environment(env, data)
                if not callable(getattr(env.task, "expert_action", None)):
                    raise ValueError("Benchmark requires a task-provided expert_action")
                selected = (
                    policy
                    if name == "act"
                    else DemonstrationPolicy(env.task.expert_action)
                )
                batch = run_episode_batch(
                    env,
                    selected,
                    chunk_horizon=policy.horizon if name == "act" else 1,
                )
            row[name] = {
                "episodes": len(batch.success),
                "successes": int(batch.success.sum()),
                "success_rate": float(batch.success.mean()),
                "mean_episode_return": float(batch.returns.mean()),
                "mean_episode_length": float(batch.lengths.mean()),
                "initial_observation_sha256": batch.initial_observation_sha256,
                "episode_success": batch.success.tolist(),
                "episode_return": batch.returns.tolist(),
                "episode_length": batch.lengths.tolist(),
            }
        if (
            row["act"]["initial_observation_sha256"]
            != row["expert"]["initial_observation_sha256"]
        ):
            raise ValueError("ACT and expert initial observations differ")
        results.append(row)
    summary = {}
    for name in ("act", "expert"):
        episodes = sum(row[name]["episodes"] for row in results)
        successes = sum(row[name]["successes"] for row in results)
        summary[name] = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "mean_episode_return": float(
                np.mean(
                    [value for row in results for value in row[name]["episode_return"]]
                )
            ),
        }
    result = {
        "schema_version": 1,
        "mode": "paired_single_episode_per_environment",
        "seeds": seeds,
        "num_envs": args.num_envs,
        "max_steps": config.max_steps,
        "summary": summary,
        "per_seed": results,
        "training_environment_config": data["config"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_selection": selection,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "checkpoint_fingerprint": policy.checkpoint_fingerprint,
        "inference_settings": policy.inference_settings(),
        "clipped_predicted_action_fraction": policy.clipped_values
        / max(policy.total_values, 1),
        "runtime": runtime_report(),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, result)
    return result


def compare(args):
    """Compare two saved benchmark reports in the core environment."""
    from ._act_compare import compare_reports

    output = getattr(args, "output", None)
    if output is not None and output.exists():
        raise FileExistsError(output)
    result = compare_reports(args.baseline, args.candidate)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, result)
    return result


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python", type=Path, help="Run in an isolated LeRobot environment"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Show ACT implementation and loaded package paths"
    )
    listing = commands.add_parser(
        "checkpoints", help="List checkpoint readiness in a stopped run"
    )
    listing.add_argument("--run", type=Path, required=True)
    comparing = commands.add_parser(
        "compare", help="Compare saved paired benchmarks without loading LeRobot"
    )
    comparing.add_argument("--baseline", type=Path, required=True)
    comparing.add_argument("--candidate", type=Path, required=True)
    comparing.add_argument("--output", type=Path)
    bundling = commands.add_parser("bundle", help="Export a portable inference bundle")
    bundling.add_argument("--run", type=Path, required=True)
    bundling.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export", help="Export core episode recordings")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--repo-id", required=True)
    export.add_argument("--images", action="store_true")
    training = commands.add_parser("train", help="Train ACT with LeRobot")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--steps", type=_positive, default=10000)
    training.add_argument("--batch-size", type=_positive, default=8)
    training.add_argument("--chunk-size", type=_positive, default=20)
    training.add_argument("--action-steps", type=_positive, default=5)
    training.add_argument(
        "--learning-rate", type=_positive_float, help="ACT optimizer learning rate"
    )
    training.add_argument(
        "--save-freq",
        type=_nonnegative,
        default=1000,
        help="Checkpoint interval in steps; 0 saves only the final checkpoint",
    )
    training.add_argument(
        "--small-model", action="store_true", help="Small model for smoke tests"
    )
    resuming = commands.add_parser(
        "resume", help="Resume optimizer/RNG state into a new run"
    )
    resuming.add_argument("--run", type=Path, required=True)
    resuming.add_argument("--output", type=Path, required=True)
    resuming.add_argument(
        "--steps",
        type=_positive,
        required=True,
        help="Total target steps, including saved steps",
    )
    resuming.add_argument(
        "--device", choices=("cpu", "cuda"), help="Default: original training device"
    )
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate in the original core task"
    )
    evaluation.add_argument(
        "--output", type=Path, help="Save metrics to a new JSON file"
    )
    evaluation.add_argument("--steps", type=_positive, default=300)
    evaluation.add_argument("--num-envs", type=_positive, default=16)
    comparison = commands.add_parser(
        "benchmark", help="Paired ACT/expert evaluation with fixed episodes"
    )
    comparison.add_argument("--output", type=Path)
    comparison.add_argument("--num-envs", type=_positive, default=32)
    comparison.add_argument(
        "--seeds", type=_nonnegative, nargs="+", default=[2001, 2002, 2003]
    )
    comparison.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    for command in (evaluation, comparison):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--run", type=Path)
        source.add_argument("--bundle", type=Path, help="Portable ACT inference bundle")
    for command in (resuming, evaluation, comparison, bundling):
        command.add_argument(
            "--checkpoint",
            default="last",
            help="last (strict default), latest usable, or a saved step number",
        )
    for command in (training, evaluation):
        command.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        command.add_argument("--seed", type=int, default=1001)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(arguments)
    if getattr(args, "seed", 0) < 0:
        parser.error("--seed must be nonnegative")
    if args.python:
        # Keep the environment's interpreter path: resolving a venv symlink loses it.
        executable = str(args.python.absolute())
        forwarded = arguments.copy()
        for index, argument in enumerate(forwarded):
            if argument == "--python":
                del forwarded[index : index + 2]
                break
            if argument.startswith("--python="):
                del forwarded[index]
                break
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = (
            source_root + os.pathsep + environment.get("PYTHONPATH", "")
        )
        result = subprocess.run(
            [executable, "-m", "embodiedforge.act", *forwarded],
            env=environment,
        )
        raise SystemExit(result.returncode)
    if args.command == "doctor":
        result = doctor()
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["ok"] else 1)
    if args.command == "export":
        result = export_dataset(
            args.source, args.output, args.repo_id, images=args.images
        )
    elif args.command == "train":
        result = train(args)
    elif args.command == "resume":
        result = resume(args)
    elif args.command == "benchmark":
        result = benchmark(args)
    elif args.command == "checkpoints":
        result = checkpoints(args)
    elif args.command == "compare":
        result = compare(args)
    elif args.command == "bundle":
        result = bundle(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
