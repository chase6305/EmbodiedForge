"""Evaluate the exported PickCube actor on CPU in the upstream GPU simulator."""

import hashlib
import re
from pathlib import Path
from time import perf_counter

from .rlinf import REVISION


def load_policy(path, expected_sha256):
    import onnxruntime as ort

    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise ValueError("RLinf ONNX policy changed before loading")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Load the verified bytes. An export is self-contained, without sidecar weights.
    session = ort.InferenceSession(payload, options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    for key, expected in (
        ("task", "PickCube-v1"),
        ("rlinf_revision", REVISION),
        ("control_mode", "pd_ee_delta_pos"),
        ("policy_mode", "eval: tanh(actor_mean), without exploration"),
        ("observation", "ManiSkill obs_mode=state, unchanged 42-element ordering"),
        ("action_range", "[-1, 1] normalized controller inputs"),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"Incompatible RLinf ONNX {key}: expected {expected}")
    if not re.fullmatch("[0-9a-f]{64}", metadata.get("checkpoint_sha256", "")):
        raise ValueError("RLinf ONNX is missing its source checkpoint SHA-256")
    for nodes, name, width in (
        (session.get_inputs(), "states", 42),
        (session.get_outputs(), "actions", 4),
    ):
        if (
            len(nodes) != 1
            or nodes[0].name != name
            or nodes[0].type != "tensor(float)"
            or len(nodes[0].shape) != 2
            or isinstance(nodes[0].shape[0], int)
            or nodes[0].shape[1] != width
        ):
            raise ValueError(f"RLinf ONNX requires float32 {name}[batch, {width}]")
    return session, {
        "path": str(path),
        "sha256": digest,
        "file_bytes": len(payload),
        "metadata": metadata,
        "provider": "CPUExecutionProvider",
    }


def evaluate(cfg, path, expected_sha256):
    import numpy as np
    import torch
    from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv
    from rlinf.utils.utils import seed_everything
    from torch.utils.tensorboard import SummaryWriter

    session, policy = load_policy(path, expected_sha256)
    env_cfg = cfg.env.eval
    if not (
        env_cfg.init_params.id == "PickCube-v1"
        and env_cfg.init_params.obs_mode == "state"
        and env_cfg.init_params.control_mode == "pd_ee_delta_pos"
        and env_cfg.auto_reset
        and env_cfg.ignore_terminations
        and env_cfg.max_episode_steps == env_cfg.max_steps_per_rollout_epoch == 50
    ):
        raise ValueError(
            "ONNX evaluation requires the fixed PickCube evaluation recipe"
        )
    torch.set_num_threads(1)
    seed_everything(env_cfg.seed)
    num_envs = env_cfg.total_num_envs
    steps = env_cfg.max_steps_per_rollout_epoch * env_cfg.rollout_epoch
    totals = dict.fromkeys(
        ("success_once", "success_at_end", "return", "episode_len", "reward"), 0.0
    )
    trajectories = 0
    inference_ms = []
    env = ManiskillEnv(
        env_cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset()
        for _ in range(steps):
            states = obs["states"].detach().cpu().numpy()
            if states.shape != (num_envs, 42) or not np.isfinite(states).all():
                raise ValueError("Invalid PickCube observation for ONNX policy")
            started = perf_counter()
            actions = session.run(["actions"], {"states": states})[0]
            inference_ms.append((perf_counter() - started) * 1000)
            if (
                actions.shape != (num_envs, 4)
                or not np.isfinite(actions).all()
                or np.any(np.abs(actions) > 1.0)
            ):
                raise ValueError("Invalid normalized action from RLinf ONNX policy")
            obs, rewards, terminated, truncated, info = env.step(actions)
            if not torch.isfinite(rewards).all():
                raise ValueError("Nonfinite PickCube evaluation reward")
            done = terminated | truncated
            count = int(done.sum())
            if count:
                episode = info["final_info"]["episode"]
                for key in totals:
                    values = episode[key][done].double()
                    if not torch.isfinite(values).all():
                        raise ValueError(f"Nonfinite PickCube episode metric: {key}")
                    totals[key] += float(values.sum())
                trajectories += count
    finally:
        env.env.close()
    expected = num_envs * env_cfg.rollout_epoch
    if trajectories != expected:
        raise RuntimeError(
            f"PickCube completed {trajectories} trajectories, expected {expected}"
        )
    metrics = {
        f"eval/{key}": {"step": 0, "value": total / trajectories}
        for key, total in totals.items()
    }
    metrics["eval/num_trajectories"] = {"step": 0, "value": trajectories}
    with SummaryWriter(Path(cfg.runner.logger.log_path) / "tensorboard") as writer:
        for key, item in metrics.items():
            writer.add_scalar(key, item["value"], item["step"])
    return {
        "status": "complete",
        "policy_driver": "onnx_cpu",
        "onnx": policy,
        "metrics": metrics,
        "action_observations": num_envs * steps,
        "inference": {
            "scope": "onnxruntime_session_run_cpu",
            "batch_size": num_envs,
            "calls": len(inference_ms),
            "includes_first_call": True,
            "mean_batch_ms": float(np.mean(inference_ms)),
            "p50_batch_ms": float(np.percentile(inference_ms, 50)),
            "p95_batch_ms": float(np.percentile(inference_ms, 95)),
        },
        "success_once_count": int(totals["success_once"]),
        "success_at_end_count": int(totals["success_at_end"]),
    }
