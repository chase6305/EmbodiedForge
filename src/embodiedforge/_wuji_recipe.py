"""Wuji owner recipe and artifact validation in its dedicated environment."""

from __future__ import annotations

import math
import sys
from pathlib import Path


def finite_tensors(value):
    import torch

    count = 0
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite checkpoint tensor")
        count = 1
    elif isinstance(value, dict):
        count = sum(finite_tensors(child) for child in value.values())
    elif isinstance(value, (tuple, list)):
        count = sum(finite_tensors(child) for child in value)
    return count


def verify_training(request):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    updates = request["updates"]
    start = request.get("start_iteration", 0)
    checkpoints = list(Path("logs").rglob(f"model_{start + updates - 1}.pt"))
    if len(checkpoints) != 1:
        raise ValueError("Expected exactly one final Wuji checkpoint")
    checkpoint = checkpoints[0]
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if data["iter"] != start + updates - 1:
        raise ValueError("Wuji checkpoint iteration mismatch")
    actor, critic = data["actor_state_dict"], data["critic_state_dict"]
    if actor["mlp.0.weight"].shape[1] != 207 or actor["mlp.6.weight"].shape[0] != 40:
        raise ValueError("Wuji actor must map 207 observations to 20 Gaussian actions")
    if critic["mlp.0.weight"].shape[1] != 413:
        raise ValueError("Wuji critic must consume 413 privileged observations")
    count = finite_tensors(data)
    events = EventAccumulator(
        str(checkpoint.parent), size_guidance={"scalars": 0}
    ).Reload()
    metrics = {}
    scalar_count = 0
    for name in events.Tags()["scalars"]:
        history = events.Scalars(name)
        if any(not math.isfinite(item.value) for item in history):
            raise ValueError(f"Non-finite scalar: {name}")
        scalar_count += len(history)
        metrics[name] = history[-1].value
    for name in ("Loss/value", "Loss/surrogate"):
        if name not in metrics or [item.step for item in events.Scalars(name)] != list(
            range(start, start + updates)
        ):
            raise ValueError(f"Incomplete Wuji update history: {name}")
    if not data.get("optimizer_state_dict", {}).get("state"):
        raise ValueError("Missing optimizer state")
    if not data.get("infos", {}).get("uni_rl_training_state"):
        raise ValueError("Missing Wuji curriculum training state")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": data["iter"],
        "completed_updates": updates,
        "finite_tensor_count": count,
        "scalar_samples": scalar_count,
        "latest_metrics": metrics,
        "action_dim": 20,
        "policy_observation_dim": 207,
        "critic_observation_dim": 413,
        "behavior_validated": False,
    }


def run(request):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Wuji requires CUDA; no CPU fallback")
    torch.set_num_threads(request["threads"])
    if request["command"] == "train":
        from wuji_unilab.cli import train_main

        sys.argv = [
            "wuji-train",
            "--task",
            request["upstream_task"],
            f"algo.num_envs={request['num_envs']}",
            f"algo.max_iterations={request['updates']}",
            f"algo.num_steps_per_env={request['horizon']}",
            f"algo.seed={request['seed']}",
        ]
        if request.get("checkpoint"):
            sys.argv += ["--checkpoint-file", request["checkpoint"]]
        train_main()
        return verify_training(request)
    from wuji_unilab import eval as owner

    from embodiedforge._wuji_diagnostics import TrialDiagnostics

    diagnostics = TrialDiagnostics()
    policy_kind = request.get("policy", "trained")
    original_wrapper = owner.WujiWrapper
    original_runner = owner.TrainingStateOnPolicyRunner
    original_refresh = owner._refresh_policy_observation
    original_session = owner.SnapshotPlaybackSession

    class BoundedSession(original_session):
        def __init__(self, env, **kwargs):
            super().__init__(env, width=640, height=368, num_processes=1, **kwargs)

    class DiagnosticWrapper(original_wrapper):
        def step(self, actions):
            result = super().step(actions)
            state = owner.task(self.env)
            position, quaternion = state.cube_tag()
            diagnostics.update(
                position[0], quaternion[0], actions.detach().cpu().numpy()[0]
            )
            return result

    class DiagnosticRunner(original_runner):
        def get_inference_policy(self, *args, **kwargs):
            policy = super().get_inference_policy(*args, **kwargs)

            def act(observation):
                action = policy(observation)
                if not torch.isfinite(action).all():
                    raise ValueError("Non-finite Wuji policy action")
                return torch.zeros_like(action) if policy_kind == "zero" else action

            return act

    def refresh(env, wrapper, *, reset_history):
        result = original_refresh(env, wrapper, reset_history=reset_history)
        state = owner.task(env)
        position, quaternion = state.cube_tag()
        diagnostics.start(position[0], quaternion[0], state.goal[0])
        return result

    # The upstream --seed seeds goal sampling only. Also seed the environment's
    # configured reset stream so the managed seed describes the whole trial.
    original_compose = owner.compose_task

    def compose(task, overrides):
        return original_compose(
            task, overrides=[*overrides, f"algo.seed={request['seed']}"]
        )

    owner.compose_task = compose
    owner.WujiWrapper = DiagnosticWrapper
    owner.TrainingStateOnPolicyRunner = DiagnosticRunner
    owner._refresh_policy_observation = refresh
    owner.SnapshotPlaybackSession = BoundedSession
    try:
        args = owner._parser().parse_args(
            [
                "--task",
                request["upstream_task"],
                "--checkpoint-file",
                request["checkpoint"],
                "--num-trials",
                str(request["num_trials"]),
                "--seed",
                str(request["seed"]),
                "--trial-timeout",
                str(request["steps"] * 0.05),
                "--output",
                "evaluation.json",
            ]
        )
        args.record = request.get("record_video", False)
        args.video_output = Path("evaluation.mp4").resolve()
        result = owner.run(args)
    finally:
        owner.compose_task = original_compose
        owner.WujiWrapper = original_wrapper
        owner.TrainingStateOnPolicyRunner = original_runner
        owner._refresh_policy_observation = original_refresh
        owner.SnapshotPlaybackSession = original_session
    if result["num_trials"] != request["num_trials"]:
        raise ValueError("Incomplete Wuji evaluation")
    for trial in result["trials"]:
        for key in ("final_orientation_error_rad", "min_orientation_error_rad"):
            if not math.isfinite(trial[key]):
                raise ValueError("Non-finite Wuji evaluation")
    result["protocol"]["environment_seed"] = request["seed"]
    result["protocol"]["policy"] = policy_kind
    result["diagnostics"] = diagnostics.report()
    if len(result["diagnostics"]["trials"]) != request["num_trials"]:
        raise ValueError("Diagnostic trial count mismatch")
    result["acceptance"] = None
    from embodiedforge.recipes import write_json

    write_json(Path("evaluation.json"), result)
    return result
