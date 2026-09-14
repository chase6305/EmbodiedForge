"""Diagnostics executed only by the isolated upstream Python interpreter."""

import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import threading
from contextlib import ExitStack, contextmanager
from pathlib import Path


def check() -> dict:
    import torch

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Microduck requires Python 3.12")
    expected = {"mjlab": "1.3.0", "warp-lang": "1.12.0", "torch": "2.9.1"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    for name, version in versions.items():
        if version.split("+")[0] != expected[name]:
            raise RuntimeError(f"Unexpected {name} version: {version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Microduck environment")
    value = torch.ones((16, 16), device="cuda")
    if not torch.isfinite(value @ value).all().item():
        raise RuntimeError("CUDA matrix multiplication produced nonfinite values")
    return {
        "python": sys.version,
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cuda_check": "passed",
        "installed_packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
    }


def onnx_session(path: Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or inputs[0].shape != [1, 61]:
        raise RuntimeError(f"Expected one 61D actor input, got {inputs}")
    if len(outputs) != 1 or outputs[0].shape != [1, 14]:
        raise RuntimeError(f"Expected one 14D action output, got {outputs}")
    if inputs[0].type != "tensor(float)" or outputs[0].type != "tensor(float)":
        raise RuntimeError("Expected float32 ONNX input and output")
    return session


def validate_onnx(path: Path) -> dict:
    import numpy as np

    session = onnx_session(path)
    inputs = session.get_inputs()
    rng = np.random.default_rng(0)
    for observation in (
        np.zeros((1, 61), dtype=np.float32),
        rng.normal(0, 0.1, size=(1, 61)).astype(np.float32),
    ):
        actions = session.run(None, {inputs[0].name: observation})[0]
        if actions.shape != (1, 14) or not np.isfinite(actions).all():
            raise RuntimeError("ONNX inference produced invalid actions")
    return {
        "onnx": str(path),
        "actor_dim": 61,
        "action_dim": 14,
        "finite_inference": True,
    }


def compare_policy_actions(reference, candidate) -> float:
    """Compare raw deterministic actions, before the environment clips/scales them."""
    import numpy as np

    if (
        reference.shape != (1, 14)
        or candidate.shape != (1, 14)
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise RuntimeError("Parity check requires finite [1, 14] actions")
    error = np.abs(candidate - reference)
    if not np.all(error <= 1e-4 + 1e-4 * np.abs(reference)):
        raise RuntimeError(
            f"ONNX action mismatch: max absolute error={error.max():.8g}"
        )
    return float(error.max())


def configure_evaluation(cfg, velocity, no_pushes: bool) -> None:
    """Set command ranges before managers deep-copy the upstream configuration."""
    if no_pushes:
        cfg.events.pop("push_robot", None)
    if velocity is None:
        return
    if len(velocity) != 3 or not all(map(math.isfinite, velocity)):
        raise ValueError("Fixed velocity must contain three finite values")
    twist = cfg.commands["twist"]
    for axis, value in zip(
        ("lin_vel_x", "lin_vel_y", "ang_vel_z"), velocity, strict=True
    ):
        setattr(twist.ranges, axis, (value, value))
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.rel_turn_in_place_envs = 0.0
    for name in ("head_pose", "body_pose"):
        pose = cfg.commands[name]
        pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)
    for name in ("standing_envs", "head_pose_range", "body_pose_range"):
        cfg.curriculum.pop(name, None)


def verify_fixed_commands(env, velocity) -> None:
    import torch

    for name, expected in (("twist", velocity), ("head_pose", 0.0), ("body_pose", 0.0)):
        actual = env.command_manager.get_command(name)
        target = torch.as_tensor(expected, dtype=actual.dtype, device=actual.device)
        if not torch.isfinite(actual).all() or not torch.allclose(
            actual, target.expand_as(actual), atol=1e-6, rtol=0.0
        ):
            raise RuntimeError(f"Fixed evaluation command changed: {name}")


class EvaluationMetrics:
    """Accumulate terminal states through mjlab's pre-reset metrics callback.

    No reset method: these totals span the entire evaluation, including unfinished
    episodes. Storage is O(num_envs), and accumulators stay on the simulation device.
    """

    def __init__(self, cfg, env):
        import torch

        self.num_envs = env.num_envs
        self.steps = 0
        self.sums = torch.zeros(
            (4, env.num_envs, 3), dtype=torch.float64, device=env.device
        )
        self.finite = torch.ones((), dtype=torch.bool, device=env.device)
        self.lengths = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.completed = torch.zeros(2, dtype=torch.long, device=env.device)
        self.term_names = tuple(env.cfg.terminations)
        self.term_counts = torch.zeros(
            len(self.term_names), dtype=torch.long, device=env.device
        )
        self.first_steps = torch.zeros_like(self.lengths)
        self.first_ended = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.first_terminated = torch.zeros_like(self.first_ended)
        self.first_time_limit = torch.zeros_like(self.first_ended)

    def __call__(self, env):
        import torch

        data = env.scene["robot"].data
        actual = torch.cat(
            (data.root_link_lin_vel_b[:, :2], data.root_link_ang_vel_b[:, 2:3]), dim=1
        ).double()
        command = env.command_manager.get_command("twist").double()
        self.finite &= torch.isfinite(actual).all() & torch.isfinite(command).all()
        error = actual - command
        self.sums[0] += command
        self.sums[1] += actual
        self.sums[2] += error.abs()
        self.sums[3] += error.square()
        terminated, time_out = env.reset_terminated.bool(), env.reset_time_outs.bool()
        done = terminated | time_out
        self.steps += 1
        self.lengths += 1
        self.completed[0] += done.sum()
        self.completed[1] += torch.where(done, self.lengths, 0).sum()
        self.lengths.masked_fill_(done, 0)
        self.first_steps += (~self.first_ended).long()
        first_done = ~self.first_ended & done
        self.first_terminated |= first_done & terminated
        self.first_time_limit |= first_done & time_out & ~terminated
        self.first_ended |= done
        for index, name in enumerate(self.term_names):
            self.term_counts[index] += env.termination_manager.get_term(name).sum()
        return error[:, :2].square().sum(dim=1).sqrt().float()

    def report(self, *, steps: int, step_dt: float) -> dict:
        import torch

        if self.steps != steps or steps <= 0:
            raise RuntimeError(
                f"Expected {steps} evaluation metric callbacks, got {self.steps}"
            )
        if not self.finite.item() or not torch.isfinite(self.sums).all().item():
            raise RuntimeError("Nonfinite pre-reset evaluation velocity metrics")
        per_env = (self.sums / steps).cpu()
        means = per_env.mean(dim=1)
        counts = dict(
            zip(self.term_names, self.term_counts.cpu().tolist(), strict=True)
        )
        if counts.get("nan_state", 0):
            raise RuntimeError("Nonfinite simulation state during evaluation")
        completed, completed_steps = self.completed.cpu().tolist()
        first_steps = self.first_steps.cpu()
        first_terminated = self.first_terminated.cpu()
        return {
            "completed_episodes": completed,
            "mean_completed_episode_steps": completed_steps / completed
            if completed
            else None,
            "unfinished_episode_steps": self.lengths.cpu().tolist(),
            "termination_counts": counts,
            "velocity_tracking": {
                "axes": ["vx", "vy", "wz"],
                "units": ["m/s", "m/s", "rad/s"],
                "frame": "root link body frame",
                "sampling": "after physics, before reset and command resampling; mjlab metrics phase",
                "samples": steps * self.num_envs,
                "mean_command": means[0].tolist(),
                "mean_actual": means[1].tolist(),
                "mae": means[2].tolist(),
                "rmse": means[3].sqrt().tolist(),
                "planar_velocity_rmse_m_s": float(means[3, :2].sum().sqrt()),
                "includes_terminal_steps": True,
                "per_environment": {
                    "scope": "Each environment slot across all control steps, including terminal steps and subsequent episodes",
                    "samples_per_environment": steps,
                    "mean_command": per_env[0].tolist(),
                    "mean_actual": per_env[1].tolist(),
                    "bias": (per_env[1] - per_env[0]).tolist(),
                    "mae": per_env[2].tolist(),
                    "rmse": per_env[3].sqrt().tolist(),
                    "planar_velocity_rmse_m_s": per_env[3, :, :2]
                    .sum(dim=1)
                    .sqrt()
                    .tolist(),
                },
            },
            "initial_episodes": {
                "duration_steps": first_steps.tolist(),
                "duration_seconds": (first_steps.double() * step_dt).tolist(),
                "mean_observed_duration_seconds": float(
                    first_steps.double().mean() * step_dt
                ),
                "terminated": first_terminated.tolist(),
                "time_limit_only": self.first_time_limit.cpu().tolist(),
                "right_censored": (~first_terminated).tolist(),
                "still_running": (~self.first_ended).cpu().tolist(),
                "survived_full_horizon_count": int(
                    ((first_steps == steps) & ~first_terminated).sum()
                ),
                "censored_before_horizon_count": int(
                    ((first_steps < steps) & ~first_terminated).sum()
                ),
            },
        }


def validate_metrics(
    checkpoint: Path, iterations: int, start_iteration: int = 0
) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(checkpoint.parent), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
    steps = {event.step for event in scalars.get("Loss/value", [])}
    if steps != set(range(start_iteration, start_iteration + iterations)):
        raise RuntimeError(
            f"Expected {iterations} PPO updates, found steps {sorted(steps)}"
        )
    for tag, events in scalars.items():
        if any(not math.isfinite(event.value) for event in events):
            raise RuntimeError(f"Nonfinite training metric: {tag}")
    nan_states = scalars.get("Episode_Termination/nan_state", [])
    if not nan_states or any(event.value != 0 for event in nan_states):
        raise RuntimeError("Missing NaN-state metric or simulation reported NaN states")
    return {
        "iterations": iterations,
        "start_iteration": start_iteration,
        "all_scalars_finite": True,
        "nan_states": 0,
        "last_values": {
            tag: events[-1].value for tag, events in scalars.items() if events
        },
    }


def training_progress(directory: Path) -> dict:
    """Read flushed TensorBoard events without loading a policy or accessing CUDA."""
    import statistics
    import time

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory = directory.expanduser().resolve()
    manifest = json.loads((directory / "run.json").read_text())
    if manifest.get("workflow") not in ("train", "smoke"):
        raise ValueError("progress requires a train or smoke run directory")
    commands = [
        command
        for command in manifest.get("commands", [])
        if "--agent.max-iterations" in command
    ]
    requested = (
        int(commands[-1][commands[-1].index("--agent.max-iterations") + 1])
        if commands
        else None
    )
    start = manifest.get("resume", {}).get("iteration", 0)
    report = {
        "run": str(directory),
        "run_status": manifest["status"],
        "requested_updates": requested,
        "start_iteration": start,
        "observed_updates": 0,
        "latest_iteration": None,
        "seconds_since_last_event": None,
        "estimated_remaining_seconds": None,
        "all_recorded_scalars_finite": None,
        "nan_states_seen": None,
        "latest_metrics": {},
    }
    logs = {
        path.parent
        for path in directory.glob("logs/rsl_rl/microduck/*/events.out.tfevents.*")
    }
    if len(logs) > 1:
        raise ValueError("Ambiguous training log directories")
    checkpoints = list(directory.glob("logs/rsl_rl/microduck/*/model_*.pt"))
    report["latest_checkpoint"] = (
        str(max(checkpoints, key=lambda path: int(path.stem.removeprefix("model_"))))
        if checkpoints
        else None
    )
    if not logs:
        return report
    accumulator = EventAccumulator(str(logs.pop()), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {tag: accumulator.Scalars(tag) for tag in accumulator.Tags()["scalars"]}
    losses = scalars.get("Loss/value", [])
    if losses:
        steps = {event.step for event in losses}
        report["observed_updates"] = len(steps)
        report["latest_iteration"] = max(steps)
        report["seconds_since_last_event"] = max(
            0.0, time.time() - max(event.wall_time for event in losses)
        )
    if any(scalars.values()):
        report["all_recorded_scalars_finite"] = all(
            math.isfinite(event.value)
            for values in scalars.values()
            for event in values
        )
    nan_events = scalars.get("Episode_Termination/nan_state", [])
    report["nan_states_seen"] = (
        any(event.value != 0 for event in nan_events) if nan_events else None
    )
    selected = (
        "Loss/value",
        "Loss/surrogate",
        "Loss/learning_rate",
        "Policy/mean_std",
        "Train/mean_reward",
        "Train/mean_episode_length",
        "Perf/total_fps",
    )
    report["latest_metrics"] = {
        tag: scalars[tag][-1].value for tag in selected if scalars.get(tag)
    }
    collection = {
        event.step: event.value for event in scalars.get("Perf/collection_time", [])
    }
    learning = {
        event.step: event.value for event in scalars.get("Perf/learning_time", [])
    }
    paired = sorted(collection.keys() & learning.keys())[-20:]
    if paired and requested is not None and report["all_recorded_scalars_finite"]:
        median = statistics.median(collection[step] + learning[step] for step in paired)
        report["median_iteration_seconds_last_20"] = median
        if manifest["status"] in ("running", "complete"):
            report["estimated_remaining_seconds"] = (
                max(0, requested - report["observed_updates"]) * median
            )
    # JSON must remain valid even when diagnostics discover a corrupt metric.
    report["latest_metrics"] = {
        tag: value if math.isfinite(value) else None
        for tag, value in report["latest_metrics"].items()
    }
    return report


def checkpoint_metadata(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    iteration = checkpoint["iter"]
    counter = checkpoint["infos"]["env_state"]["common_step_counter"]
    if (
        type(iteration) is not int
        or iteration < 0
        or type(counter) is not int
        or counter < 0
    ):
        raise ValueError("Invalid checkpoint iteration or curriculum counter")
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        if not checkpoint.get(key):
            raise ValueError(f"Checkpoint has no {key}")
    rates = [
        group["lr"] for group in checkpoint["optimizer_state_dict"]["param_groups"]
    ]
    if not rates or any(
        not math.isfinite(rate) or rate <= 0 or rate != rates[0] for rate in rates
    ):
        raise ValueError("Expected a single positive PPO learning rate")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "iteration": iteration,
        "common_step_counter": counter,
        "learning_rate": rates[0],
        "sha256": digest.hexdigest(),
    }


def restore_training_counter(env, env_ids, *, counter):
    """Startup event: restore curricula before the wrapper's first reset."""
    env.common_step_counter = counter


def audit_training_reset(env, env_ids, *, counter, report_path, checkpoint_sha256):
    """Record the first reset after curricula ran; never change later resets."""
    if getattr(env, "_ef_resume_audited", False):
        return
    if env.common_step_counter != counter:
        raise RuntimeError(
            "Resume curriculum counter was not restored before first reset"
        )
    report = {
        "checkpoint_sha256": checkpoint_sha256,
        "counter_at_first_reset": env.common_step_counter,
        "num_envs": env.num_envs,
        "action_rate_weight": env.reward_manager.get_term_cfg("action_rate_l2").weight,
        "standing_env_fraction": env.command_manager.get_term_cfg(
            "twist"
        ).rel_standing_envs,
        "head_pose_ranges": env.command_manager.get_term_cfg("head_pose").ranges,
    }
    with Path(report_path).open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    env._ef_resume_audited = True


def resume_training(checkpoint: Path, num_envs: int, iterations: int) -> dict:
    """Use the upstream launcher with local startup/reset events for restoration."""
    from dataclasses import replace

    from mjlab.managers import EventTermCfg
    from mjlab.scripts.train import TrainConfig, launch_training

    if num_envs <= 0 or iterations <= 0:
        raise ValueError("Environment and iteration counts must be positive")
    expected = Path("logs/rsl_rl/microduck/resume_source/checkpoint.pt").resolve()
    if checkpoint.resolve() != expected:
        raise ValueError("Resume training requires the launcher's checkpoint snapshot")
    metadata = checkpoint_metadata(checkpoint)
    cfg = replace(
        TrainConfig.from_task("Mjlab-Velocity-Flat-MicroDuck"), enable_nan_guard=True
    )
    cfg.env.scene.num_envs = num_envs
    cfg.agent.max_iterations = iterations
    cfg.agent.seed = 0
    cfg.agent.logger = "tensorboard"
    cfg.agent.experiment_name = "microduck"
    cfg.agent.run_name = "embodiedforge"
    cfg.agent.resume = True
    cfg.agent.load_run = "^resume_source$"
    cfg.agent.load_checkpoint = r"^checkpoint\.pt$"
    cfg.agent.algorithm.learning_rate = metadata["learning_rate"]
    cfg.env.events["ef_resume_counter"] = EventTermCfg(
        func=restore_training_counter,
        mode="startup",
        params={"counter": metadata["common_step_counter"]},
    )
    audit = Path("resume.initialization.json").resolve()
    cfg.env.events["ef_resume_audit"] = EventTermCfg(
        func=audit_training_reset,
        mode="reset",
        params={
            "counter": metadata["common_step_counter"],
            "report_path": str(audit),
            "checkpoint_sha256": metadata["sha256"],
        },
    )
    launch_training("Mjlab-Velocity-Flat-MicroDuck", cfg)
    return json.loads(audit.read_text())


@contextmanager
def policy_environment(
    checkpoint: Path,
    num_envs: int,
    seed: int,
    *,
    velocity=None,
    no_pushes=False,
    full_precision=False,
    evaluation_metrics=False,
    curriculum_step=None,
):
    """Own one environment for both bounded evaluation and interactive playback."""
    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    metadata = checkpoint_metadata(checkpoint)
    if curriculum_step is not None and (
        type(curriculum_step) is not int or curriculum_step < 0
    ):
        raise ValueError("Evaluation curriculum step must be a nonnegative integer")
    start_counter = (
        metadata["common_step_counter"] if curriculum_step is None else curriculum_step
    )
    configure_torch_backends(allow_tf32=not full_precision)
    task = "Mjlab-Velocity-Flat-MicroDuck"
    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.nan_guard.enabled = True
    configure_evaluation(cfg, velocity, no_pushes)
    if evaluation_metrics:
        from mjlab.managers.metrics_manager import MetricsTermCfg

        cfg.metrics["ef_velocity_tracking"] = MetricsTermCfg(func=EvaluationMetrics)
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        # The wrapper resets immediately; curricula must already use the saved
        # counter on that first reset, before runner.load restores it again.
        env.common_step_counter = start_counter
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
            wrapped, asdict(agent_cfg), device="cuda:0"
        )
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location="cuda:0",
        )
        # runner.load also restores the saved counter, including actor-only loads.
        env.common_step_counter = start_counter
        policy = runner.get_inference_policy(device="cuda:0")
        yield wrapped, policy, metadata
    finally:
        env.close()


class JoinedRenderThreadMixin:
    """Wait for MuJoCo 3.10's owned render thread before GLFW's atexit cleanup.

    launch_passive does not expose its Thread through Handle. The pinned SDK's
    _launch_internal target identifies only the threads created by this setup.
    No SDK methods, warning handlers or global GLFW state are modified.
    """

    def setup(self):
        existing = set(threading.enumerate())
        self._owned_render_threads = []
        try:
            return super().setup()
        finally:
            self._owned_render_threads = [
                thread
                for thread in threading.enumerate()
                if thread not in existing
                and getattr(thread, "_target", None) is self._render_target
            ]

    def close(self):
        try:
            super().close()
        finally:
            for thread in getattr(self, "_owned_render_threads", []):
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError(
                        "MuJoCo render thread did not stop within 5 seconds"
                    )
            self._owned_render_threads = []


def play(checkpoint: Path, viewer_kind: str, steps: int | None, seed: int) -> dict:
    if steps is not None and steps <= 0:
        raise ValueError("Playback step count must be positive")
    if viewer_kind != "native":
        raise ValueError(f"Unknown viewer: {viewer_kind}")
    import mujoco.viewer
    from mjlab.viewer import NativeMujocoViewer

    if importlib.metadata.version("mujoco") != "3.10.0":
        raise RuntimeError("Joined native viewer requires the pinned MuJoCo 3.10.0")

    class ManagedViewer(JoinedRenderThreadMixin, NativeMujocoViewer):
        _render_target = staticmethod(mujoco.viewer._launch_internal)

    with policy_environment(checkpoint, 1, seed) as (env, policy, metadata):
        viewer = ManagedViewer(env, policy)
        try:
            viewer.run(num_steps=steps)
            if viewer._last_error is not None:
                raise RuntimeError(f"Playback failed: {viewer._last_error}")
            return {
                "viewer": viewer_kind,
                "simulation_steps": viewer._step_count,
                "interrupted": viewer._interrupted,
                "checkpoint_metadata": metadata,
            }
        finally:
            viewer.close()


@contextmanager
def evaluation_video(env, path: Path | None):
    """Stream env 0 to MP4; bound memory and close rendering before the env."""
    if path is None:
        yield lambda: None, None
        return
    from imageio_ffmpeg import get_ffmpeg_exe
    from mjlab.viewer.offscreen_renderer import OffscreenRenderer
    from mjlab.viewer.viewer_config import ViewerConfig

    if path.exists():
        raise FileExistsError(path)
    cfg = ViewerConfig(
        width=640,
        height=480,
        distance=0.8,
        elevation=-20,
        azimuth=135,
        origin_type=ViewerConfig.OriginType.ASSET_ROOT,
        entity_name="robot",
        max_extra_envs=0,
    )
    info = {
        "path": path.name,
        "environment": 0,
        "width": cfg.width,
        "height": cfg.height,
        "fps": 1 / env.step_dt,
        "frames": 0,
        "sampling": "after each control step, including automatic episode resets",
        "gl_backend": os.environ.get("MUJOCO_GL"),
    }
    renderer = OffscreenRenderer(env.sim.mj_model, cfg, env.scene)
    try:
        renderer.initialize()
        from OpenGL import GL

        info["opengl"] = {
            name: (GL.glGetString(token) or b"").decode("utf-8", errors="replace")
            for name, token in (
                ("vendor", GL.GL_VENDOR),
                ("renderer", GL.GL_RENDERER),
                ("version", GL.GL_VERSION),
            )
        }
        process = subprocess.Popen(
            [
                get_ffmpeg_exe(),
                "-nostdin",
                "-loglevel",
                "error",
                "-n",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{cfg.width}x{cfg.height}",
                "-r",
                str(info["fps"]),
                "-i",
                "pipe:0",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )
        try:

            def capture():
                renderer.update(env.sim.data)
                frame = renderer.render()
                process.stdin.write(frame.tobytes())
                info["frames"] += 1

            yield capture, info
        finally:
            primary_error = sys.exc_info()[1]
            try:
                process.stdin.close()
                code = process.wait(timeout=10)
                if code:
                    raise RuntimeError(f"Video encoder exited with code {code}")
            except (OSError, subprocess.TimeoutExpired, RuntimeError) as cleanup_error:
                # SIGINT reaches the encoder in the same process group. Its 255
                # exit must not turn an interrupted evaluation into a failure.
                if primary_error is None:
                    raise
                if not isinstance(primary_error, KeyboardInterrupt) and hasattr(
                    primary_error, "add_note"
                ):
                    primary_error.add_note(
                        f"Video cleanup also reported: {cleanup_error}"
                    )
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        info["sha256"] = digest.hexdigest()
    finally:
        renderer.close()


def evaluate(
    checkpoint: Path,
    num_envs: int,
    steps: int,
    seed: int,
    *,
    velocity=None,
    no_pushes=False,
    onnx=None,
    curriculum_step=None,
    video=None,
) -> dict:
    import torch

    if num_envs <= 0 or steps <= 0:
        raise ValueError("Environment and step counts must be positive")
    if video:
        # This worker owns the process; select headless GL before importing MuJoCo.
        os.environ.setdefault("MUJOCO_GL", "egl")
    session = onnx_session(Path(onnx)) if onnx else None
    parity = None
    if session is not None:
        parity = {
            "onnx": onnx,
            "sha256": hashlib.sha256(Path(onnx).read_bytes()).hexdigest(),
            "provider": "CPUExecutionProvider",
            "sampling": "one environment per step, index = step % num_envs",
            "atol": 1e-4,
            "rtol": 1e-4,
            "samples": 0,
            "max_absolute_error": 0.0,
        }
    with (
        policy_environment(
            checkpoint,
            num_envs,
            seed,
            velocity=velocity,
            no_pushes=no_pushes,
            full_precision=session is not None,
            evaluation_metrics=True,
            curriculum_step=curriculum_step,
        ) as (wrapped, policy, metadata),
        ExitStack() as resources,
    ):
        env = wrapped.unwrapped
        capture, video_info = resources.enter_context(
            evaluation_video(env, Path(video) if video else None)
        )
        metrics = env.metrics_manager.cfg["ef_velocity_tracking"].func
        obs = wrapped.get_observations()
        reward_sum = torch.zeros((), device="cuda:0")
        for step in range(steps):
            if velocity is not None:
                verify_fixed_commands(env, velocity)
            if (
                obs["actor"].shape != (num_envs, 61)
                or not torch.isfinite(obs["actor"]).all()
            ):
                raise RuntimeError("Invalid actor observations during evaluation")
            with torch.inference_mode():
                actions = policy(obs)
                if actions.shape != (num_envs, 14) or not torch.isfinite(actions).all():
                    raise RuntimeError("Invalid policy actions during evaluation")
                if session is not None:
                    index = step % num_envs
                    observation = obs["actor"][index : index + 1].cpu().numpy()
                    candidate = session.run(
                        None, {session.get_inputs()[0].name: observation}
                    )[0]
                    try:
                        error = compare_policy_actions(
                            actions[index : index + 1].cpu().numpy(), candidate
                        )
                    except RuntimeError as exc:
                        raise RuntimeError(
                            f"Parity failed at step {step}, environment {index}: {exc}"
                        ) from exc
                    parity["samples"] += 1
                    parity["max_absolute_error"] = max(
                        parity["max_absolute_error"], error
                    )
                obs, rewards, _, _ = wrapped.step(actions)
            if not torch.isfinite(rewards).all():
                raise RuntimeError("Nonfinite evaluation rewards")
            reward_sum += rewards.sum()
            capture()
        if not torch.isfinite(obs["actor"]).all():
            raise RuntimeError("Nonfinite simulation state during evaluation")
        if velocity is not None:
            verify_fixed_commands(env, velocity)
        return {
            "checkpoint": str(checkpoint),
            "checkpoint_metadata": metadata,
            "task": "Mjlab-Velocity-Flat-MicroDuck",
            "recipe": "upstream play=True with explicit evaluation overrides",
            "conditions": {
                "velocity_body_frame": velocity,
                "neutral_head_body_commands": velocity is not None,
                "pushes_enabled": "push_robot" in env.cfg.events,
                "fixed_command_checks": steps + 1 if velocity is not None else 0,
                "other_randomization": "upstream play=True",
                "policy_matmul_precision": torch.backends.cuda.matmul.fp32_precision,
                "curriculum_start_step": metadata["common_step_counter"]
                if curriculum_step is None
                else curriculum_step,
                "curriculum_source": "checkpoint"
                if curriculum_step is None
                else "override",
            },
            "onnx_parity": parity,
            "video": video_info,
            "seed": seed,
            "num_envs": num_envs,
            "steps_per_env": steps,
            "transitions": steps * num_envs,
            "sim_seconds_per_env": steps * env.step_dt,
            "mean_reward_per_transition": float(reward_sum.item() / (steps * num_envs)),
            **metrics.report(steps=steps, step_dt=env.step_dt),
            "finite_observations_actions_rewards": True,
        }


if __name__ == "__main__":
    if sys.argv[1] == "check":
        report = check()
        if len(sys.argv) > 2:
            Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
    elif sys.argv[1] == "onnx":
        report = validate_onnx(Path(sys.argv[2]))
        Path(sys.argv[2]).with_suffix(".validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    elif sys.argv[1] == "metrics":
        report = validate_metrics(
            Path(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[5]) if len(sys.argv) > 5 else 0,
        )
        Path(sys.argv[4]).write_text(json.dumps(report, indent=2) + "\n")
    elif sys.argv[1] == "checkpoint":
        report = checkpoint_metadata(Path(sys.argv[2]))
        Path(sys.argv[3]).write_text(json.dumps(report, indent=2) + "\n")
    elif sys.argv[1] == "progress":
        report = training_progress(Path(sys.argv[2]))
    elif sys.argv[1] == "resume-train":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("checkpoint", type=Path)
        parser.add_argument(
            "--env.scene.num-envs", dest="num_envs", type=int, required=True
        )
        parser.add_argument(
            "--agent.max-iterations", dest="iterations", type=int, required=True
        )
        args = parser.parse_args(sys.argv[2:])
        report = resume_training(args.checkpoint, args.num_envs, args.iterations)
    elif sys.argv[1] == "evaluate":
        try:
            options = json.loads(sys.argv[7]) if len(sys.argv) > 7 else {}
            if options.get("video"):
                options["video"] = str(Path(sys.argv[6]).parent / "policy.mp4")
            report = evaluate(
                Path(sys.argv[2]),
                int(sys.argv[3]),
                int(sys.argv[4]),
                int(sys.argv[5]),
                **options,
            )
        except Exception as exc:
            Path(sys.argv[6]).with_suffix(".failure.json").write_text(
                json.dumps(
                    {"error_type": type(exc).__name__, "error": str(exc)}, indent=2
                )
                + "\n"
            )
            raise
        Path(sys.argv[6]).write_text(json.dumps(report, indent=2) + "\n")
    elif sys.argv[1] == "play":
        report = play(
            Path(sys.argv[2]),
            sys.argv[3],
            None if sys.argv[4] == "none" else int(sys.argv[4]),
            int(sys.argv[5]),
        )
    else:
        raise ValueError(f"Unknown worker operation: {sys.argv[1]}")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "installed_packages"}, indent=2
        )
    )
