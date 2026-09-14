"""Bounded H1 policy evaluation in the isolated IsaacLab Python process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    # Add the package parent, never the directory containing our logging.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from embodiedforge._h1_metrics import BASIC_COMMANDS, FirstEpisodeMetrics
    from embodiedforge._h1_motion import MotionRecorder
    from embodiedforge._h1_worker import preflight
    from embodiedforge.h1 import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--record-motion", action="store_true")
    parser.add_argument("--record-env", type=int, default=0)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--velocity", type=float, nargs=3)
    selection.add_argument("--suite", choices=["basic"])
    args = parser.parse_args()
    runtime = preflight(args)
    cases = BASIC_COMMANDS if args.suite else {"custom": args.velocity}
    current_velocity = next(iter(cases.values()))

    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    import torch
    from isaaclab.managers import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import DatasetExportMode
    from isaaclab.utils.math import quat_apply_inverse, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import (
        add_launcher_args,
        launch_simulation,
        resolve_task_config,
    )
    from rsl_rl.runners import OnPolicyRunner

    launcher_parser = argparse.ArgumentParser()
    add_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args(["--visualizer", "none"])
    sys.argv = [sys.argv[0], "physics=newton_mjwarp"]
    task = "Isaac-Velocity-Flat-H1-Play-v0"
    env_cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = (args.steps + 1) * dt
    command = env_cfg.commands.base_velocity
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = command.rel_standing_envs = 0.0

    def set_velocity(config, velocity):
        config.ranges.lin_vel_x = (velocity[0], velocity[0])
        config.ranges.lin_vel_y = (velocity[1], velocity[1])
        config.ranges.ang_vel_z = (velocity[2], velocity[2])

    set_velocity(command, current_velocity)
    command.debug_vis = False
    agent_cfg.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, runtime["versions"]["rsl-rl-lib"]
    )
    metrics = FirstEpisodeMetrics(args.num_envs, args.steps, dt)
    motion = None

    class Snapshot(RecorderTerm):
        def record_post_step(self):
            env = self._env
            data = env.scene["robot"].data
            linear = quat_apply_inverse(
                yaw_quat(data.root_quat_w.torch), data.root_lin_vel_w.torch
            )
            velocity = torch.cat(
                (linear[:, :2], data.root_ang_vel_w.torch[:, 2:3]), dim=1
            )
            actual_command = env.command_manager.get_command("base_velocity")
            expected = torch.tensor(current_velocity, device=velocity.device)
            if not torch.allclose(actual_command, expected.expand_as(actual_command)):
                raise ValueError(
                    "Evaluation command differs from the requested fixed velocity"
                )
            metrics.update(
                velocity.cpu().numpy(),
                actual_command.cpu().numpy(),
                env.reset_terminated.cpu().numpy(),
                env.reset_time_outs.cpu().numpy(),
            )
            if motion is not None and not motion.ended:
                i = args.record_env
                motion.append(
                    metrics.steps * dt,
                    data.body_link_pos_w.torch[i].cpu().numpy(),
                    data.body_link_quat_w.torch[i].cpu().numpy(),
                    data.joint_pos.torch[i].cpu().numpy(),
                    bool(env.reset_terminated[i]),
                    bool(env.reset_time_outs[i]),
                )
            return None, None

    recorder = RecorderManagerBaseCfg(dataset_export_mode=DatasetExportMode.EXPORT_NONE)
    recorder.snapshot = RecorderTermCfg(class_type=Snapshot)
    env_cfg.recorders = recorder
    env_cfg.log_dir = str(args.output.parent)
    with launch_simulation(env_cfg, launcher_args):
        base = gym.make(task, cfg=env_cfg)
        try:
            env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device
            )
            runner.load(str(args.checkpoint), map_location=env.unwrapped.device)
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            robot = env.unwrapped.scene["robot"]
            if args.record_motion:
                # Resolve the real model topology in the same link order as poses.
                view = robot.root_view
                model = view.model
                aid = int(view.articulation_ids.numpy()[args.record_env, 0])
                start, end = model.articulation_start.numpy()[aid : aid + 2]
                parents = model.joint_parent.numpy()[start:end]
                children = model.joint_child.numpy()[start:end]
                body_ids = sorted(set(int(i) for i in children))
                if [
                    model.body_label[i].rsplit("/", 1)[-1] for i in body_ids
                ] != robot.body_names:
                    raise ValueError("Motion topology and body pose ordering differ")
                local_ids = {body_id: i for i, body_id in enumerate(body_ids)}
                edges = [
                    [local_ids[int(p)], local_ids[int(c)]]
                    for p, c in zip(parents, children, strict=True)
                    if int(p) in local_ids
                ]
                estimated_bytes = (
                    args.steps * (7 * len(body_ids) + len(robot.joint_names) + 4) * 4
                )
                if estimated_bytes > 256 * 1024 * 1024:
                    raise ValueError(
                        "Requested motion exceeds the 256 MiB recording limit"
                    )
            reports = []
            for case, current_velocity in cases.items():
                metrics = FirstEpisodeMetrics(args.num_envs, args.steps, dt)
                if args.suite:
                    # Reset the task between cases; startup material/mass samples
                    # remain paired within a seed. Reset randomization is reseeded.
                    set_velocity(
                        env.unwrapped.command_manager.get_term("base_velocity").cfg,
                        current_velocity,
                    )
                    env.seed(args.seed)
                    obs, _ = env.reset()
                else:
                    obs = env.get_observations()
                motion = (
                    MotionRecorder(
                        {
                            "schema": 1,
                            "case": case,
                            "seed": args.seed,
                            "env_id": args.record_env,
                            "velocity_command": current_velocity,
                            "dt": dt,
                            "body_names": robot.body_names,
                            "joint_names": robot.joint_names,
                            "edges": edges,
                            "coordinate_frame": "world",
                            "quaternion_order": "xyzw",
                            "sampling": "post-physics, before auto-reset; first episode only",
                        }
                    )
                    if args.record_motion
                    else None
                )
                with torch.inference_mode():
                    for _ in range(args.steps):
                        if not all(
                            torch.isfinite(value).all() for value in obs.values()
                        ):
                            raise ValueError("Non-finite evaluation observation")
                        actions = policy(obs)
                        if not torch.isfinite(actions).all():
                            raise ValueError("Non-finite evaluation action")
                        obs, _, dones, _ = env.step(actions)
                        policy.reset(dones)
                        if not metrics.active.any():
                            break
                report = metrics.report()
                report.update(
                    seed=args.seed,
                    case=case,
                    velocity_command=current_velocity,
                    runtime=runtime,
                    task=task,
                    physics="newton_mjwarp",
                    velocity_frame="linear: yaw-aligned base frame; angular: world z",
                    sampling="post-physics, before auto-reset; first episode only",
                    reset_protocol="reseed and reset before each case"
                    if args.suite
                    else "initial wrapper reset",
                    randomization="upstream Play: observation noise and pushes disabled; startup and reset randomization retained",
                )
                if motion is not None:
                    motion_path = (
                        args.output.parent
                        / f"motion-seed-{args.seed}-{case}-env-{args.record_env}.npz"
                    )
                    motion.save(motion_path)
                    report["motion"] = {
                        "path": motion_path.name,
                        "frames": len(motion.frames),
                        "env_id": args.record_env,
                    }
                reports.append(report)
                print(
                    f"[H1] {case}: survival={report['survival_fraction']:.3f}, mean_velocity={report['mean_velocity']}",
                    flush=True,
                )
        finally:
            base.close()
    write_json(
        args.output,
        {"suite": args.suite, "seed": args.seed, "cases": reports}
        if args.suite
        else reports[0],
    )


if __name__ == "__main__":
    main()
