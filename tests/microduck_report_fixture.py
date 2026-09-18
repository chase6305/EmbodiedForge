"""Complete deterministic report for process-boundary tests without a simulator."""

import json

from embodiedforge._microduck_run import checkpoint_digest


def evaluation_report(command):
    num_envs, steps, seed = map(int, command[5:8])
    options = json.loads(command[9])
    parity = None
    if options["onnx"]:
        parity = {
            "onnx": options["onnx"],
            "sha256": checkpoint_digest(options["onnx"]),
            "provider": "CPUExecutionProvider",
            "atol": 1e-4,
            "rtol": 1e-4,
            "sampling": "one environment per step, index = step % num_envs",
            "samples": steps,
            "max_absolute_error": 0.0,
        }
    return {
        "seed": seed,
        "task": "Mjlab-Velocity-Flat-MicroDuck",
        "checkpoint": command[4],
        "checkpoint_metadata": {
            "sha256": checkpoint_digest(command[4]),
            "common_step_counter": 0,
        },
        "conditions": {
            "velocity_body_frame": options["velocity"],
            "pushes_enabled": not options["no_pushes"],
            "policy_matmul_precision": "ieee",
            "curriculum_start_step": options.get("curriculum_step") or 0,
            "curriculum_source": "checkpoint"
            if options.get("curriculum_step") is None
            else "override",
        },
        "num_envs": num_envs,
        "steps_per_env": steps,
        "sim_seconds_per_env": steps * 0.02,
        "transitions": num_envs * steps,
        "onnx_parity": parity,
        "finite_observations_actions_rewards": True,
        "termination_counts": {"nan_state": 0, "fell_over": 0},
        "mean_reward_per_transition": 0.1,
        "velocity_tracking": {
            "samples": num_envs * steps,
            "mean_command": [0.0] * 3,
            "mean_actual": [0.0] * 3,
            "rmse": [0.0] * 3,
            "planar_velocity_rmse_m_s": 0.0,
            "axes": ["vx", "vy", "wz"],
            "units": ["m/s", "m/s", "rad/s"],
            "frame": "root link body frame",
            "sampling": "pre-reset",
            "includes_terminal_steps": True,
        },
        "initial_episodes": {
            "duration_steps": [steps] * num_envs,
            "mean_observed_duration_seconds": steps * 0.02,
            "survived_full_horizon_count": num_envs,
            "censored_before_horizon_count": 0,
        },
    }
