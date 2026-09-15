"""Native H1 train/resume/evaluate entry point, independent of IsaacLab."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from ._training_runtime import record_training_runtime


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_tensors(value):
    import math

    import torch

    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite checkpoint tensor")
    elif isinstance(value, dict):
        for child in value.values():
            validate_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_tensors(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite checkpoint scalar")


def load_run(directory):
    import mujoco
    import torch

    from .locomotion.h1_native import VERSION
    from .locomotion.h1_ppo import Policy

    metadata = json.loads((directory / "run.json").read_text())
    if metadata.get("task") != VERSION or metadata.get("status") != "complete":
        raise ValueError(
            "Expected a complete native H1 run; IsaacLab checkpoints are incompatible"
        )
    for file, key in (
        ("model.mjb", "model_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"),
    ):
        if digest(directory / file) != metadata[key]:
            raise ValueError(f"Native H1 artifact hash mismatch: {file}")
    checkpoint = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=True
    )
    validate_tensors(checkpoint)
    if type(checkpoint.get("updates")) is not int or checkpoint["updates"] <= 0:
        raise ValueError("Invalid native H1 checkpoint update count")
    if (
        checkpoint.get("task") != VERSION
        or checkpoint.get("model_sha256") != metadata["model_sha256"]
    ):
        raise ValueError("Checkpoint task/model does not match the run")
    policy = Policy()
    policy.load_state_dict(checkpoint["policy"], strict=True)
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise ValueError("Checkpoint contains non-finite policy parameters")
    model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
    names = [model.joint(int(j)).name for j in model.actuator_trnid[:, 0]]
    if (
        names != checkpoint["joint_names"]
        or checkpoint["updates"] != metadata["updates"]
    ):
        raise ValueError("Checkpoint joint order/update count does not match run")
    return model, policy, checkpoint, metadata


def train(args):
    import importlib.metadata

    import mujoco
    import numpy as np
    import torch

    from .locomotion.h1_native import H1, VERSION, build_model
    from .locomotion.h1_ppo import Policy, collect, optimize

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.resume:
        model, policy, checkpoint, previous = load_run(args.resume)
        start = checkpoint["updates"]
    else:
        model, policy, checkpoint, start = build_model(args.model), Policy(), None, 0
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    if args.resume:
        shutil.copyfile(args.resume / "model.mjb", root / "model.mjb")
    else:
        mujoco.mj_saveModel(model, str(root / "model.mjb"))
    metadata = {
        "task": VERSION,
        "status": "running",
        "updates": start,
        "requested_updates": args.updates,
        "initial_updates": start,
        "num_envs": args.num_envs,
        "threads": args.threads,
        "seed": args.seed,
        "horizon": args.horizon,
        "learning_rate": args.learning_rate,
        "model_sha256": digest(root / "model.mjb"),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mujoco", "mjbatch", "torch")
        },
        "physics": "MuJoCo CPU / mjbatch",
        "learner": "EmbodiedForge PPO CPU",
        "resume_semantics": "policy+optimizer; environment and RNG reset",
    }
    write_json(root / "run.json", metadata)
    began = time.monotonic()
    try:
        env = H1(model, args.num_envs, seed=args.seed, threads=args.threads)
        record_training_runtime(
            root,
            environment=env,
            task=env,
            learner=optimize,
            physics_adapter=env.batch,
            physics="mujoco/mjbatch",
            core_vector_env=False,
            packages=["numpy", "torch", "mujoco", "mjbatch"],
        )
        metadata["joint_names"] = list(env.robot.names)
        metadata["joint_groups"] = {
            k: [env.robot.names[i] for i in ids] for k, ids in env.groups.items()
        }
        observation = env.obs()
        with (root / "metrics.jsonl").open("x") as log:
            for update in range(start + 1, start + args.updates + 1):
                rollout, observation = collect(policy, env, observation, args.horizon)
                loss = optimize(policy, optimizer, rollout, rng)
                row = {
                    "update": update,
                    "loss": loss,
                    "mean_reward": float(rollout["reward"].mean()),
                    "falls": int(rollout["terminated"].sum()),
                    "timeouts": int(rollout["truncated"].sum()),
                    "seconds": time.monotonic() - began,
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                if (
                    update == start + 1
                    or update % 10 == 0
                    or update == start + args.updates
                ):
                    print(json.dumps(row), flush=True)
                if update % 50 == 0 or update == start + args.updates:
                    temporary = root / "checkpoint.pt.tmp"
                    validate_tensors(optimizer.state_dict())
                    torch.save(
                        {
                            "task": VERSION,
                            "updates": update,
                            "model_sha256": metadata["model_sha256"],
                            "joint_names": list(env.robot.names),
                            "policy": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        temporary,
                    )
                    temporary.replace(root / "checkpoint.pt")
                    metadata.update(
                        updates=update, checkpoint_sha256=digest(root / "checkpoint.pt")
                    )
                    write_json(root / "run.json", metadata)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["seconds"] = time.monotonic() - began
        write_json(root / "run.json", metadata)


def evaluate(args):
    import numpy as np
    import torch

    from ._h1_motion import MotionRecorder
    from .locomotion.h1_native import CTRL_DT, H1

    torch.set_num_threads(args.threads)
    model, policy, _, metadata = load_run(args.run)
    policy.eval()
    env = H1(
        model,
        args.num_envs,
        seed=args.seed,
        threads=args.threads,
        training=False,
        episode_steps=args.steps + 1,
    )
    env.set_commands(np.tile(args.velocity, (args.num_envs, 1)))
    observation = env.obs()
    args.output.mkdir(parents=True, exist_ok=False)
    alive = np.ones(args.num_envs, bool)
    rewards, counts, torque_peak = (
        np.zeros(args.num_envs),
        np.zeros(args.num_envs, int),
        0.0,
    )
    squared_error, velocity_sum = (
        np.zeros((args.num_envs, 3)),
        np.zeros((args.num_envs, 3)),
    )
    recorder = None
    if args.record_motion:
        recorder = MotionRecorder(
            {
                "title": "Native H1",
                "body_names": [model.body(i).name for i in range(1, model.nbody)],
                "joint_names": list(env.robot.names),
                "quaternion_order": "xyzw",
                "edges": [
                    [int(model.body_parentid[i]) - 1, i - 1]
                    for i in range(1, model.nbody)
                    if model.body_parentid[i] > 0
                ],
                "velocity_command": args.velocity,
                "source": metadata["task"],
            }
        )
    for step in range(args.steps):
        with torch.no_grad():
            action = policy.actor(torch.as_tensor(observation)).numpy()
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards[alive] += reward[alive]
        counts[alive] += 1
        squared_error[alive] += (env.measured_velocity[alive] - args.velocity) ** 2
        velocity_sum[alive] += env.measured_velocity[alive]
        torque_peak = max(
            torque_peak, float(np.abs(env.robot.joint_force[alive]).max())
        )
        if recorder is not None and alive[0]:
            recorder.append(
                (step + 1) * CTRL_DT,
                env.batch.bind("xpos")[0, 1:],
                np.roll(env.batch.bind("xquat")[0, 1:], -1, axis=1),
                env.robot.qpos[0, env.robot.qadr],
                bool(terminated[0]),
                bool(truncated[0]),
            )
        alive &= ~(terminated | truncated)
        if not alive.any():
            break
    if recorder is not None:
        recorder.save(args.output / "motion.npz")
    survival = float(alive.mean())
    planar_rmse = float(np.sqrt(squared_error[:, :2].sum() / counts.sum()))
    yaw_rmse = float(np.sqrt(squared_error[:, 2].sum() / counts.sum()))
    checks = [
        passed
        for limit, passed in (
            (
                args.min_survival,
                args.min_survival is None or survival >= args.min_survival,
            ),
            (
                args.max_planar_rmse,
                args.max_planar_rmse is None or planar_rmse <= args.max_planar_rmse,
            ),
            (
                args.max_yaw_rmse,
                args.max_yaw_rmse is None or yaw_rmse <= args.max_yaw_rmse,
            ),
        )
        if limit is not None
    ]
    report = {
        "task": metadata["task"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "steps": args.steps,
        "num_envs": args.num_envs,
        "velocity_command": args.velocity,
        "seed": args.seed,
        "survival_fraction": survival,
        "return": rewards.tolist(),
        "observed_seconds": (counts * CTRL_DT).tolist(),
        "peak_joint_torque": torque_peak,
        "planar_rmse": planar_rmse,
        "yaw_rmse": yaw_rmse,
        "mean_velocity": (velocity_sum / counts[:, None]).tolist(),
        "accepted": all(checks) if checks else None,
        "limits": {
            "min_survival": args.min_survival,
            "max_planar_rmse": args.max_planar_rmse,
            "max_yaw_rmse": args.max_yaw_rmse,
        },
        "identical_initial_states": True,
        "reset_protocol": "deterministic nominal pose, no training randomization",
    }
    write_json(args.output / "evaluation.json", report)
    print(json.dumps(report, indent=2))
    return 2 if report["accepted"] is False else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Native H1 MuJoCo/mjbatch PPO; no IsaacLab required"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument(
            "--num-envs", type=int, default=64 if command == "train" else 1
        )
        sub.add_argument("--threads", type=int, default=4)
        sub.add_argument("--seed", type=int, default=0)
        if command == "train":
            source = sub.add_mutually_exclusive_group(required=True)
            source.add_argument(
                "--model",
                type=Path,
                help="H1 MJCF, with accessible referenced mesh files",
            )
            source.add_argument(
                "--resume", type=Path, help="Complete native run directory"
            )
            sub.add_argument("--updates", type=int, default=1000)
            sub.add_argument("--horizon", type=int, default=24)
            sub.add_argument("--learning-rate", type=float, default=1e-3)
        else:
            sub.add_argument("--run", type=Path, required=True)
            sub.add_argument("--steps", type=int, default=500)
            sub.add_argument("--velocity", type=float, nargs=3, default=[0.5, 0, 0])
            sub.add_argument("--record-motion", action="store_true")
            sub.add_argument("--min-survival", type=float)
            sub.add_argument("--max-planar-rmse", type=float)
            sub.add_argument("--max-yaw-rmse", type=float)
    args = parser.parse_args(argv)
    import math

    if (
        any(
            getattr(args, key, 1) <= 0
            for key in ("num_envs", "threads", "updates", "horizon", "steps")
        )
        or args.seed < 0
    ):
        parser.error("Counts must be positive and seed nonnegative")
    if args.command == "train" and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        parser.error("Learning rate must be finite and positive")
    if args.command == "evaluate":
        if any(not math.isfinite(v) or abs(v) > 1 for v in args.velocity):
            parser.error("Velocity components must be finite in [-1, 1]")
        if args.min_survival is not None and not 0 <= args.min_survival <= 1:
            parser.error("Minimum survival must be in [0, 1]")
        if any(
            value is not None and (not math.isfinite(value) or value < 0)
            for value in (args.max_planar_rmse, args.max_yaw_rmse)
        ):
            parser.error("RMSE limits must be finite and nonnegative")
        if args.record_motion and args.steps > 10000:
            parser.error("Motion recording is bounded to 10000 steps")
    result = train(args) if args.command == "train" else evaluate(args)
    if result:
        raise SystemExit(result)


if __name__ == "__main__":
    main()
