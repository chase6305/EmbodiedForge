"""Managed adapters for Apache-2.0 mjbatch Go1 PPO and predictive-sampling MPC.

Go1 task and PPO live in embodiedforge.locomotion. MPC and arm tasks still import
pinned upstream examples. This module owns run artifacts and bounded evaluation.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from embodiedforge._training_runtime import record_training_runtime
from embodiedforge.recipes import write_json


def load_example(request, name):
    from mjbatch import Batch

    sys.path.insert(0, str(Path(request["project"]) / "examples"))
    owner = importlib.import_module(name)

    # The upstream example always selects every logical CPU. Bound thread count
    # avoids oversubscription alongside the Torch learner and other applications.
    class ThreadedBatch(Batch):
        def __init__(self, model, count):
            super().__init__(model, count, num_threads=request["threads"])

    owner.Batch = ThreadedBatch
    return owner


def check_finite(*values):
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("Non-finite mjbatch state, policy output or metric")


def go1_train(request, owner):
    import torch

    from embodiedforge._wuji_recipe import finite_tensors
    from embodiedforge.locomotion import go1_ppo as learner

    torch.set_num_threads(request["threads"])
    torch.manual_seed(request["seed"])
    configuration = {}
    for name in (
        "TIMESTEP",
        "DECIMATION",
        "KP",
        "KD",
        "CTRL_DT",
        "ACTION_SCALE",
        "COMMAND_RANGE",
        "COMMAND_ON",
        "COMMAND_SECONDS",
        "FRICTION",
        "REWARD",
        "GAMMA",
        "LAMBDA",
        "CLIP",
        "LR",
        "LR_END",
        "LR_TO",
        "EPOCHS",
        "MINIBATCHES",
        "ENT_COEF",
        "LOG_STD",
        "HIDDEN",
        "OBS_DIM",
        "ACT_DIM",
    ):
        value = getattr(owner if hasattr(owner, name) else learner, name)
        configuration[name] = value.tolist() if isinstance(value, np.ndarray) else value
    configuration.update(
        learning_rate_override=request["go1_learning_rate"],
        COMMAND_RANGE=list(
            owner.COMMAND_PROFILES[request["go1_command_profile"]]["range"]
        ),
        COMMAND_ON=list(owner.COMMAND_PROFILES[request["go1_command_profile"]]["on"]),
        COMMAND_SECONDS=owner.COMMAND_PROFILES[request["go1_command_profile"]][
            "seconds"
        ],
        command_profile=request["go1_command_profile"],
        COMMAND_STOP_PROBABILITY=owner.COMMAND_PROFILES[
            request["go1_command_profile"]
        ].get("stop_probability", 0.0),
        COMMAND_KEEP=owner.COMMAND_PROFILES[request["go1_command_profile"]].get(
            "keep", 0.5
        ),
        REWARD=owner.REWARD_PROFILES[request["go1_reward_profile"]],
        reward_profile=request["go1_reward_profile"],
        NUM_ENVS=request["num_envs"],
        HORIZON=request["horizon"],
        EPISODE=500,
        implementation="embodiedforge.locomotion.go1+go1_ppo",
        task_semantics=request["go1_semantics"],
        seed=request["seed"],
        learner="cpu",
        physics_threads=request["threads"],
        optimizer="Adam (non-fused)",
        updates=request["updates"],
    )
    if request["go1_learning_rate"] is not None:
        configuration.update(
            LR=request["go1_learning_rate"], LR_END=request["go1_learning_rate"]
        )
    write_json(Path("recipe-config.json"), configuration)
    env = owner.Go1(
        request["num_envs"],
        seed=request["seed"],
        num_threads=request["threads"],
        semantics=request["go1_semantics"],
        reward_profile=request["go1_reward_profile"],
        command_profile=request["go1_command_profile"],
    )
    net = learner.ActorCritic()
    record_training_runtime(
        Path.cwd(),
        environment=env,
        task=env,
        learner=learner.update,
        physics_adapter=env.batch,
        physics="mujoco/mjbatch",
        core_vector_env=False,
        packages=["numpy", "torch", "mujoco", "mjbatch", "mujoco-menagerie"],
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=learner.LR)
    first_iteration = request.get("start_iteration", 0)
    if request.get("checkpoint"):
        previous = torch.load(
            request["checkpoint"], weights_only=True, map_location="cpu"
        )
        finite_tensors(previous)
        if (
            previous.get("learning_rate_override")
            != request["input_learning_rate_override"]
        ):
            raise ValueError("Go1 checkpoint learning rate differs from input run")
        if (
            previous.get("command_profile", "original")
            != request["input_command_profile"]
        ):
            raise ValueError("Go1 checkpoint command profile differs from input run")
        if (
            previous.get("reward_profile", "original")
            != request["input_reward_profile"]
        ):
            raise ValueError("Go1 checkpoint reward profile differs from input run")
        if (
            previous.get("task_semantics", "upstream-v1")
            != request["input_task_semantics"]
        ):
            raise ValueError("Go1 checkpoint task semantics differs from input run")
        if previous["iteration"] + 1 != first_iteration:
            raise ValueError("Resume iteration does not match the input model")
        net.load_state_dict(previous["model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
    start = time.perf_counter()
    history = []
    with Path("metrics.jsonl").open("w") as stream:
        for offset in range(request["updates"]):
            iteration = first_iteration + offset
            optimizer.param_groups[0]["lr"] = float(
                request["go1_learning_rate"]
                if request["go1_learning_rate"] is not None
                else np.interp(
                    iteration, (0, learner.LR_TO), (learner.LR, learner.LR_END)
                )
            )
            batch, stats = learner.rollout(
                net, env, horizon=request["horizon"], record_policy=True
            )
            finite_tensors(batch)
            diagnostics = learner.update(
                net, optimizer, batch, *learner.gae(batch), diagnostics=True
            )
            finite_tensors(net.state_dict())
            row = {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "iteration": iteration,
                "elapsed_seconds": time.perf_counter() - start,
                "transitions": (offset + 1) * request["num_envs"] * request["horizon"],
                **{key: float(value) for key, value in stats.items()},
                **diagnostics,
            }
            row["weighted_reward_terms"] = sum(
                env.reward_weights[key] * row[key] for key in env.reward_weights
            )
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            history.append(row)
            if (iteration + 1) % 25 == 0 or offset == request["updates"] - 1:
                temporary = Path("model.pt.tmp")
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": iteration,
                        "recipe": "go1-joystick",
                        "learning_rate_override": request["go1_learning_rate"],
                        "reward_profile": request["go1_reward_profile"],
                        "command_profile": request["go1_command_profile"],
                        "task_semantics": request["go1_semantics"],
                        "seed": request["seed"],
                    },
                    temporary,
                )
                temporary.replace("model.pt")
                print(json.dumps(row, allow_nan=False), flush=True)
    checkpoint = torch.load("model.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if (
        checkpoint["iteration"] != first_iteration + request["updates"] - 1
        or len(history) != request["updates"]
    ):
        raise ValueError("Incomplete Go1 training artifacts")
    tensors = finite_tensors(checkpoint)
    return {
        "checkpoint": "model.pt",
        "learning_rate_override": request["go1_learning_rate"],
        "reward_profile": request["go1_reward_profile"],
        "command_profile": request["go1_command_profile"],
        "task_semantics": request["go1_semantics"],
        "completed_updates": request["updates"],
        "checkpoint_iteration": checkpoint["iteration"],
        "finite_tensor_count": tensors,
        "action_dim": owner.ACT_DIM,
        "policy_observation_dim": owner.OBS_DIM,
        "latest_metrics": history[-1],
        "physics": "mjbatch_cpu",
        "learner": "cpu",
        "behavior_validated": False,
    }


def go1_evaluate_case(request, owner):
    import torch

    from embodiedforge._go1_metrics import SWITCHING_SUITES
    from embodiedforge._h1_metrics import FirstEpisodeMetrics
    from embodiedforge._wuji_recipe import finite_tensors

    torch.set_num_threads(request["threads"])
    from embodiedforge.locomotion.go1_ppo import ActorCritic

    env = owner.Go1(
        request["num_envs"],
        seed=request["seed"],
        num_threads=request["threads"],
        episode_steps=request["steps"] + 1,
        semantics=request["go1_semantics"],
        reward_profile=request["go1_reward_profile"],
        command_profile=request["go1_command_profile"],
    )
    net = ActorCritic().eval()
    data = torch.load(request["checkpoint"], weights_only=True, map_location="cpu")
    finite_tensors(data)
    if data.get("learning_rate_override") != request["input_learning_rate_override"]:
        raise ValueError("Go1 checkpoint learning rate differs from input run")
    if data.get("command_profile", "original") != request["input_command_profile"]:
        raise ValueError("Go1 checkpoint command profile differs from input run")
    if data.get("reward_profile", "original") != request["input_reward_profile"]:
        raise ValueError("Go1 checkpoint reward profile differs from input run")
    if data.get("task_semantics", "upstream-v1") != request["input_task_semantics"]:
        raise ValueError("Go1 checkpoint task semantics differs from input run")
    net.load_state_dict(data["model_state_dict"], strict=True)
    metrics = FirstEpisodeMetrics(request["num_envs"], request["steps"], owner.CTRL_DT)
    switching = request.get("suite") in SWITCHING_SUITES
    if switching:
        from embodiedforge._go1_metrics import SwitchingMetrics

        metrics = SwitchingMetrics(
            request["num_envs"], request["steps"], request["suite"]
        )
    command = np.broadcast_to(
        request["velocity"] or [0.0, 0.0, 0.0], (request["num_envs"], 3)
    )
    motion = None
    if request.get("record_motion"):
        from embodiedforge._h1_motion import MotionRecorder

        model = env.batch.model
        body_names = [model.body(i).name for i in range(1, model.nbody)]
        if (
            request["steps"] * ((model.nbody + 3) * 7 + owner.ACT_DIM) * 4
            > 256 * 1024**2
        ):
            raise ValueError("Motion trace exceeds the 256 MiB array budget")
        positions, quaternions = env.batch.bind("xpos"), env.batch.bind("xquat")
        foot_ids = [model.site(name).id for name in owner.FEET]
        site_matrices = env.batch.bind("site_xmat")
        body_count = len(body_names)
        metadata = {
            "schema": 1,
            "case": request.get("case", "custom"),
            "seed": request["seed"],
            "env_id": 0,
            "velocity_command": None if switching else list(request["velocity"]),
            "dt": owner.CTRL_DT,
            "body_names": body_names + [name + "_foot_site" for name in owner.FEET],
            "joint_names": [model.joint(i).name for i in range(1, model.njnt)],
            "edges": [
                [int(model.body_parentid[i]) - 1, i - 1]
                for i in range(1, model.nbody)
                if model.body_parentid[i] > 0
            ]
            + [
                [int(model.site_bodyid[site]) - 1, body_count + i]
                for i, site in enumerate(foot_ids)
            ],
            "point_types": ["body"] * body_count + ["site"] * 4,
            "quaternion_order": "xyzw",
            "world_frame": "MuJoCo world, Z up",
            "title": "Go1 运动回放",
            "view_height": 0.25,
            "view_scale": 350,
        }
        if switching:
            metadata["command_schedule"] = metrics.schedule
        motion = MotionRecorder(metadata)
    with torch.inference_mode():
        for _ in range(request["steps"]):
            if switching:
                command = metrics.command()
            env.command[:], env.until[:] = command, 2
            obs = env.obs()
            action = net.action_mean(torch.as_tensor(obs)).numpy()
            check_finite(obs, action)
            _, _, fell, _ = env.step(action)
            # mj_step derived sensors lag by one physics substep. Refresh before
            # measuring velocity; there is no autoreset in this evaluation.
            env.batch.forward()
            velocity = np.column_stack((env.vel[:, :2], env.gyro[:, 2]))
            if motion is not None and metrics.active[0]:
                import mujoco

                foot_quaternions = np.empty((4, 4))
                for i, site in enumerate(foot_ids):
                    mujoco.mju_mat2Quat(
                        foot_quaternions[i], site_matrices[0, site].ravel()
                    )
                motion.append(
                    (metrics.steps + 1) * owner.CTRL_DT,
                    np.concatenate(
                        (positions[0, 1:], np.stack(env.foot_pos, axis=1)[0])
                    ),
                    np.roll(
                        np.concatenate((quaternions[0, 1:], foot_quaternions)),
                        -1,
                        axis=-1,
                    ),
                    env.qpos[0, owner.JOINTS],
                    bool(fell[0]),
                    False,
                )
            metrics.update(velocity, command, fell, np.zeros(request["num_envs"], bool))
            if not metrics.active.any():
                break
    result = {
        **metrics.report(),
        "learning_rate_override": request["go1_learning_rate"],
        "reward_profile": request["go1_reward_profile"],
        "command_profile": request["go1_command_profile"],
        "task_semantics": request["go1_semantics"],
        "seed": request["seed"],
        "command": None if switching else list(request["velocity"]),
        "protocol": (
            "first_episode_switching_command_no_autoreset_v1"
            if switching
            else "first_episode_fixed_command_no_autoreset_v1"
        ),
        "velocity_frame": "trunk_local_linear_and_angular",
        "termination": "up_z_below_zero (upstream)",
        "acceptance": None,
    }
    result["case"] = request.get("case", "custom")
    if motion is not None:
        from embodiedforge._h1_motion import render_motion
        from embodiedforge.recipes import sha256

        path = Path(f"motion-seed-{request['seed']}-{result['case']}.npz")
        motion.save(path)
        render_motion(path, path.with_suffix(".html"))
        result["motion"] = {
            "path": str(path),
            "sha256": sha256(path),
            "html": str(path.with_suffix(".html")),
        }
    return result


def go1_evaluate(request, owner):
    from embodiedforge._go1_metrics import evaluation_commands

    commands = evaluation_commands(request.get("suite"), request.get("velocity"))
    seeds = request.get("seeds") or [request["seed"]]
    cases = []
    for seed in seeds:
        for name, command in commands.items():
            case = go1_evaluate_case(
                {**request, "seed": seed, "velocity": command, "case": name}, owner
            )
            cases.append(case)
            write_json(Path(f"evaluation-seed-{seed}-{name}.json"), case)
    return (
        {"cases": cases, "suite": request.get("suite"), "seeds": seeds}
        if request.get("suite") or request.get("seeds")
        else cases[0]
    )


def cartpole_solve(request):
    import mujoco

    owner = load_example(request, "cartpole_mpc")
    model = mujoco.MjModel.from_xml_path(str(owner.MODEL))
    data, baseline = mujoco.MjData(model), mujoco.MjData(model)
    for item in (data, baseline):
        mujoco.mj_resetDataKeyframe(model, item, model.key("hang").id)
        mujoco.mj_forward(model, item)
    planner = owner.Batch(model, request["num_envs"])
    qpos, qvel, ctrl, now = (
        planner.bind(field) for field in ("qpos", "qvel", "ctrl", "time")
    )
    rng = np.random.default_rng(request["seed"])
    horizon, count = request["horizon"], request["num_envs"]
    plan, scale = np.zeros((horizon, model.nu)), 1.0
    records, baseline_cost = [], []
    start = time.perf_counter()
    for _ in range(request["steps"]):
        qpos[:], qvel[:], now[:] = data.qpos, data.qvel, data.time
        candidates = np.clip(
            plan + scale * owner.SIGMA * rng.normal(size=(count, horizon, model.nu)),
            -1,
            1,
        )
        candidates[0] = plan
        total = np.zeros(count)
        for knot in range(horizon):
            ctrl[:] = candidates[:, knot]
            planner.step(nstep=owner.SUB)
            state = np.concatenate((qpos, qvel), axis=1)
            total += owner.cost(state, candidates[:, knot])
        check_finite(total, state)
        best = int(np.argmin(total))
        scale = np.clip(total[best] / horizon / 0.2, 0.1, 1.0)
        plan = np.roll(candidates[best], -1, axis=0)
        plan[-1] = plan[-2]
        data.ctrl[:] = candidates[best, 0]
        mujoco.mj_step(model, data, nstep=owner.SUB)
        mujoco.mj_step(model, baseline, nstep=owner.SUB)
        state = np.r_[data.qpos, data.qvel]
        check_finite(state, data.ctrl)
        records.append(
            [data.time, *state, *data.ctrl, float(owner.cost(state, data.ctrl))]
        )
        baseline_cost.append(
            float(owner.cost(np.r_[baseline.qpos, baseline.qvel], baseline.ctrl))
        )
    trajectory = np.asarray(records)
    np.savez_compressed(
        "trajectory.npz",
        time=trajectory[:, 0],
        qpos=trajectory[:, 1:3],
        qvel=trajectory[:, 3:5],
        ctrl=trajectory[:, 5:6],
        cost=trajectory[:, 6],
        zero_control_cost=np.asarray(baseline_cost),
    )
    tail = trajectory[
        -min(len(trajectory), round(1 / (model.opt.timestep * owner.SUB))) :
    ]
    return {
        "steps": len(records),
        "control_dt": model.opt.timestep * owner.SUB,
        "seed": request["seed"],
        "rollouts": count,
        "horizon": horizon,
        "elapsed_seconds": time.perf_counter() - start,
        "mean_cost": float(trajectory[:, 6].mean()),
        "zero_control_mean_cost": float(np.mean(baseline_cost)),
        "final_tilt_rad": float(abs(owner.wrap(data.qpos[1]))),
        "final_cart_position_m": float(data.qpos[0]),
        "last_second_max_tilt_rad": float(np.abs(owner.wrap(tail[:, 2])).max()),
        "trajectory": "trajectory.npz",
        "acceptance": None,
    }


def arm_throw_solve(request):
    owner = load_example(request, "arm_throw")
    if request["num_envs"] < 20:
        raise ValueError("Arm CEM needs at least 20 candidates (two elites)")
    owner.POP, owner.GENERATIONS = request["num_envs"], request["generations"]
    population = owner.Throws(owner.POP)
    start = time.perf_counter()
    stock, stock_history = owner.search(
        population, np.random.default_rng(request["seed"]), "throw only", fixed=True
    )
    best, history = owner.search(
        population, np.random.default_rng(request["seed"]), "arm + throw"
    )
    designs = np.array([stock, best])
    results = owner.Throws(2).run(designs, record=True)
    check_finite(designs, history, stock_history, results["distance"], results["path"])
    if (results["score"] <= owner.MISS).any():
        raise ValueError("CEM did not produce two valid forward throws")
    np.savez_compressed(
        "trajectory.npz",
        qpos=results["path"],
        designs=designs,
        stock_history=stock_history,
        codesign_history=history,
        release_position=results["pos"],
        release_velocity=results["vel"],
    )
    return {
        "seed": request["seed"],
        "population": owner.POP,
        "generations": owner.GENERATIONS,
        "elapsed_seconds": time.perf_counter() - start,
        "stock_range_m": float(results["distance"][0]),
        "codesign_range_m": float(results["distance"][1]),
        "range_definition": "ballistic landing x from release state (upstream objective)",
        "stock_design": stock.tolist(),
        "codesign": best.tolist(),
        "trajectory": "trajectory.npz",
        "physics_dt": owner.DT,
        "acceptance": None,
    }


def run(request):
    if request["task"] == "arm-throw-codesign":
        return arm_throw_solve(request)
    if request["task"] == "cartpole-mpc":
        if request["horizon"] < 2:
            raise ValueError("MPC horizon must have at least two knots")
        return cartpole_solve(request)
    from embodiedforge._go1_assets import verified_go1_assets
    from embodiedforge.locomotion import go1 as owner

    assets = verified_go1_assets(request.get("input_assets"))
    write_json(Path("assets.json"), assets)
    result = (
        go1_train(request, owner)
        if request["command"] == "train"
        else go1_evaluate(request, owner)
    )
    result["assets"] = assets
    return result
