"""Executed only by the selected Weave Python, never imported by the core CLI."""

import importlib.metadata
import importlib.util
import json
import math
import sys
from pathlib import Path

from .h1 import sha256, write_json
from .recipes import validate_snapshot


def synchronize_resume_optimizer(algorithm):
    """Rebind Weave's flattened groups after Torch replaces child groups on load."""
    optimizer = algorithm.optimizer
    groups = [group for child in optimizer.optimizers for group in child.param_groups]
    rates = {group["lr"] for group in groups}
    if len(rates) != 1 or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError(
            "Weave resume requires a common finite positive optimizer learning rate"
        )
    optimizer.param_groups = groups
    algorithm.learning_rate = rates.pop()


def validate_resume_optimizer(optimizer, state):
    """Validate the pinned Muon/AdamW state before loading any parameters."""
    import torch

    children = state.get("optimizers") if isinstance(state, dict) else None
    if not isinstance(children, list) or len(children) != len(optimizer.optimizers):
        raise ValueError("Weave resume requires all Muon / AdamW optimizer states")
    if state.get("class") != type(optimizer).__name__:
        raise ValueError("Weave checkpoint optimizer class differs")
    rates = set()
    for child, saved in zip(optimizer.optimizers, children, strict=True):
        groups = saved.get("param_groups") if isinstance(saved, dict) else None
        if not isinstance(groups, list) or len(groups) != len(child.param_groups):
            raise ValueError("Weave optimizer parameter groups differ")
        parameters = {}
        for group, live in zip(groups, child.param_groups, strict=True):
            ids = group.get("params") if isinstance(group, dict) else None
            if not isinstance(ids, list) or len(ids) != len(live["params"]):
                raise ValueError("Weave optimizer parameter count differs")
            rate = group.get("lr")
            if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
                raise ValueError("Invalid Weave optimizer learning rate")
            rates.add(rate)
            for name, expected in live.items():
                if name not in ("params", "lr") and group.get(name) != expected:
                    raise ValueError(f"Weave optimizer setting differs: {name}")
            for key, parameter in zip(ids, live["params"], strict=True):
                if type(key) is not int or key < 0 or key in parameters:
                    raise ValueError(
                        "Invalid or duplicate Weave optimizer parameter ID"
                    )
                parameters[key] = parameter
        slots = saved.get("state")
        if (
            not isinstance(slots, dict)
            or any(type(key) is not int for key in slots)
            or set(slots) != set(parameters)
        ):
            raise ValueError(
                "Weave optimizer state is incomplete or has extra parameters"
            )
        if isinstance(child, torch.optim.Muon):
            buffers = {"momentum_buffer"}
        elif isinstance(child, torch.optim.AdamW):
            buffers = {"exp_avg", "exp_avg_sq", "step"}
        else:
            raise ValueError(f"Unsupported Weave optimizer: {type(child).__name__}")
        for key, parameter in parameters.items():
            values = slots[key]
            if not isinstance(values, dict) or set(values) != buffers:
                raise ValueError("Invalid Weave optimizer buffers")
            for name, value in values.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid Weave optimizer tensor: {name}")
                if name == "step":
                    if (
                        value.ndim != 0
                        or not value.is_floating_point()
                        or value < 1
                        or value != value.floor()
                    ):
                        raise ValueError("Invalid Weave optimizer step")
                elif value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise ValueError(
                        f"Weave optimizer tensor shape/dtype differs: {name}"
                    )
                elif name == "exp_avg_sq" and (value < 0).any():
                    raise ValueError("Negative Weave optimizer second moment")
    if len(rates) != 1:
        raise ValueError("Weave resume requires a common optimizer learning rate")


def validate_policy_state(policy, state):
    """Reject incompatible weights and invalid normalizers before copying them."""
    import torch

    expected = policy.state_dict()
    if not isinstance(state, dict) or state.keys() != expected.keys():
        raise ValueError("Weave policy state keys differ")
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != expected[name].shape
            or value.dtype != expected[name].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid Weave policy tensor: {name}")
        if (
            "_obs_normalizer." in name
            and name.rsplit(".", 1)[-1] in ("_var", "_std", "count")
            and (value < 0).any()
        ):
            raise ValueError(f"Negative Weave normalization state: {name}")
        if name == "std" and (value <= 0).any():
            raise ValueError("Weave policy standard deviation must be positive")


def read_checkpoint(runner, path, *, resume):
    """Read and validate saved state on CPU without changing the runner."""
    # Keep unused inference optimizer buffers off GPU and validate resume buffers
    # before allocating GPU state. Torch copies loaded tensors to live devices.
    import torch

    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid Weave checkpoint")
    optimizer = runner.alg.optimizer
    saved = checkpoint.get("optimizer_state_dict")
    policy = checkpoint.get("model_state_dict")
    validate_policy_state(runner.alg.policy, policy)
    if resume:
        validate_resume_optimizer(optimizer, saved)
    iteration = checkpoint.get("iter")
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Weave checkpoint requires a nonnegative saved iteration")
    return checkpoint


def load_checkpoint(runner, path, *, resume):
    """Copy validated saved state onto the runner's selected device."""
    checkpoint = read_checkpoint(runner, path, resume=resume)
    # The pinned G1 task has no RND module or auxiliary optimizer.
    runner.alg.policy.load_state_dict(checkpoint["model_state_dict"])
    if resume:
        runner.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        synchronize_resume_optimizer(runner.alg)
    # RSL-RL 3.1.2 saves the last completed zero-based iteration, but learn()
    # starts at current_learning_iteration rather than the next iteration.
    runner.current_learning_iteration = checkpoint["iter"] + int(resume)


def training_result(runner, output, first_iteration, iterations):
    """Report completion only for the requested budget and a valid saved state."""
    final = first_iteration + iterations - 1
    if runner.current_learning_iteration != final:
        raise RuntimeError("Weave training did not complete the requested iterations")
    path = output / f"model_{final}.pt"
    checkpoint = read_checkpoint(runner, path, resume=True)
    if checkpoint["iter"] != final:
        raise ValueError("Weave final checkpoint iteration differs from the runner")
    state = checkpoint["model_state_dict"]
    return {
        "status": "complete",
        "first_iteration": first_iteration,
        "last_iteration": final,
        "iterations": iterations,
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": sha256(path),
        "policy_parameters": sum(p.numel() for p in runner.alg.policy.parameters()),
        "policy_state_tensor_bytes": sum(
            v.numel() * v.element_size() for v in state.values()
        ),
    }


def close_resource(resource, name):
    """Close from a finally block without replacing an existing error."""
    if resource is None:
        return
    active_error = sys.exc_info()[0] is not None
    try:
        resource.close()
    except Exception as exc:
        if not active_error:
            raise
        print(f"Failed to close {name}: {exc}", file=sys.stderr)


def learn_policy(runner, iterations):
    """Drain TensorBoard on normal exit and errors, including KeyboardInterrupt."""
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)
    finally:
        close_resource(getattr(runner, "writer", None), "training log")


def evaluate_policy(policy, env, app, collector, output, inputs):
    """Retain first-episode results if stepping raises or the user interrupts."""
    try:
        obs = env.get_observations()
        for _ in range(int(max(collector.lengths)) + 1):
            if not app.is_running() or collector.complete:
                break
            obs, _, _, _ = env.step(policy(obs))
    except BaseException:
        try:
            report = collector.report()
            report["inputs"] = inputs
            write_json(output / "result.json", report)
        except Exception as exc:
            # A full disk or failed report must not replace the stepping error
            # or convert a user interruption into a different exit cause.
            print(f"Failed to save evaluation report: {exc}", file=sys.stderr)
        raise
    return collector.report()


def check_runtime(request):
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            "Weave requires an isolated Python 3.11 / Isaac Sim 5.1 environment"
        )
    dependencies = ["isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy"]
    if request["command"] == "export":
        dependencies.extend(("onnx", "onnxscript"))
    try:
        versions = {name: importlib.metadata.version(name) for name in dependencies}
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Missing Weave SDK dependency: {exc.name}; install it in the --python environment"
        ) from exc
    binary_version = Path(request["isaaclab_root"]) / "_isaac_sim/VERSION"
    versions["isaacsim"] = (
        binary_version.read_text().strip()
        if binary_version.is_file()
        else importlib.metadata.version("isaacsim")
    )
    if not versions["isaacsim"].startswith("5.1.") or versions["rsl-rl-lib"] != "3.1.2":
        raise RuntimeError(
            f"Require Isaac Sim 5.1.x and rsl-rl-lib 3.1.2; found {versions}"
        )
    if int(versions["numpy"].split(".")[0]) >= 2:
        raise RuntimeError("Weave SDK requires numpy<2")
    for module in ("isaaclab", "isaaclab_rl"):
        origin = Path(importlib.util.find_spec(module).origin).resolve()
        if not origin.is_relative_to(Path(request["isaaclab_root"]).resolve()):
            raise RuntimeError(
                f"{module} is installed from a different checkout: {origin}"
            )
    import torch

    torch_version = tuple(
        int(part) for part in versions["torch"].split("+")[0].split(".")[:2]
    )
    if torch_version < (2, 10) or not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Weave requires torch.optim.Muon (Torch >= 2.10)")
    return versions


def attach_evaluator(env_cfg, collector):
    import torch
    from g1_hoi_learning.tasks.hoi.mdp.terminations import motion_clip_end
    from isaaclab.managers import RewardTermCfg, TerminationTermCfg

    def collect_tracking(env):
        motion = env.command_manager.get_term("motion")
        motion._update_metrics()
        manager = env.termination_manager

        def cpu(tensor):
            return tensor.detach().cpu().numpy()

        # Batch transfers by dtype: three GPU-to-CPU synchronizations per step,
        # independent of the number of metrics and termination terms.
        metric_names = list(motion.metrics)
        metrics = cpu(torch.stack([motion.metrics[key] for key in metric_names]))
        term_names = manager.active_terms
        flags = cpu(
            torch.stack(
                [manager.terminated, manager.time_outs]
                + [manager.get_term(key) for key in term_names]
            )
        )
        terms = dict(zip(term_names, flags[2:], strict=True))
        collector.update(
            dict(zip(metric_names, metrics, strict=True)),
            cpu(motion.time_steps),
            flags[0],
            flags[1],
            terms["clip_end"],
            terms,
        )
        return motion.metrics["error_anchor_pos"].new_zeros(env.num_envs)

    env_cfg.terminations.clip_end = TerminationTermCfg(
        func=motion_clip_end,
        time_out=True,
        params={"command_name": "motion"},
    )
    # Reward computation follows termination and precedes reset. A recorder would
    # trigger an extra observation computation, changing noise RNG and doing the
    # object geometry queries twice. Weight must be nonzero so the SDK calls us;
    # the term itself returns zero and contributes no reward.
    env_cfg.rewards.ef_tracking = RewardTermCfg(func=collect_tracking, weight=1.0)


def export_policy(runner, env, output):
    """Export after checking the original device; leave this terminal policy on CPU."""
    import torch
    from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx

    policy = runner.alg.policy
    policy.eval()
    obs = env.get_observations()
    groups = policy.obs_groups["policy"]
    flat = torch.cat([obs[key][:1] for key in groups], dim=-1).cpu()
    with torch.inference_mode():
        expected = policy.act_inference(obs)[:1].cpu()
    # SDK exporters deepcopy the actor before moving it to CPU. Move the source
    # first to avoid a transient second actor allocation on the simulation GPU.
    # This is the final operation on the runner; inference above stays on device.
    policy.cpu()
    export_policy_as_jit(policy, policy.actor_obs_normalizer, str(output))
    export_policy_as_onnx(policy, str(output), policy.actor_obs_normalizer)
    with torch.inference_mode():
        actual = torch.jit.load(str(output / "policy.pt"))(flat)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    import onnx
    from onnx.reference import ReferenceEvaluator

    onnx.checker.check_model(str(output / "policy.onnx"))
    model = onnx.load(str(output / "policy.onnx"))
    reference = ReferenceEvaluator(model)
    onnx_actions = reference.run(None, {reference.input_names[0]: flat.numpy()})[0]
    onnx_actual = torch.from_numpy(onnx_actions)
    torch.testing.assert_close(onnx_actual, expected, atol=1e-5, rtol=1e-4)
    action = env.unwrapped.action_manager.get_term("joint_pos")

    def action_value(value):
        if isinstance(value, (int, float)):
            return value
        return (value[0] if value.ndim == 2 else value).tolist()

    metadata = {
        "observation_groups": [
            {"name": key, "size": obs[key].shape[-1]} for key in groups
        ],
        "normalization_included": True,
        "input_shape": list(flat.shape),
        "input_dtype": str(flat.dtype),
        "action_joint_names": action._joint_names,
        "action_scale": action_value(action._scale),
        "action_offset": action_value(action._offset),
        "mimic_joints": action.cfg.mimic,
        "control_dt": env.unwrapped.step_dt,
        "clip_actions": runner.cfg.get("clip_actions"),
        "actor_parameters": sum(p.numel() for p in policy.actor.parameters()),
        "actor_critic_parameters": sum(p.numel() for p in policy.parameters()),
        "jit_parity_max_abs_error": float((actual - expected).abs().max()),
        "onnx_parity_max_abs_error": float((onnx_actual - expected).abs().max()),
        "onnx_validation": "Structural and ONNX reference evaluator parity; target deployment runtime not validated",
        "deployment_scope": "Policy only; requires Weave reference, object sensing, observation and joint adapters",
        "artifacts": {
            name: {
                "bytes": (output / name).stat().st_size,
                "sha256": sha256(output / name),
            }
            for name in [
                "policy.pt",
                *sorted(path.name for path in output.glob("policy.onnx*")),
            ]
        },
    }
    write_json(output / "policy.json", metadata)
    return metadata


def execute(request, app):
    import g1_hoi_learning  # noqa: F401 -- register task and algorithm after AppLauncher
    import gymnasium as gym
    import torch
    import yaml
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils import load_cfg_from_registry
    from rsl_rl.runners import OnPolicyRunner

    from ._weave_evaluation import ClipEvaluation
    from .locomotion.microduck.checkpoint import atomic_save

    class Runner(OnPolicyRunner):
        def save(self, path, infos=None):
            atomic_save(super().save, path, infos)

    output = Path(request["output"])
    env_cfg = load_cfg_from_registry(request["task"], "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(request["task"], "rsl_rl_cfg_entry_point")
    config = yaml.safe_load(
        (Path(request["weave_root"]) / "configs/track/train.yaml").read_text()
    )
    env_cfg.from_dict(config["env"])
    agent_cfg.from_dict(config["agent"])
    motions = request["motions"]
    inputs = {"motions": motions["files"], "checkpoint": request["checkpoint"]}
    env_cfg.commands.motion.motion_files = [entry["path"] for entry in motions["files"]]
    env_cfg.seed = agent_cfg.seed = request["seed"]
    env_cfg.sim.device = agent_cfg.device = request["device"]
    env_cfg.commands.motion.debug_vis = False
    agent_cfg.logger = "tensorboard"
    collector = None
    if request["command"] == "train":
        # Match the pinned scripts/rsl_rl/train.py numerical backend settings.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        env_cfg.scene.num_envs = request["num_envs"]
        agent_cfg.max_iterations = request["iterations"]
        agent_cfg.save_interval = request["save_interval"]
    else:
        agent_cfg.policy.compile = False
        # RSL-RL allocates PPO rollout storage even for inference-only runners.
        # Evaluation/export never collect training rollouts; keep one slot.
        agent_cfg.num_steps_per_env = 1
        env_cfg.commands.motion.rsi = False
        env_cfg.scene.num_envs = 1
        if request["command"] == "evaluate":
            env_cfg.scene.num_envs = len(motions["names"])
            env_cfg.commands.motion.eval_mode = True
            env_cfg.episode_length_s = max(motions["lengths"]) / 50 + 1
            collector = ClipEvaluation(motions["names"], motions["lengths"])
    if collector:
        attach_evaluator(env_cfg, collector)
    env = gym.make(request["task"], cfg=env_cfg)
    try:
        dump_yaml(str(output / "env.yaml"), env_cfg)
        dump_yaml(str(output / "agent.yaml"), agent_cfg)
        robot = env.unwrapped.scene["robot"]
        for key, actual in (
            ("joint_names", robot.joint_names),
            ("body_names", robot.body_names),
        ):
            if list(actual) != motions["layout"][key]:
                raise ValueError(
                    f"Motion {key} do not match the spawned G1 asset order"
                )
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = Runner(
            env, agent_cfg.to_dict(), log_dir=str(output), device=request["device"]
        )
        if request["checkpoint"]:
            load_checkpoint(
                runner,
                request["checkpoint"]["path"],
                resume=request["command"] == "train",
            )
        if request["command"] == "train":
            first_iteration = runner.current_learning_iteration
            learn_policy(runner, request["iterations"])
            result = training_result(
                runner, output, first_iteration, request["iterations"]
            )
            result["torch_backends"] = {
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            }
        elif request["command"] == "evaluate":
            policy = runner.get_inference_policy(device=env.device)
            with torch.inference_mode():
                result = evaluate_policy(policy, env, app, collector, output, inputs)
        else:
            result = {
                "status": "complete",
                "policy": export_policy(runner, env, output),
            }
        result["inputs"] = inputs
        write_json(output / "result.json", result)
        if result["status"] != "complete":
            raise RuntimeError(
                "Evaluation stopped before all clips completed; see result.json"
            )
    finally:
        close_resource(env, "environment")


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    sys.argv = sys.argv[:1]
    # Historical requests did not record an implementation snapshot.
    if "implementation" in request:
        package = Path(request["output"]) / "implementation" / "embodiedforge"
        if Path(__file__).resolve().parent != package.resolve():
            raise ValueError("Weave worker must load its implementation snapshot")
        validate_snapshot(
            package, request["implementation"], label="Weave implementation"
        )
    versions = check_runtime(request)
    for entry in [
        *request["motions"]["files"],
        *([request["checkpoint"]] if request["checkpoint"] else []),
    ]:
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"Input snapshot changed: {entry['path']}")
    write_json(Path(request["output"]) / "runtime.json", versions)
    from isaaclab.app import AppLauncher

    app = AppLauncher(headless=request["headless"], device=request["device"]).app
    try:
        execute(request, app)
    finally:
        close_resource(app, "simulation app")


if __name__ == "__main__":
    main()
