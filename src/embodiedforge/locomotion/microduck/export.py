"""Local-checkpoint ONNX export adapted from Microduck's export.py (see NOTICE)."""

import argparse
from dataclasses import asdict
from pathlib import Path


def policy_metadata(env, run_path):
    """Describe the controlled joints in action order without patching the SDK."""
    import torch

    from mjlab.envs.mdp.actions import JointPositionAction

    robot = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    assert isinstance(action, JointPositionAction)
    joint_names = list(action.target_names)
    joint_ids = action.target_ids.cpu().tolist()
    actuator_ids = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
    ctrl_ids = [actuator_ids[name] for name in joint_names]
    defaults = robot.data.default_joint_pos[0, joint_ids].cpu().tolist()
    scale = action.scale
    # The SDK rounds numeric lists to three decimals. Preserve the exact action
    # transform in its existing CSV format, including per-joint scale values.
    return {
        "run_path": run_path,
        "joint_names": joint_names,
        "joint_stiffness": env.sim.mj_model.actuator_gainprm[ctrl_ids, 0].tolist(),
        "joint_damping": (-env.sim.mj_model.actuator_biasprm[ctrl_ids, 2]).tolist(),
        "default_joint_pos": ",".join(str(value) for value in defaults),
        "command_names": list(env.command_manager.active_terms),
        "observation_names": env.observation_manager.active_terms["actor"],
        "action_scale": ",".join(str(value) for value in scale[0].cpu().tolist())
        if isinstance(scale, torch.Tensor)
        else scale,
    }


def main(argv=None):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["Mjlab-Velocity-Flat-MicroDuck"])
    parser.add_argument("--checkpoint-file", type=Path, required=True)
    parser.add_argument("--onnx-file", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    args = parser.parse_args(argv)
    configure_torch_backends()
    cfg = load_env_cfg(args.task, play=True)
    cfg.scene.num_envs = args.num_envs
    agent = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(args.task)(wrapped, asdict(agent), device="cuda:0")
        runner.load(
            str(args.checkpoint_file), load_cfg={"actor": True}, map_location="cpu"
        )
        output = args.onnx_file.resolve()
        runner.export_policy_to_onnx(str(output.parent), output.name)
        metadata = policy_metadata(env, run_path=str(args.checkpoint_file))
        attach_metadata_to_onnx(str(output), metadata)
    finally:
        env.close()
