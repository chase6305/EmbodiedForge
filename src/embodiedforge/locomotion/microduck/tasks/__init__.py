"""Flat walking registration adapted from Microduck; see ../NOTICE."""

import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from rsl_rl.algorithms import PPO

from ..checkpoint import atomic_save
from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg


class MicroduckPPO(PPO):
    """Use SDK PPO, rejecting nonfinite targets before the policy update."""

    def compute_returns(self, obs):
        super().compute_returns(obs)
        storage = self.storage
        if not (
            torch.isfinite(storage.advantages).all()
            & torch.isfinite(storage.returns).all()
        ).item():
            raise RuntimeError(
                "Nonfinite PPO advantages or returns; stopping before policy update"
            )


class MicroduckOnPolicyRunner(MjlabOnPolicyRunner):
    """Save training state; ONNX export is an explicit CLI operation."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # Preserve the upstream YAML fix without changing PPO's symmetry state.
        algorithm = train_cfg.get("algorithm", {})
        symmetry = (
            algorithm.get("symmetry_cfg") if isinstance(algorithm, dict) else None
        )
        if isinstance(symmetry, dict) and "_env" in symmetry:
            algorithm["symmetry_cfg"] = {
                k: v for k, v in symmetry.items() if k != "_env"
            }

    def save(self, path, infos=None):
        # The CLI uses local TensorBoard logging. Keep SDK upload semantics for
        # externally configured loggers, which may upload during super().save().
        if self.logger.logger_type != "tensorboard":
            return super().save(path, infos)
        return atomic_save(super().save, path, infos)


register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)
