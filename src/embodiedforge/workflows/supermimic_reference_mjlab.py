"""mjlab physics bridge for the pinned vibed SuperMimic algorithm.

SDK imports stay inside functions. The generated upstream TrackingSim retains
its sampling, reset noise, next-frame analytic rewards and episode accounting.
Only simulator construction, state I/O and PD application are replaced here.
"""

from contextlib import contextmanager
from types import SimpleNamespace

_active = []


def add_launcher_args(parser):
    return parser


@contextmanager
def launch_simulation(config, args):
    del config, args
    try:
        yield
    finally:
        while _active:
            _active.pop().close()


def make_sim(device, args):
    return SimpleNamespace(device=device, seed=args.seed)


def initialize(
    owner, config, count, motion, mdp, gamma, eval_mode, max_steps, reset_noise
):
    import mujoco_warp as mjw
    import numpy as np
    import torch
    from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
        FULL_COLLISION_WITHOUT_SELF,
    )
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.envs.mdp import dr
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.scene import SceneCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg
    from mjlab.utils.spec_config import CollisionCfg
    from supermimic.mdp import MAX_STEP_REWARD
    from supermimic.robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list

    class BridgeSimulationCfg(SimulationCfg):
        def apply_wp_opt(self, wp_opt):
            super().apply_wp_opt(wp_opt)
            # Keep contact/Jacobian capacity warnings. Reaching a configured
            # line-search iteration limit is common and would flood long logs.
            wp_opt.warn_overflow &= ~int(mjw.OverflowType.LS_ITERATIONS)

    robot = get_g1_robot_cfg()
    robot.collisions = (FULL_COLLISION_WITHOUT_SELF,)
    events = {
        "friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)),
                "operation": "abs",
                "ranges": (1.0, 1.0) if eval_mode else (0.3, 1.2),
                "shared_random": True,
            },
        ),
    }
    if not eval_mode:
        events["com"] = EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
            },
        )
    terrain = TerrainEntityCfg(
        terrain_type="plane",
        collisions=(
            CollisionCfg(
                geom_names_expr=("terrain",),
                contype=1,
                conaffinity=1,
                condim=3,
                priority=0,
                friction=(1.0 if eval_mode else 0.3,),
            ),
        ),
    )
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=terrain, num_envs=count, entities={"robot": robot}),
        observations={},
        commands={},
        rewards={},
        terminations={},
        recorders={},
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=G1_ACTION_SCALE,
                use_default_offset=True,
            )
        },
        events=events,
        seed=config.seed,
        auto_reset=False,
        sim=BridgeSimulationCfg(
            nconmax=128,
            njmax=500,
            mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=50),
        ),
        decimation=4,
        episode_length_s=1e9,
    )
    raw = ManagerBasedRlEnv(cfg, device=config.device)
    _active.append(raw)
    raw.reset()
    owner.raw = raw
    owner.device, owner.N = config.device, count
    owner.sim, owner.scene, owner.robot = raw.sim, raw.scene, raw.scene["robot"]
    assert list(owner.robot.joint_names) == G1_JOINT_NAMES
    assert raw.sim.data.qpos.shape == (count, 36) and raw.sim.data.qvel.shape == (
        count,
        35,
    )
    owner.origins = raw.scene.env_origins.clone()
    owner.motion, owner.mdp, owner.gamma = motion, mdp, gamma
    owner.eval_mode, owner.max_episode_steps = (
        eval_mode,
        10**9 if eval_mode else max_steps,
    )
    owner.reset_noise = reset_noise and not eval_mode
    owner.default_q = owner.robot.data.default_joint_pos[0].clone()
    assert torch.allclose(
        owner.default_q,
        torch.tensor(default_joint_pos_list(), device=owner.device),
        atol=1e-6,
    )
    owner.alpha = torch.tensor(
        [action_scale(name) for name in G1_JOINT_NAMES], device=owner.device
    )
    action = raw.action_manager.get_term("joint_pos")
    owner.actual_scale = action.scale.clone()
    assert torch.allclose(owner.actual_scale, owner.alpha[None], atol=1e-5, rtol=1e-5)
    assert not torch.any(owner.robot.data.encoder_bias)
    assert abs(raw.step_dt - 0.02) < 1e-10
    owner.xi = torch.zeros(count, 33, device=owner.device)
    foot_ids, _ = owner.robot.find_geoms(r"^(left|right)_foot[1-7]_collision$")
    foot = int(owner.robot.indexing.geom_ids[foot_ids[0]])
    torso = int(
        owner.robot.indexing.body_ids[owner.robot.body_names.index("torso_link")]
    )
    friction = raw.sim.model.geom_friction
    owner.xi[:, 0] = friction[:, foot, 0] if friction.ndim == 3 else friction[foot, 0]
    nominal = torch.tensor(
        np.array(raw.sim.mj_model.body_ipos[torso]), device=owner.device
    )
    ipos = raw.sim.model.body_ipos
    owner.xi[:, 30:33] = (ipos[:, torso] if ipos.ndim == 3 else ipos[torso]) - nominal
    owner.f = torch.zeros(count, dtype=torch.long, device=owner.device)
    owner.ep_step = torch.zeros_like(owner.f)
    owner.a_prev = torch.zeros(count, 29, device=owner.device)
    owner.start_bin = torch.zeros_like(owner.f)
    owner.ep_disc_ret, owner.ep_ret = (
        torch.zeros(count, device=owner.device),
        torch.zeros(count, device=owner.device),
    )
    owner.G_max = MAX_STEP_REWARD / (1 - gamma)
    owner.bin_G = torch.zeros(motion.num_bins, device=owner.device)
    owner.bin_w_avg = torch.full(
        (motion.num_bins,), 1.0 / motion.num_bins, device=owner.device
    )
    owner.done_eval = torch.zeros(count, dtype=torch.bool, device=owner.device)
    owner.failed_eval = torch.zeros_like(owner.done_eval)
    owner.reset(torch.arange(count, device=owner.device))


def read_state(owner):
    from supermimic.rotations import quat_to_mat

    qpos, qvel = owner.sim.data.qpos[:].clone(), owner.sim.data.qvel[:].clone()
    rotation = quat_to_mat(qpos[:, 3:7])
    angular_world = (rotation @ qvel[:, 3:6, None]).squeeze(-1)
    return (
        qpos[:, :3] - owner.origins,
        rotation,
        qvel[:, :3],
        angular_world,
        qpos[:, 7:],
        qvel[:, 6:],
    )


def write_state(owner, ids, p, rotation, v, w, q, qd):
    import torch
    from supermimic.rotations import mat_to_quat

    owner.robot.reset(env_ids=ids)
    owner.raw.action_manager.reset(ids)
    owner.sim.data.qpos[ids] = torch.cat(
        (p + owner.origins[ids], mat_to_quat(rotation), q), -1
    )
    angular_body = (rotation.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
    owner.sim.data.qvel[ids] = torch.cat((v, angular_body, qd), -1)
    owner.sim.data.qacc_warmstart[ids] = 0
    owner.sim.forward()


def step_physics(owner, actions):
    # The upstream xi bias is an actuator offset and subtracted in observations.
    # mjlab's own encoder_bias remains zero; apply the upstream offset explicitly.
    target = owner.default_q + owner.xi[:, 1:30] + owner.alpha * actions
    owner.raw.step((target - owner.default_q) / owner.actual_scale)


def refresh(owner):
    owner.sim.forward()
