"""Physical evaluation and interactive play of owned reference-recipe models."""

from pathlib import Path
from types import SimpleNamespace

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import copy_file_snapshot, snapshot_file_hashes
from embodiedforge.workflows import supermimic_backflip as backflip


def validate_checkpoint(saved, *, training=False, iterations=None):
    required = {"pi", "wm", "args", "iteration"}
    if not isinstance(saved, dict) or "config" in saved or not required.issubset(saved):
        raise ValueError(
            "Expected an owned reference-recipe checkpoint, not a Cuboi model"
        )
    args = saved["args"]
    if not isinstance(args, dict) or any(
        not isinstance(args.get(name), int) or args[name] <= 0
        for name in ("hist", "goal_frames")
    ):
        raise ValueError("Invalid reference-policy history/goal configuration")
    if training and not {"critic", "opt_wm", "opt_pi", "opt_v"}.issubset(saved):
        raise ValueError(
            "Training resume requires full last.pt with critic/optimizer state; small snapshots are evaluation-only"
        )
    if training and iterations is not None and iterations <= saved["iteration"]:
        raise ValueError(
            "Training target is global and must exceed the resumed iteration"
        )
    return args


def snapshot_diagnostic_reference(options, output):
    """Keep saved-state checks independent of subsequent shared-cache changes."""
    data = backflip.task_directory(options["workspace"]) / "data"
    expected = options.get("expected_motions")
    if expected is None:
        expected = backflip.prepared(data)["eval_files"]
    source = data / "eval/backflip.npz"
    private = Path(output) / "inputs/diagnostic-reference.npz"
    private.parent.mkdir(parents=True, exist_ok=True)
    digest = copy_file_snapshot(source, private)
    if digest != expected["backflip.npz"]:
        raise ValueError(
            "Diagnostic reference differs from the expected input manifest"
        )
    private.chmod(0o444)
    if snapshot_file_hashes(private)[0] != digest:
        raise ValueError("Private diagnostic reference changed after copying")
    return private, digest


def components(options, motion_csv):
    import supermimic.networks as networks
    import torch
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP, OBS_DIM
    from supermimic.motion import MotionLib
    from supermimic.networks import Policy
    from supermimic.robot import default_joint_pos_list

    saved = torch.load(options["checkpoint"], map_location="cpu", weights_only=False)
    args = validate_checkpoint(saved)
    torch.manual_seed(0)
    networks.AMP_DTYPE = torch.bfloat16 if args.get("amp", "none") == "bf16" else None
    kin = G1Kinematics("cuda:0")
    motion = MotionLib([str(motion_csv)], kin, "cuda:0", src_fps=50)
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device="cuda:0"))
    policy = Policy(OBS_DIM, 73 * args["goal_frames"], hist=args["hist"])
    policy.load_state_dict(saved["pi"])
    for name, value in policy.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite reference policy tensor: {name}")
    policy = policy.to("cuda:0").eval()
    return saved, args, motion, mdp, policy


def actual_state(env):
    import torch

    state = torch.cat((env.sim.data.qpos[:], env.sim.data.qvel[:]), -1)
    state[:, :3] -= env.origins
    return state


def evaluate(options, output, motion_csv):
    import numpy as np
    import torch
    from supermimic.env import TrackingSim, make_sim
    from supermimic.mdp import OBS_DIM

    from embodiedforge.workflows.supermimic_backflip_quality import (
        airborne_geometry,
        combine_quality,
    )
    from embodiedforge.workflows.supermimic_contacts import ContactRecorder
    from embodiedforge.workflows.supermimic_reference_mjlab import launch_simulation

    reference, reference_sha = snapshot_diagnostic_reference(options, output)
    saved, args, motion, mdp, policy = components(options, motion_csv)
    if motion.num_frames != 571:
        raise ValueError("Physical evaluation requires the reference plus5sextra hold")
    config = make_sim("cuda:0", SimpleNamespace(seed=0))
    history = torch.zeros(1, args["hist"], OBS_DIM, device="cuda:0")
    states, rewards, actions = [], [], []
    failed, finished = False, False
    with launch_simulation(config, args):
        env = TrackingSim(config, 1, motion, mdp, eval_mode=True)
        ids = torch.zeros(1, dtype=torch.long, device="cuda:0")
        env.reset(ids, frames=ids)
        recorder = ContactRecorder(env)
        states.append(actual_state(env)[0].cpu().numpy())
        try:
            with torch.no_grad():
                for step in range(570):
                    observation = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                    history = (
                        observation[:, None].expand_as(history).clone()
                        if step == 0
                        else torch.cat((history[:, 1:], observation[:, None]), 1)
                    )
                    action = policy(history, motion.goal(env.f, args["goal_frames"]))
                    result = env.step(action)
                    states.append(actual_state(env)[0].cpu().numpy())
                    rewards.append(float(result["r"][0]))
                    actions.append(action[0].cpu().numpy())
                    failed, finished = bool(result["term"][0]), bool(result["done"][0])
                    if failed or finished:
                        break  # Evaluation never restarts an episode after failure.
            forces = recorder.save(output)
        finally:
            recorder.close()
    state = np.stack(states)
    np.savez_compressed(
        output / "backflip-rollout.npz",
        state=state,
        time_s=np.arange(len(state)) / 50,
        rewards=np.array(rewards),
        actions=np.array(actions),
    )
    complete = not failed and finished and len(state) == 571
    initial_velocity = float(np.abs(state[0, 36:]).max())
    native_return = float(np.sum(rewards[:320]) / (0.1 * 320))
    diagnostics = backflip.diagnose_rollout(state, reference, output, complete)
    geometry = airborne_geometry(state, reference, output / "physics-model.mjb")
    quality = combine_quality(
        complete=complete,
        initial_velocity=initial_velocity,
        native_return=native_return,
        control_steps=len(rewards),
        state_samples=len(state),
        diagnostics=diagnostics,
        geometry=geometry,
        forces=forces,
    )
    write_json(output / "airborne-geometry.json", geometry)
    write_json(output / "physical-quality.json", quality)
    if snapshot_file_hashes(reference)[0] != reference_sha:
        raise ValueError("Private diagnostic reference changed during evaluation")
    return {
        "iteration": saved["iteration"],
        "goal_dim": 73 * args["goal_frames"],
        "obs_dim": OBS_DIM,
        "summary": {
            "mean_norm_return": sum(rewards) / (0.1 * len(rewards)),
            "full_clip_zero_padded_return": sum(rewards) / (0.1 * 570),
            "native320step_zero_padded_return": native_return,
            "failed": int(failed or not finished),
            "executed_control_steps": len(rewards),
            "actual_state_samples": len(state),
        },
        "backflip_diagnostics": diagnostics,
        "airborne_geometry": geometry,
        "rollout_artifact_sha256": {
            name: snapshot_file_hashes(output / name)[0]
            for name in (
                "backflip-rollout.npz",
                "contact-forces-200hz.npz",
                "physics-model.mjb",
            )
        },
        "diagnostic_reference_snapshot": {
            "private_path": str(reference),
            "sha256": reference_sha,
            "read_only": True,
        },
        "contact_force_audit": forces,
        "physical_quality": quality,
        "skill_success": quality["ported_quality_passed"],
        "strict_v14_equivalence": False,
        "initial_velocity_max_abs": initial_velocity,
        "torch_flags": {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        },
        "evaluation_protocol": "Initial state plus570actual controls covers571timestamps0..11.4s, with2280actual solved-force substeps. No resets, state substitution or retiming. Native next-frame rewards are not interchangeable with Cuboi simulator rewards. Ported thresholds on current mjlab assets are not exactV14reproduction.",
    }


def play(options, output, motion_csv):
    del output
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
    from supermimic.env import TrackingSim, make_sim

    from embodiedforge.workflows.supermimic_reference_mjlab import launch_simulation
    from embodiedforge.workflows.supermimic_reference_play import build_viewer_policy

    if options["num_envs"] != 1:
        raise ValueError("Reference play requires one environment")
    saved, args, motion, mdp, policy = components(options, motion_csv)
    config = make_sim("cuda:0", SimpleNamespace(seed=0))
    with launch_simulation(config, args):
        backend = TrackingSim(config, 1, motion, mdp, eval_mode=True)
        wrapped, act = build_viewer_policy(backend, policy, args)
        viewer = (
            NativeMujocoViewer if options["viewer"] == "native" else ViserPlayViewer
        )
        viewer(wrapped, act).run()
    return {
        "iteration": saved["iteration"],
        "goal_dim": 73 * args["goal_frames"],
        "obs_dim": 99,
        "scope": "Interactive actual-policy physics; playback loops/reset after episode end or failure. Use eval for uninterrupted physical acceptance.",
    }
