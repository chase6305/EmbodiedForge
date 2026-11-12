"""Run a small, real CuboiLeo SuperMimic training/resume/evaluation milestone.

Invoke with the dedicated SuperMimic SDK Python, under process_supervisor.py.
This reduced configuration checks the pipeline; it is not a paper-scale result.
Only trusted locally produced checkpoints may be supplied to --resume.
"""

import argparse
import importlib.metadata
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--motions", required=True, type=Path)
    parser.add_argument("--motion-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--profile", choices=("smoke", "lafan"), default="smoke")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument(
        "--anchor-obs",
        action="store_true",
        help="Separate+9-goal experiment; changes the paper observation protocol",
    )
    parser.add_argument(
        "--resume", type=Path, help="Trusted checkpoint from a prior local run"
    )
    args = parser.parse_args()
    if args.max_hours is not None and (
        not math.isfinite(args.max_hours) or args.max_hours <= 0
    ):
        parser.error("max-hours must be positive and finite")
    if args.iterations <= 20 or args.num_envs < 8:
        parser.error("iterations must exceed warmup=20; num-envs must be at least8")
    if args.warmup_iterations is not None and (
        args.warmup_iterations < 20
        or (args.resume is None and args.iterations <= args.warmup_iterations + 2)
    ):
        parser.error("warmup must be at least20 and leave steps for updates")
    args.root = args.root.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from embodiedforge.runtime.artifacts import write_json
    from embodiedforge.runtime.hashing import copy_file_snapshot, snapshot_file_hashes
    from embodiedforge.runtime.provenance import (
        snapshot_implementation,
        validate_snapshot,
    )
    from embodiedforge.workflows.supermimic_catalog import REFERENCES
    from embodiedforge.workflows.supermimic_inputs import (
        motion_manifest,
        smoke_training_clips,
        snapshot_motions,
        validate_motion_snapshot,
    )
    from embodiedforge.workflows.supermimic_source import (
        matches_version,
        source_identity,
    )

    reference = REFERENCES["cuboi-reproduced"]
    required_files = reference["source_files"] + [
        str(p.relative_to(args.root))
        for p in (args.root / "src/supermimic").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    source = source_identity(args.root, reference["revision"], required_files)
    implementation = args.output / "implementation/supermimic"
    manifest = snapshot_implementation(args.root / "src/supermimic", implementation)
    for name, digest in manifest["files"].items():
        if source["files"].get("src/supermimic/" + name) != digest:
            raise ValueError(f"Frozen source differs from pinned source: {name}")
    copy_file_snapshot(args.root / "LICENSE", args.output / "implementation/LICENSE")
    sys.path.insert(0, str(implementation.parent))
    record = {
        "status": "preparing",
        "backend": "cuboi-reproduced",
        "provenance": "independent_reproduction",
        "smoke_only": args.profile == "smoke",
        "paper_scale_result": False,
        "profile": args.profile,
        "anchor_observation_experiment": args.anchor_obs,
        "official_weights_used": False,
        "source": source,
        "implementation_manifest": manifest,
        "seed": args.seed,
        "resume_sha256": snapshot_file_hashes(args.resume)[0] if args.resume else None,
        "scope": "Independent G1/MuJoCo-Warp world-model+BPTT milestone with reduced environment/batch budget; no paper-scale or continuous-RNG resume equivalence claimed.",
        "runner_sha256": snapshot_file_hashes(Path(__file__))[0],
    }
    write_json(args.output / "result.json", record)
    trainer = None
    try:
        # Validate before importing CUDA and train from a private file copy.
        # The source cache can then change without replacing active run inputs.
        inputs = snapshot_motions(
            args.motions,
            args.output / "inputs/motions",
            motion_manifest(args.motion_manifest) if args.motion_manifest else None,
        )
        record["motion_input_snapshot"] = inputs
        record["motion_files"] = inputs["files"]
        write_json(args.output / "inputs-manifest.json", inputs)
        import supermimic
        import torch
        from supermimic.config import TrainConfig
        from supermimic.envs.mdp.motion_library import discover_npz
        from supermimic.eval.benchmark import (
            load_policy_from_checkpoint,
            run_benchmark,
            save_results,
            summarize,
        )
        from supermimic.training.loop import Trainer

        if not Path(supermimic.__file__).resolve().is_relative_to(implementation):
            raise ValueError("Imported SuperMimic is outside the frozen implementation")
        if not torch.cuda.is_available():
            raise RuntimeError("This reproduction milestone requires CUDA")
        torch.set_num_threads(1)
        record["environment"] = {
            "python": sys.version,
            "gpu": torch.cuda.get_device_name(0),
            "packages": {
                name: importlib.metadata.version(name)
                for name in reference["dependencies"]
            },
            "mjlab_origin": json.loads(
                importlib.metadata.distribution("mjlab").read_text("direct_url.json")
            ),
        }
        for name, constraint in reference["dependencies"].items():
            if not matches_version(record["environment"]["packages"][name], constraint):
                raise ValueError(f"Unsupported SDK version: {name}")
        if sys.version_info[:2] != tuple(reference["python_minor"]):
            raise ValueError("The pinned reproduction requires Python3.12")
        commit = record["environment"]["mjlab_origin"]["vcs_info"]["commit_id"]
        if commit != reference["framework"]["revision"]:
            raise ValueError("mjlab commit differs from the pinned framework")
        from supermimic.data.motion_convert import g1_xml_path

        asset_root = g1_xml_path().parent.parent
        record["g1_assets"] = {
            str(p.relative_to(asset_root)): snapshot_file_hashes(p)[0]
            for p in sorted(asset_root.rglob("*"))
            if p.is_file()
        }
        cfg = TrainConfig(run_name="training", log_root=args.output, seed=args.seed)
        cfg.data.npz_dir = Path(inputs["root"])
        cfg.data.clips = smoke_training_clips(inputs["files"])
        cfg.env.num_envs = args.num_envs
        cfg.env.replay_capacity = 200
        cfg.world_model.hidden = (256, 256)
        cfg.policy.encoder_hidden = (256, 128)
        cfg.policy.decoder_hidden = (256, 128)
        cfg.wm_trainer.batch_size = 128
        cfg.wm_trainer.batches_per_update = 2
        cfg.policy_trainer.batch_size = args.num_envs
        cfg.policy_trainer.horizon = 8
        cfg.policy_trainer.critic_epochs = 1
        cfg.policy_trainer.critic_minibatches = 2
        cfg.policy_trainer.compile = None
        cfg.schedule.total_iterations = args.iterations
        cfg.schedule.warmup_iterations = 20
        cfg.schedule.log_interval = 20
        cfg.schedule.checkpoint_interval = args.iterations
        if args.profile == "lafan":
            # Keep paper-width networks, H=32 and all supplied clips. Batch/env
            # reductions fit this24GiB machine; these differ from the paper.
            cfg = TrainConfig(run_name="training", log_root=args.output, seed=args.seed)
            cfg.data.npz_dir = Path(inputs["root"])
            cfg.env.num_envs = args.num_envs
            cfg.wm_trainer.batch_size = min(args.num_envs, 512)
            cfg.policy_trainer.batch_size = min(args.num_envs, 256)
            cfg.schedule.total_iterations = args.iterations
            cfg.schedule.log_interval = 50
            cfg.schedule.checkpoint_interval = 1000
        cfg.env.anchor_obs = args.anchor_obs
        if args.warmup_iterations is not None:
            cfg.schedule.warmup_iterations = args.warmup_iterations
        cfg.schedule.max_hours = args.max_hours
        cfg.resume = args.resume.resolve() if args.resume else None
        record["configuration"] = json.loads(json.dumps(asdict(cfg), default=str))
        motions = discover_npz(cfg.data.npz_dir)
        if not motions:
            raise ValueError("No converted motion files")
        record["status"] = "training"
        write_json(args.output / "result.json", record)
        trainer = Trainer(cfg)
        initial_iteration, initial_updates = trainer.iteration, trainer.num_updates
        record["configuration"] = json.loads(json.dumps(asdict(cfg), default=str))
        if args.iterations <= initial_iteration:
            raise ValueError(
                "iterations is a global target and must exceed the resumed iteration"
            )
        # Upstream checkpoints restore weights/optimizers, not replay rows.
        # The upstream absolute warmup threshold has already elapsed on resume.
        # Refill fresh history/windows before permitting any optimizer update.
        if args.resume:
            refill_steps = max(20, cfg.policy.history + cfg.wm_trainer.window + 2)
            cfg.schedule.warmup_iterations = initial_iteration + refill_steps
            if args.iterations <= cfg.schedule.warmup_iterations + 2:
                raise ValueError(
                    "Resume target must leave steps for refill and new updates"
                )
            record["resume_replay_refill_steps"] = refill_steps
            record["configuration"] = json.loads(json.dumps(asdict(cfg), default=str))
            write_json(trainer.log_dir / "config.json", record["configuration"])
            write_json(args.output / "result.json", record)
        trainer.run()
        if trainer.num_updates <= initial_updates:
            raise RuntimeError("No new world-model/policy updates executed")
        for name, module in (
            ("world_model", trainer.wm),
            ("policy", trainer.policy),
            ("critic", trainer.critic),
        ):
            for key, value in module.state_dict().items():
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite tensor in {name}.{key}")
        checkpoint = trainer.log_dir / "ckpt_latest.pt"
        policy, loaded_cfg = load_policy_from_checkpoint(checkpoint, cfg.device)
        obs = torch.linspace(-0.1, 0.1, 3 * 930, device=cfg.device).reshape(3, 930)
        goal_dim = loaded_cfg["policy"]["goal_dim"]
        goal = torch.linspace(-0.2, 0.2, 3 * goal_dim, device=cfg.device).reshape(
            3, goal_dim
        )
        with torch.no_grad():
            expected, actual = trainer.policy(obs, goal), policy(obs, goal)
            if actual.shape != (3, 29) or not torch.equal(expected, actual):
                raise ValueError("Policy checkpoint reload changed fixed-input output")
        record.update(
            initial_iteration=initial_iteration,
            final_iteration=trainer.iteration,
            new_environment_steps=trainer.iteration - initial_iteration,
            initial_updates=initial_updates,
            final_updates=trainer.num_updates,
            new_updates=trainer.num_updates - initial_updates,
            checkpoint_sha256=snapshot_file_hashes(checkpoint)[0],
            checkpoint_reload_action_parity=True,
            finite_model_tensors=True,
            status="evaluating",
        )
        write_json(args.output / "result.json", record)
        trainer.env.close()
        validate_motion_snapshot(cfg.data.npz_dir, record["motion_files"])
        results = run_benchmark(
            policy,
            [str(p) for p in motions],
            num_envs=min(40, args.num_envs),
            device=cfg.device,
            future_frames=loaded_cfg["env"]["future_frames"],
            anchor_obs=loaded_cfg["env"].get("anchor_obs", False),
        )
        save_results(results, args.output / "evaluation.json")
        record["evaluation"] = summarize(results)
        record["evaluation"]["full_clip_zero_padded_return"] = sum(
            x.norm_return * x.steps for x in results
        ) / sum(x.frames for x in results)
        if snapshot_file_hashes(Path(__file__))[0] != record["runner_sha256"]:
            raise ValueError("Runner changed during this run")
        if any(
            snapshot_file_hashes(asset_root / name)[0] != digest
            for name, digest in record["g1_assets"].items()
        ):
            raise ValueError("G1 assets changed during this run")
        # Recheck both sources and converted inputs after all computation.
        source_identity(args.root, reference["revision"], required_files)
        validate_snapshot(implementation, manifest)
        validate_snapshot(args.root / "src/supermimic", manifest)
        validate_motion_snapshot(cfg.data.npz_dir, record["motion_files"])
        record["status"] = "complete"
        write_json(args.output / "result.json", record)
        print(
            json.dumps({k: record[k] for k in ("status", "new_updates", "evaluation")}),
            flush=True,
        )
    except BaseException as error:
        record.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(args.output / "result.json", record)
        raise
    finally:
        if trainer is not None:
            trainer.env.close()
            if trainer.writer is not None:
                trainer.writer.close()


if __name__ == "__main__":
    main()
