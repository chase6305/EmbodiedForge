"""SDK process for MorphMimic; the upstream implementation owns all learning."""

import json
import sys
from pathlib import Path


def prepare(request):
    import numpy as np
    import torch
    import yaml
    from morphmimic.data.seed_convert import (
        body_names,
        clip_stats,
        finite_diff_velocities,
        foot_geoms,
        forward_kinematics,
        make_model,
        zmin_bodies,
    )
    from morphmimic.robots import get_robot

    output = Path(request["output"])
    source_spec = get_robot("g1")
    source_floor = {}
    records = []
    for robot in request["robots"]:
        spec = get_robot(robot)
        model = make_model(robot)
        directory = output / robot
        directory.mkdir()
        names = []
        for path_string in request["clips"]:
            path = Path(path_string)
            with np.load(path, allow_pickle=False) as data:
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if fps != 50:
                    raise ValueError(f"Expected existing 50 Hz motion: {path}")
                joints = tuple(str(x) for x in data["joint_names"])
                bodies = tuple(str(x) for x in data["body_names"])
                if len(set(joints)) != len(joints):
                    raise ValueError(f"Duplicate joints: {path}")
                permutation = [joints.index(j) for j in source_spec.joint_names]
                count = min(len(data["joint_pos"]), round(request["seconds"] * fps))
                if count < 64:
                    raise ValueError("Each clip needs at least 64 frames")
                q = data["joint_pos"][:count, permutation].astype(np.float64)
                root = bodies.index(source_spec.root_body)
                pos = data["body_pos_w"][:count, root].astype(np.float64)
                quat = data["body_quat_w"][:count, root].astype(np.float64)
            if not all(np.isfinite(x).all() for x in (q, pos, quat)):
                raise ValueError(f"Nonfinite source data: {path}")
            if not np.allclose(np.linalg.norm(quat, axis=-1), 1, atol=1e-3):
                raise ValueError(f"Expected unit wxyz quaternions: {path}")
            canonical = source_spec.translator.to_canonical(torch.from_numpy(q))
            native = spec.translator.to_native(
                canonical, torch.from_numpy(model.qpos0[7:].copy())
            ).numpy()
            if path not in source_floor:
                source_model = make_model("g1")
                source_fk = forward_kinematics(
                    source_model,
                    pos,
                    quat,
                    q,
                    zmin_body_ids=zmin_bodies(
                        source_model, source_spec.zmin_exclude_regex
                    ),
                    feet=foot_geoms(source_model, source_spec.foot_geom_regex),
                )
                source_floor[path] = source_fk.foot_dist.min(axis=1) + max(
                    0.0, -float(source_fk.foot_dist.min())
                )
            # A diagnostic joint transfer, not SOMA/BONES-SEED retargeting.
            if robot != "g1":
                pos *= spec.height_scale / source_spec.height_scale
            limits = model.jnt_range[1:]
            limited = model.jnt_limited[1:].astype(bool)
            excess = np.maximum(limits[:, 0] - native, native - limits[:, 1])
            violations = int((excess[:, limited] > 1e-6).sum())
            # Keep every prepared reference inside the target's hard limits.
            native[:, limited] = np.clip(
                native[:, limited], limits[limited, 0], limits[limited, 1]
            )
            fk = forward_kinematics(
                model,
                pos,
                quat,
                native,
                zmin_body_ids=zmin_bodies(model, spec.zmin_exclude_regex),
                feet=foot_geoms(model, spec.foot_geom_regex),
            )
            # Transfer the source's ground clearance, including running flight.
            # A constant shift alone left X2/T2 standing in mid-air.
            shift = np.full(count, max(0.0, -float(fk.foot_dist.min())))
            if robot != "g1":
                desired = source_floor[path] * (
                    spec.height_scale / source_spec.height_scale
                )
                shift = desired - fk.foot_dist.min(axis=1)
            if np.any(shift):
                pos[:, 2] += shift
                fk = forward_kinematics(
                    model,
                    pos,
                    quat,
                    native,
                    zmin_body_ids=zmin_bodies(model, spec.zmin_exclude_regex),
                    feet=foot_geoms(model, spec.foot_geom_regex),
                )
            # Match the author's saved representation (q and -q encode the same
            # rotation, but the file convention is w >= 0).
            body_quat = np.where(
                fk.body_quat_w[..., :1] < 0, -fk.body_quat_w, fk.body_quat_w
            )
            jv, bv, av = finite_diff_velocities(native, fk.body_pos_w, body_quat, 50)
            destination = directory / path.name
            arrays = dict(
                joint_pos=native,
                joint_vel=jv,
                body_pos_w=fk.body_pos_w,
                body_quat_w=body_quat,
                body_lin_vel_w=bv,
                body_ang_vel_w=av,
                z_min=fk.z_min,
                foot_dist=fk.foot_dist,
            )
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError(f"Nonfinite target kinematics: {robot}/{path.name}")
            motion = dict(
                **{k: v.astype(np.float32) for k, v in arrays.items()},
                fps=np.array(50, dtype=np.int64),
                body_names=np.array(body_names(model)),
                joint_names=np.array(spec.joint_names),
                robot=np.array(robot),
                root_body=np.array(spec.root_body),
            )
            stats = clip_stats(path.stem, motion, robot)
            if stats["feas_reject"] is not None:
                raise ValueError(
                    f"Infeasible reference {robot}/{path.name}: {stats['feas_reject']}"
                )
            np.savez_compressed(destination, **motion)
            from embodiedforge.runtime.hashing import file_sha256

            records.append(
                dict(
                    source=str(path),
                    source_sha256=file_sha256(path),
                    robot=robot,
                    frames=count,
                    fps=50,
                    joint_permutation=permutation,
                    clipped_joint_samples=violations,
                    root_height_adjustment_m=dict(
                        min=float(shift.min()),
                        max=float(shift.max()),
                        rms=float(np.sqrt(np.mean(shift**2))),
                    ),
                    feasibility=stats,
                    output=str(destination),
                    sha256=file_sha256(destination),
                    transfer="native G1"
                    if robot == "g1"
                    else "functional joint mapping; not SOMA retargeting",
                )
            )
            names.append(destination.name)
        (output / f"{robot}.yaml").write_text(
            yaml.safe_dump(
                dict(
                    root=str(directory),
                    clips=names,
                )
            )
        )
    return dict(dataset="existing LAFAN1 G1", paper_dataset=False, clips=records)


def cpu_collectives():
    """Synchronous Gloo transport for several CUDA ranks on one physical GPU.

    Only communication changes. The upstream gradient averaging and latent
    alignment still execute at their original call sites.
    """
    import torch
    import torch.distributed as dist

    originals = {
        name: getattr(dist, name) for name in ("all_reduce", "broadcast", "all_gather")
    }

    def wrap(name):
        def call(tensor, *args, **kwargs):
            if kwargs.get("async_op", False):
                raise ValueError(
                    "Single-GPU transport supports synchronous collectives only"
                )
            if name == "all_gather":
                outputs, value = tensor, args[0]
                if not value.is_cuda:
                    return originals[name](outputs, *args, **kwargs)
                host = value.detach().cpu()
                gathered = [torch.empty_like(host) for _ in outputs]
                result = originals[name](gathered, host, *args[1:], **kwargs)
                with torch.no_grad():
                    for target, received in zip(outputs, gathered, strict=True):
                        target.copy_(received)
                return result
            if not tensor.is_cuda:
                return originals[name](tensor, *args, **kwargs)
            host = tensor.detach().cpu()
            result = originals[name](host, *args, **kwargs)
            with torch.no_grad():
                tensor.copy_(host)
            return result

        return call

    for name in originals:
        setattr(dist, name, wrap(name))


def train_rank(rank, request, rendezvous=None):
    import faulthandler
    import random

    import numpy as np
    import torch
    import torch.distributed as dist
    from mjlab.utils.torch import configure_torch_backends
    from morphmimic.train.trainer import TrainCfg, Trainer

    faulthandler.enable()
    torch.set_num_threads(2)
    configure_torch_backends()
    random.seed(request["seed"] + rank)
    np.random.seed(request["seed"] + rank)
    torch.manual_seed(request["seed"] + rank)
    robots = request["robots"]
    multi = len(robots) > 1
    if multi:
        dist.init_process_group(
            "gloo", init_method=rendezvous, rank=rank, world_size=len(robots)
        )
        cpu_collectives()
    trainer = None
    try:
        subsets = tuple(str(Path(request["data"]) / f"{r}.yaml") for r in robots)
        cfg = TrainCfg(
            robot=robots[0],
            subset_yaml=subsets[0],
            robots=tuple(robots) if multi else None,
            subset_yamls=subsets if multi else None,
            share="all",
            morphology=True,
            actor_geometry=True,
            decoder_film=True,
            kp_interface=True,
            kp_masking="full",
            latent_align_batch=request["batch"],
            num_envs=request["num_envs"],
            replay_capacity=128,
            wm_batch=request["batch"],
            imag_batch=request["batch"],
            kp_imag_batch=request["batch"],
            iterations=request["iterations"],
            compile="none",
            log_dir=str(Path(request["output"]) / "training"),
            run_name="shared",
            log_every=10,
            ckpt_every=500,
            seed=request["seed"],
        )
        print(
            f"[EmbodiedForge] rank {rank} / {robots[rank]}: initializing Trainer",
            flush=True,
        )
        trainer = Trainer(cfg, "cuda:0")
        print(
            f"[EmbodiedForge] rank {rank} / {robots[rank]}: Trainer ready", flush=True
        )
        if request.get("resume"):
            checkpoint = Path(request["resume"])
            if checkpoint.is_dir():
                checkpoint = (
                    checkpoint / robots[rank] / "latest.pt"
                    if multi
                    else checkpoint / "latest.pt"
                )
            trainer.load(checkpoint)
        trainer.train()
        ckpt = trainer.checkpoint()
        steps = {}
        for key, optimizer in ckpt["optim"].items():
            if optimizer is None:
                continue
            counts = [
                float(v["step"]) for v in optimizer["state"].values() if "step" in v
            ]
            steps[key] = min(counts) if counts else 0
        finite = all(
            torch.isfinite(p).all().item()
            for module in (
                trainer.wm,
                trainer.policy,
                trainer.critic,
                trainer.critic_kp,
            )
            if module is not None
            for p in module.parameters()
        )
        result = dict(
            robot=robots[rank],
            iteration=trainer.it,
            optimizer_steps=steps,
            finite_parameters=finite,
            morphology_dim=int(trainer.wm.morph.numel()),
            film=trainer.cfg.decoder_film,
            latent_align_weight=trainer.cfg.latent_align_weight,
            policy_updates=max(0, trainer.it - trainer.cfg.wm_only_iters),
            checkpoint=str(trainer.run_dir / "latest.pt"),
            last_log=trainer.last_log,
        )
        if (
            not finite
            or result["morphology_dim"] != 125
            or set(steps) != {"wm", "policy", "critic", "kp", "critic_kp"}
            or min(steps.values()) <= 0
        ):
            raise RuntimeError(f"Incomplete training update coverage: {result}")
        from embodiedforge.runtime.artifacts import write_json

        write_json(Path(request["output"]) / f"{robots[rank]}-training.json", result)
    finally:
        if trainer is not None:
            trainer.close()
        if multi:
            dist.destroy_process_group()


def train(request):
    if len(request["robots"]) == 1:
        train_rank(0, request)
    else:
        import socket

        import torch.multiprocessing as mp

        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        mp.spawn(
            train_rank,
            args=(request, f"tcp://127.0.0.1:{port}"),
            nprocs=len(request["robots"]),
            join=True,
        )
    return dict(
        robots=request["robots"],
        transport="single GPU / synchronous CPU Gloo"
        if len(request["robots"]) > 1
        else "single GPU",
    )


def export(request):
    from dataclasses import asdict

    from deploy.export_onnx import export as upstream_export
    from deploy.export_onnx import resolve_paths

    report = upstream_export(
        resolve_paths([Path(request["checkpoint"])], None),
        Path(request["output"]) / "policy",
    )
    return asdict(report)


def evaluate(request):
    import runpy

    from embodiedforge.runtime.environment import child_environment
    from embodiedforge.runtime.process import run_process

    output = Path(request["output"])
    assets = output / "assets"
    results = []
    for robot in request["robots"]:
        sys.argv = [
            "export_assets",
            "--robot",
            robot,
            "--subset-yaml",
            str(Path(request["data"]) / f"{robot}.yaml"),
            "--clips",
            *request["clip_names"],
            "--out",
            str(assets),
        ]
        runpy.run_module("deploy.sim2sim.export_assets", run_name="__main__")
        for clip in request["clip_names"]:
            for interface in ("motion", "keypoint"):
                summary = output / f"{robot}-{clip}-{interface}.json"
                run_process(
                    [
                        sys.executable,
                        "-m",
                        "deploy.run",
                        "--deploy",
                        request["policy"],
                        "--assets",
                        str(assets),
                        "--robot",
                        robot,
                        "--clip",
                        clip,
                        "--interface",
                        interface,
                        "--summary",
                        str(summary),
                    ],
                    cwd=request["source"],
                    env=child_environment(Path(sys.executable).parent.parent),
                    log_path=output / f"{robot}-{clip}-{interface}.log",
                    timeout=300,
                )
                results.append(json.loads(summary.read_text()))
    completed = sum(episode["result"] == "completed" for episode in results)
    return dict(
        episodes=results,
        episodes_completed=completed,
        episodes_total=len(results),
        tracking_passed=bool(results) and completed == len(results),
        paper_benchmark=False,
        physical_robot_tested=False,
    )


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    source = Path(request["source"])
    sys.path[:0] = [
        str(source / "src"),
        str(source),
        str(Path(__file__).resolve().parents[1]),
    ]
    from embodiedforge.runtime.artifacts import write_json

    command = request["command"]
    if command == "doctor":
        import importlib.metadata as metadata

        import morphmimic  # noqa: F401
        import torch

        result = dict(
            cuda=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(0),
            versions={
                name: metadata.version(name)
                for name in ("torch", "mjlab", "mujoco", "numpy", "warp-lang")
            },
        )
    else:
        result = {
            "prepare": prepare,
            "train": train,
            "export": export,
            "eval": evaluate,
        }[command](request)
    write_json(Path(request["output"]) / "result.json", result)


if __name__ == "__main__":
    main()
