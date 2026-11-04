"""Workspace launcher for the pinned independent SuperMimic SDK.

The catalog/preflight CLI remains metadata-only. This separate entry point
explicitly runs a trusted local model in its dedicated SDK subprocess.
"""

import argparse
import importlib.metadata
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import snapshot_file_hashes
from embodiedforge.runtime.process import run_process
from embodiedforge.runtime.provenance import snapshot_implementation, validate_snapshot
from embodiedforge.runtime.records import run_identity, run_lifecycle
from embodiedforge.supermimic import matches_version, source_identity
from embodiedforge.workflows import supermimic_backflip as backflip
from embodiedforge.workflows.supermimic_catalog import REFERENCES
from embodiedforge.workflows.supermimic_inputs import (
    inspect_motion_files,
    snapshot_motions,
    validate_motion_snapshot,
)

WORKSPACE = Path(__file__).resolve().parents[2]
BASELINE = "supermimic-reproduction-20261027-fourhour"


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "action",
        nargs="?",
        default="play",
        choices=("play", "eval", "train", "doctor", "prepare"),
    )
    result.add_argument("--task", choices=("lafan", "backflip"), default="lafan")
    result.add_argument(
        "--recipe",
        choices=("cuboi", "reference"),
        default=None,
        help="Backflip training implementation; reference uses the complete vibed algorithm on mjlab",
    )
    result.add_argument("--root", type=Path, help="Dedicated SDK checkout")
    result.add_argument("--checkpoint", type=Path, help="Trusted local checkpoint")
    result.add_argument("--motions", type=Path, help="Converted NPZ directory")
    result.add_argument("--output", type=Path, help="New output directory")
    result.add_argument("--clip", help="Motion for play")
    result.add_argument("--viewer", choices=("viser", "native"), default="viser")
    result.add_argument("--num-envs", type=positive_int)
    result.add_argument("--iterations", type=positive_int, default=100000)
    result.add_argument("--hours", type=positive_float)
    result.add_argument("--seed", type=int)
    result.add_argument("--profile", choices=("smoke", "lafan"), default="lafan")
    result.add_argument("--fresh", action="store_true", help="Train from scratch")
    result.add_argument(
        "--anchor-obs", action="store_true", help="Separate +9-goal training experiment"
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Show paths without running the SDK or creating files",
    )
    result.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--backflip-config", type=Path, help=argparse.SUPPRESS)
    return result


def resolve_options(args, workspace=WORKSPACE):
    workspace = Path(workspace)
    root = (
        (args.root or workspace / ".cache/external/supermimic-reproduced")
        .expanduser()
        .resolve()
    )
    from embodiedforge.workflows import supermimic_reference

    recipe = args.recipe or (
        supermimic_reference.preferred_recipe(workspace)
        if args.task == "backflip"
        else "cuboi"
    )
    if recipe == "reference" and args.anchor_obs:
        raise ValueError(
            "The reference recipe requires --task backflip and its own99dimensional observation protocol"
        )
    expected_checkpoint = None
    expected_motions = None
    fresh = args.fresh
    if args.task == "backflip":
        registry = (
            None
            if args.fresh
            or args.checkpoint is not None
            or args.action == "prepare"
            or (args.profile == "smoke" and args.checkpoint is None)
            else (
                supermimic_reference.latest_checkpoint(workspace)
                if recipe == "reference"
                else backflip.latest_checkpoint(workspace)
            )
        )
        checkpoint = args.checkpoint or (
            Path(registry["checkpoint"]) if registry else None
        )
        expected_checkpoint = (
            registry["checkpoint_sha256"]
            if registry and args.checkpoint is None
            else None
        )
        fresh = args.action == "train" and (args.fresh or checkpoint is None)
        phase = "eval" if args.action == "eval" else "train"
        data = backflip.task_directory(workspace) / "data"
        motions = data / phase
        if (data / "provenance.json").is_file():
            expected_motions = backflip.prepared(data)[phase + "_files"]
    else:
        report = json.loads(
            (workspace / "docs/experiments" / (BASELINE + ".json")).read_text()
        )
        if (
            report["training_profiles"]["paper-resume"]["training_integrity_acceptance"]
            != "accepted"
        ):
            raise ValueError("Default baseline training is not accepted")
        entry = report["matched_comparison"]["selection"]["checkpoints"]["paper-resume"]
        previous = workspace / "runs" / BASELINE
        checkpoint = (
            args.checkpoint
            or previous / "paper-resume/training" / Path(entry["path"]).name
        )
        motions = args.motions or previous / "recovered-motions"
        expected_checkpoint = entry["sha256"] if args.checkpoint is None else None
        expected_motions = (
            report["training_profiles"]["paper-resume"]["record"]["motion_files"]
            if args.motions is None
            else None
        )
    checkpoint = checkpoint.expanduser().resolve() if checkpoint else None
    motions = motions.expanduser().resolve()
    tag = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output = (
        (args.output or workspace / "runs/supermimic" / (tag + "-" + args.action))
        .expanduser()
        .resolve()
    )
    return {
        "action": args.action,
        "workspace": str(workspace),
        "root": str(root),
        "python": str(root / ".venv/bin/python"),
        "task": args.task,
        "recipe": recipe,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "expected_checkpoint_sha256": expected_checkpoint,
        "motions": str(motions),
        "expected_motions": expected_motions,
        "output": str(output),
        "num_envs": args.num_envs
        or {
            "play": 1,
            "eval": 1 if args.task == "backflip" else 40,
            "train": 1024 if recipe == "reference" else 512,
            "doctor": 1,
            "prepare": 1,
        }[args.action],
        "clip": args.clip
        or ("backflip" if args.task == "backflip" else "walk1_subject1"),
        "viewer": args.viewer,
        "iterations": args.iterations,
        "hours": args.hours
        if args.hours is not None
        else (1.5 if args.task == "backflip" else 1.0),
        "seed": args.seed
        if args.seed is not None
        else (137 if args.task == "backflip" else 0),
        "warmup": 500 if args.task == "backflip" and args.profile == "lafan" else None,
        "profile": args.profile,
        "fresh": fresh,
        "anchor_obs": args.anchor_obs,
    }


def sdk_environment():
    environment = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
    )
    return environment


def ensure_backflip(options):
    directory = backflip.task_directory(options["workspace"])
    if not (directory / "data").exists():
        if options["action"] == "doctor":
            raise ValueError(
                "Backflip data is not prepared; run ./supermimic prepare --task backflip"
            )
        directory.mkdir(parents=True, exist_ok=True)
        config = directory / "prepare-config.json"
        write_json(config, {"workspace": options["workspace"], "root": options["root"]})
        log = directory / "prepare.log"
        run_process(
            [
                options["python"],
                "-I",
                str(WORKSPACE / "supermimic"),
                "--backflip-config",
                str(config),
            ],
            cwd=WORKSPACE,
            env=sdk_environment(),
            log_path=log,
            append_log=log.exists(),
            timeout=120,
        )
    record = backflip.prepared(directory / "data")
    phase = "eval" if options["action"] == "eval" else "train"
    options["expected_motions"] = record[phase + "_files"]
    return record


def launch(options):
    root, output = Path(options["root"]), Path(options["output"])
    if not os.access(options["python"], os.X_OK):
        raise ValueError(
            f"Dedicated SDK Python not found: {options['python']}; see docs/supermimic.md"
        )
    if options["recipe"] == "reference":
        from embodiedforge.workflows import supermimic_reference

        ensure_backflip(options)
        return supermimic_reference.launch(options, sdk_environment())
    uses_checkpoint = (
        options["action"] in ("play", "eval")
        or (options["action"] == "train" and not options["fresh"])
        or (options["action"] == "doctor" and options["checkpoint"] is not None)
    )
    if uses_checkpoint:
        if options["checkpoint"] is None:
            raise ValueError(
                "No trained backflip model yet; use ./supermimic train --task backflip, or supply --checkpoint"
            )
        digest = snapshot_file_hashes(Path(options["checkpoint"]))[0]
        expected = options["expected_checkpoint_sha256"]
        if expected is not None and digest != expected:
            raise ValueError(
                "Default checkpoint hash differs from the accepted baseline"
            )
        options["checkpoint_sha256"] = digest
    reference = REFERENCES["cuboi-reproduced"]
    files = reference["source_files"] + [
        str(p.relative_to(root))
        for p in (root / "src/supermimic").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    source = source_identity(root, reference["revision"], files)
    if options["task"] == "backflip":
        data = ensure_backflip(options)
        if options["action"] == "prepare":
            print(json.dumps(data, indent=2))
            return 0
    if options["action"] == "doctor":
        hashes = inspect_motion_files(options["motions"], options["expected_motions"])
        print(
            json.dumps(
                {
                    "status": "local_inputs_verified",
                    "python": options["python"],
                    "checkpoint": options["checkpoint"],
                    "motions": options["motions"],
                    "clips": len(hashes),
                    "source_revision": source["revision"],
                    "cuda_checked": False,
                    "policy_executed": False,
                    "task": options["task"],
                    "checkpoint_available": options["checkpoint"] is not None,
                },
                indent=2,
            )
        )
        return 0
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "preparing",
        "action": options["action"],
        "task": options["task"],
        "source": source,
        "provenance": "independent_reproduction",
        "official_weights_used": False,
        "run_contract": run_identity("supermimic", options["action"], options["task"]),
        "worker_exit_observed": False,
        "worker_returncode": None,
        "launcher_sources": {
            name: snapshot_file_hashes(WORKSPACE / name)[0]
            for name in (
                "supermimic",
                "src/embodiedforge/supermimic_run.py",
                "src/embodiedforge/workflows/supermimic_inputs.py",
                "src/embodiedforge/workflows/supermimic_backflip.py",
                "benchmarks/supermimic_reproduce.py",
            )
        },
    }
    try:
        with run_lifecycle(
            record,
            lambda current: write_json(output / "launcher.json", current),
            label=output / "launcher.json",
        ):
            command = [options["python"], "-I"]
            if options["action"] == "train":
                command += [
                    str(WORKSPACE / "benchmarks/supermimic_reproduce.py"),
                    "--root",
                    str(root),
                    "--motions",
                    options["motions"],
                    "--output",
                    str(output / "training"),
                    "--profile",
                    options["profile"],
                    "--iterations",
                    str(options["iterations"]),
                    "--max-hours",
                    str(options["hours"]),
                    "--num-envs",
                    str(options["num_envs"]),
                    "--seed",
                    str(options["seed"]),
                ]
                if options["warmup"] is not None:
                    command += ["--warmup-iterations", str(options["warmup"])]
                if uses_checkpoint:
                    command += ["--resume", options["checkpoint"]]
                if options["anchor_obs"]:
                    command += ["--anchor-obs"]
                if options["expected_motions"] is not None:
                    write_json(
                        output / "expected-motions.json",
                        {"files": options["expected_motions"]},
                    )
                    command += [
                        "--motion-manifest",
                        str(output / "expected-motions.json"),
                    ]
            else:
                inputs = snapshot_motions(
                    options["motions"],
                    output / "inputs/motions",
                    options["expected_motions"],
                    names=[options["clip"] + ".npz"]
                    if options["action"] == "play"
                    else None,
                )
                implementation = output / "implementation/supermimic"
                manifest = snapshot_implementation(
                    root / "src/supermimic", implementation
                )
                for name, digest in manifest["files"].items():
                    if source["files"].get("src/supermimic/" + name) != digest:
                        raise ValueError(
                            f"Frozen source differs from pinned source: {name}"
                        )
                options.update(inputs=inputs, implementation_manifest=manifest)
                config = output / "worker-config.json"
                write_json(config, options)
                command += [
                    str(WORKSPACE / "supermimic"),
                    "--worker-config",
                    str(config),
                ]
            record.update(command=command, status="running")
            write_json(output / "launcher.json", record)
            print(f"SuperMimic {options['action']} | output: {output}", flush=True)
            run_process(
                command,
                cwd=WORKSPACE,
                env=sdk_environment(),
                log_path=output / "command.log",
            )
            record.update(worker_exit_observed=True, worker_returncode=0)
            if options["action"] == "train":
                worker = json.loads((output / "training/result.json").read_text())
                if worker["status"] != "complete":
                    raise ValueError("Training worker did not finish validation")
            for name, digest in record["launcher_sources"].items():
                if snapshot_file_hashes(WORKSPACE / name)[0] != digest:
                    raise ValueError(
                        f"Launcher source changed during execution: {name}"
                    )
            if (
                options["action"] == "train"
                and options["task"] == "backflip"
                and options["profile"] == "lafan"
            ):
                backflip.register_checkpoint(options["workspace"], output, worker)
            return 0
    except subprocess.CalledProcessError as error:
        return error.returncode if error.returncode > 0 else 128 - error.returncode
    except KeyboardInterrupt as error:
        return 128 + getattr(error, "signum", signal.SIGINT)


def load_policy(checkpoint, device):
    """Load once on CPU; only policy weights are copied to the GPU."""
    import torch
    from supermimic.policy.nets import Policy, PolicyCfg

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = saved["config"]
    pcfg = PolicyCfg(
        **{k: tuple(v) if isinstance(v, list) else v for k, v in cfg["policy"].items()}
    )
    policy = Policy(pcfg).to(device)
    policy.load_state_dict(saved["policy"])
    policy.eval()
    return policy, cfg, saved["iteration"], saved["num_updates"]


def worker(config):
    options = json.loads(Path(config).read_text())
    output = Path(options["output"])
    implementation = output / "implementation"
    sys.path.insert(0, str(implementation))
    record = {
        "status": "preparing",
        "action": options["action"],
        "official_weights_used": False,
        "task": options["task"],
    }
    env = None
    try:
        validate_snapshot(
            implementation / "supermimic", options["implementation_manifest"]
        )
        inputs = options["inputs"]
        validate_motion_snapshot(inputs["root"], inputs["files"])
        if (
            snapshot_file_hashes(options["checkpoint"])[0]
            != options["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint changed before SDK load")
        reference = REFERENCES["cuboi-reproduced"]
        versions = {
            name: importlib.metadata.version(name) for name in reference["dependencies"]
        }
        for name, constraint in reference["dependencies"].items():
            if not matches_version(versions[name], constraint):
                raise ValueError(f"Unsupported SDK version: {name}")
        if sys.version_info[:2] != tuple(reference["python_minor"]):
            raise ValueError("The pinned SDK requires Python3.12")
        origin = json.loads(
            importlib.metadata.distribution("mjlab").read_text("direct_url.json")
        )
        if origin["vcs_info"]["commit_id"] != reference["framework"]["revision"]:
            raise ValueError("mjlab commit differs from the pinned SDK")
        record["environment"] = {
            "python": sys.version,
            "packages": versions,
            "mjlab_origin": origin,
        }
        import supermimic
        import torch
        from supermimic.data.motion_convert import g1_xml_path

        if not Path(supermimic.__file__).resolve().is_relative_to(implementation):
            raise ValueError("Policy SDK is outside the frozen implementation")
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise ValueError("Policy execution requires CUDA")
        asset_root = g1_xml_path().parent.parent
        record["g1_assets"] = {
            str(p.relative_to(asset_root)): snapshot_file_hashes(p)[0]
            for p in sorted(asset_root.rglob("*"))
            if p.is_file()
        }
        policy, cfg, iteration, updates = load_policy(options["checkpoint"], "cuda:0")
        paths = sorted(Path(inputs["root"]).glob("*.npz"))
        record.update(
            iteration=iteration,
            updates=updates,
            checkpoint_sha256=options["checkpoint_sha256"],
            goal_dim=cfg["policy"]["goal_dim"],
            anchor_obs=cfg["env"].get("anchor_obs", False),
            motion_input_snapshot=inputs,
        )
        if options["action"] == "eval":
            from supermimic.eval.benchmark import run_benchmark, save_results, summarize

            diagnostics = None
            if options["task"] == "backflip":
                rows, diagnostics = backflip.evaluate(policy, paths, cfg, output)
            else:
                rows = run_benchmark(
                    policy,
                    [str(p) for p in paths],
                    num_envs=options["num_envs"],
                    device="cuda:0",
                    future_frames=cfg["env"]["future_frames"],
                    anchor_obs=cfg["env"].get("anchor_obs", False),
                )
            if len(rows) != len(paths) or any(
                not math.isfinite(x.norm_return) for x in rows
            ):
                raise ValueError("Evaluation returned missing/nonfinite clips")
            save_results(rows, output / "evaluation.json")
            summary = summarize(rows)
            summary["full_clip_zero_padded_return"] = sum(
                x.norm_return * x.steps for x in rows
            ) / sum(x.frames for x in rows)
            record.update(
                summary=summary,
                scope="Training-clip coverage; no held-out generalization claim",
            )
            if options["task"] == "backflip":
                record.update(
                    skill_success=diagnostics["skill_success"],
                    backflip_diagnostics=diagnostics,
                    strict_backflip_acceptance="not_assessed",
                    scope="Single backflip reference plus5sadditional hold. Return/clip termination alone do not validate full rotation, airborne motion, symmetry or contact-force quality.",
                )
        else:
            from mjlab.rl import RslRlVecEnvWrapper
            from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
            from supermimic.envs import SuperMimicEnvCfgArgs, make_env

            clip = next((p for p in paths if p.stem == options["clip"]), None)
            if clip is None:
                raise ValueError(f"Motion clip not found: {options['clip']}")
            args = SuperMimicEnvCfgArgs(
                npz_files=[str(clip)],
                num_envs=options["num_envs"],
                future_frames=cfg["env"]["future_frames"],
                episode_length_s=1e5,
                action_noise_std=0.0,
                obs_noise=False,
                reset_noise=False,
                domain_randomization=False,
                sampling_mode="start",
                record_transitions=False,
                seed=0,
                anchor_obs=cfg["env"].get("anchor_obs", False),
                debug_vis=True,
            )
            env = RslRlVecEnvWrapper(make_env(args, device="cuda:0"))

            @torch.no_grad()
            def act(obs):
                return policy(obs["policy"], obs["goal"])

            viewer = (
                NativeMujocoViewer if options["viewer"] == "native" else ViserPlayViewer
            )
            viewer(env, act).run()
        validate_motion_snapshot(inputs["root"], inputs["files"])
        validate_snapshot(
            implementation / "supermimic", options["implementation_manifest"]
        )
        if (
            snapshot_file_hashes(options["checkpoint"])[0]
            != options["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint changed during SDK execution")
        for name, expected in record["g1_assets"].items():
            if snapshot_file_hashes(asset_root / name)[0] != expected:
                raise ValueError(f"G1 asset changed during SDK execution: {name}")
        record["status"] = "complete"
        print(json.dumps(record.get("summary", {"status": "complete"})), flush=True)
        return 0
    except BaseException as error:
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        if env is not None:
            env.close()
        write_json(output / "result.json", record)


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        if arguments.backflip_config:
            config = json.loads(arguments.backflip_config.read_text())
            result = backflip.prepare(config["workspace"], config["root"])
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "training_frames": result["training_frames"],
                        "evaluation_frames": result["evaluation_frames"],
                        "method": "SuperMimic",
                    }
                ),
                flush=True,
            )
            return 0
        if arguments.recipe == "reference" and (
            arguments.task != "backflip" or arguments.anchor_obs
        ):
            raise ValueError(
                "The reference recipe requires --task backflip and its own99dimensional observation protocol"
            )
        if arguments.worker_config:
            return worker(arguments.worker_config)
        if arguments.fresh and arguments.action != "train":
            raise ValueError("--fresh applies only to train")
        if arguments.anchor_obs and arguments.action != "train":
            raise ValueError(
                "--anchor-obs applies only to train; play/eval use checkpoint configuration"
            )
        if arguments.fresh and arguments.checkpoint:
            raise ValueError("--fresh cannot be combined with --checkpoint")
        if arguments.action == "prepare" and arguments.task != "backflip":
            raise ValueError("prepare currently requires --task backflip")
        if arguments.task == "backflip" and arguments.motions is not None:
            raise ValueError(
                "The backflip task uses its pinned reference; custom --motions belong to the generic task"
            )
        if (
            arguments.action == "train"
            and arguments.profile == "smoke"
            and not arguments.fresh
            and arguments.checkpoint is None
            and arguments.task != "backflip"
        ):
            raise ValueError("The small smoke architecture requires --fresh")
        if (
            arguments.action == "train"
            and arguments.num_envs is not None
            and arguments.num_envs < 8
        ):
            raise ValueError("Training requires at least8environments")
        options = resolve_options(arguments)
        if arguments.dry_run:
            plan = {k: v for k, v in options.items() if k != "expected_motions"}
            plan["expected_motion_count"] = len(options["expected_motions"] or {})
            print(json.dumps(plan, indent=2))
            return 0
        return launch(options)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"SuperMimic: {error}", file=sys.stderr)
        return 2
