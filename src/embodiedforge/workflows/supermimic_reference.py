"""Freeze and adapt the complete vibed algorithm to the existing mjlab SDK."""

import ast
import json
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import copy_file_snapshot, snapshot_file_hashes
from embodiedforge.runtime.process import run_process
from embodiedforge.runtime.provenance import snapshot_implementation
from embodiedforge.runtime.records import run_identity, run_lifecycle
from embodiedforge.supermimic import source_identity
from embodiedforge.workflows import supermimic_backflip as backflip
from embodiedforge.workflows.supermimic_catalog import REFERENCES


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Pinned adapter replacement must match once: {old[:80]}")
    return source.replace(old, new)


def adapt_environment(source):
    """Keep upstream sampling, noise and accounting; replace physical API calls."""
    source = source[source.index("class TrackingSim:") :]
    start = source.index("    def __init__(")
    stop = source.index("    def sample_frames(")
    source = (
        source[:start]
        + """    def __init__(self, sim_cfg, num_envs, motion, mdp, gamma=0.99, eval_mode=False,
                 max_episode_steps=500, reset_noise=True):
        initialize(self, sim_cfg, num_envs, motion, mdp, gamma, eval_mode, max_episode_steps, reset_noise)

    def state(self):
        return read_state(self)

"""
        + source[stop:]
    )
    start = source.index("        quat = mat_to_quat(R)")
    stop = source.index("        self.f[env_ids] = frames", start)
    source = (
        source[:start]
        + "        write_state(self, env_ids, p, R, v, w, q, qd)\n"
        + source[stop:]
    )
    source = replace_once(
        source, "        self.scene.reset(env_ids)", "        refresh(self)"
    )
    start = source.index("    def refresh(self):")
    stop = source.index("    def step(self, a:", start)
    source = (
        source[:start]
        + "    def refresh(self):\n        refresh(self)\n\n"
        + source[stop:]
    )
    start = source.index("        target = self.default_q")
    stop = source.index("        S1 = self.state()", start)
    source = source[:start] + "        step_physics(self, a)\n" + source[stop:]
    return (
        '''"""MIT upstream TrackingSim accounting with an explicit mjlab physical bridge."""
import torch
from .mdp import MAX_STEP_REWARD, flatten_state
from .rotations import so3_exp
from embodiedforge.workflows.supermimic_reference_mjlab import (
    initialize, read_state, write_state, step_physics, refresh, make_sim,
)

'''
        + source
    )


def adapt_robot(source):
    source = source[source.index("ARMATURE_5020 =") : source.index("def _actuators()")]
    return (
        '''"""MIT upstream G1 drive constants; use the current mjlab G1 MJCF."""
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
MJCF_PATH = str(G1_XML)

'''
        + source
    )


def adapt_training(source):
    source = replace_once(
        source,
        "from isaaclab.app import add_launcher_args, launch_simulation",
        "from embodiedforge.workflows.supermimic_reference_mjlab import add_launcher_args, launch_simulation",
    )
    source = replace_once(
        source,
        'parser.add_argument("--resume",',
        'parser.add_argument("--max_policy_minutes", type=float, default=90.0)\nparser.add_argument("--resume",',
    )
    source = replace_once(
        source,
        "motion = MotionLib.from_dir(args.motion_dir, kin, dev)",
        'import glob\nmotion = MotionLib(sorted(glob.glob(os.path.join(args.motion_dir, "*.csv"))), kin, dev, src_fps=50.0)',
    )
    source = replace_once(
        source,
        "    for it in range(start_it, args.iterations):",
        """    policy_budget_start = None
    for it in range(start_it, args.iterations):
        if it >= args.wm_only_iters and policy_budget_start is None:
            policy_budget_start = time.monotonic()
            print("POLICY_BUDGET_START", flush=True)""",
    )
    source = replace_once(
        source,
        "        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations:",
        """        budget_done = policy_budget_start is not None and time.monotonic() - policy_budget_start >= 60 * args.max_policy_minutes
        if (it + 1) % args.ckpt_every == 0 or it + 1 == args.iterations or budget_done:""",
    )
    source += """
        if budget_done:
            print("BOUNDED_TRAINING_FINISHED", it + 1, flush=True)
            break
    log_f.close()
    modules = (wm, pi, critic)
    assert all(torch.isfinite(value).all() for module in modules for value in module.state_dict().values())
    reloaded = Policy(OBS_DIM, GOAL_DIM, hist=Hh).to(dev)
    saved = torch.load(os.path.join(args.log_dir, "last.pt"), map_location=dev, weights_only=False)
    reloaded.load_state_dict(saved["pi"])
    obs_check = torch.linspace(-.1, .1, 3*Hh*OBS_DIM, device=dev).reshape(3,Hh,OBS_DIM)
    goal_check = torch.linspace(-.1, .1, 3*GOAL_DIM, device=dev).reshape(3,GOAL_DIM)
    with torch.no_grad():
        assert torch.equal(pi(obs_check, goal_check), reloaded(obs_check, goal_check))
    json.dump({"status":"complete", "iteration":saved["iteration"], "sim_steps":saved["sim_steps"],
               "finite_model_tensors":True, "checkpoint_reload_action_parity":True,
               "wm_only_iterations":args.wm_only_iters, "policy_updates":max(0, saved["iteration"]-max(start_it,args.wm_only_iters)),
               "policy_budget_seconds":None if policy_budget_start is None else time.monotonic()-policy_budget_start},
              open(os.path.join(args.log_dir,"training-integrity.json"),"w"),indent=2)
"""
    source = replace_once(
        source,
        "    if start_it == 0:\n        log_f.write(",
        "    if log_f.tell() == 0:\n        log_f.write(",
    )
    # Resume restores weights and optimizers, but the upstream replay is not saved.
    source = replace_once(
        source,
        "    if start_it == 0:\n        collect(args.warmup_steps, stats)",
        "    collect(args.warmup_steps, stats)",
    )
    return source


def freeze_reference(workspace, output):
    workspace, output = Path(workspace), Path(output)
    root = workspace / ".cache/external/supermimic-vibed"
    ref = REFERENCES["vibed"]
    files = ref["source_files"] + [
        str(p.relative_to(root)) for p in (root / "supermimic").glob("*.py")
    ]
    source = source_identity(root, ref["revision"], sorted(set(files)))
    implementation = output / "implementation/supermimic"
    original = snapshot_implementation(root / "supermimic", implementation)
    for name, digest in original["files"].items():
        if source["files"].get("supermimic/" + name) != digest:
            raise ValueError(f"Reference snapshot differs from pinned source: {name}")
    (implementation / "__init__.py").write_text(
        '"""Pinned independent SuperMimic with mjlab physics adapter."""\n'
    )
    for name, adapter in (
        ("env.py", adapt_environment),
        ("robot.py", adapt_robot),
        ("train.py", adapt_training),
    ):
        path = implementation / name
        text = adapter(path.read_text())
        ast.parse(text)
        path.write_text(text)
    copy_file_snapshot(root / "LICENSE", output / "implementation/LICENSE")
    adapted = {
        str(p.relative_to(implementation)): snapshot_file_hashes(p)[0]
        for p in sorted(implementation.glob("*.py"))
    }
    record = {
        "source": source,
        "original_manifest": original,
        "adapted_manifest": {"files": adapted},
        "method": "Complete vibed SuperMimic world-model/BPTT/critic algorithm with mjlab physical bridge",
        "scope": "Current mjlab G1 assets, solver and reduced batches differ from the reference IsaacLab3 experiment. No official model weights or forced action symmetry.",
    }
    write_json(output / "reference-adapter.json", record)
    return record


def latest_checkpoint(workspace):
    return backflip.latest_checkpoint(workspace, recipe="reference")


def launch(options, environment):
    workspace, output = Path(options["workspace"]), Path(options["output"])
    environment = dict(environment)
    environment.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    warmup = 4 if options["profile"] == "smoke" else 500
    if (
        options["action"] == "train"
        and options["fresh"]
        and options["iterations"] <= warmup
    ):
        raise ValueError(
            "Reference training target must leave actual policy updates after world-model-only warmup"
        )
    if options["action"] in {"eval", "play"} and options["num_envs"] != 1:
        raise ValueError("Reference physical evaluation/play requires one environment")
    if options["action"] == "prepare":
        print(
            json.dumps(
                backflip.prepared(backflip.task_directory(workspace) / "data"), indent=2
            )
        )
        return 0
    ref = REFERENCES["vibed"]
    checkout = workspace / ".cache/external/supermimic-vibed"
    source_identity(checkout, ref["revision"], ref["source_files"])
    if options["action"] == "doctor":
        print(
            json.dumps(
                {
                    "status": "local_inputs_verified",
                    "recipe": "reference",
                    "source_revision": ref["revision"],
                    "checkpoint": options["checkpoint"],
                    "cuda_checked": False,
                },
                indent=2,
            )
        )
        return 0
    checkpoint = options["checkpoint"]
    if checkpoint is not None:
        actual = snapshot_file_hashes(checkpoint)[0]
        if (
            options["expected_checkpoint_sha256"] is not None
            and actual != options["expected_checkpoint_sha256"]
        ):
            raise ValueError(
                "Reference-recipe checkpoint hash differs from the registry"
            )
        options["checkpoint_sha256"] = actual
    elif options["action"] in {"eval", "play"}:
        raise ValueError(
            "No completed reference-recipe model yet; train --task backflip --recipe reference first"
        )
    output.mkdir(parents=True, exist_ok=False)
    sources = (
        "supermimic",
        "src/embodiedforge/supermimic_run.py",
        "src/embodiedforge/workflows/supermimic_backflip.py",
        "src/embodiedforge/workflows/supermimic_reference.py",
        "src/embodiedforge/workflows/supermimic_reference_mjlab.py",
        "src/embodiedforge/workflows/supermimic_reference_worker.py",
        "benchmarks/supermimic_reference.py",
        "src/embodiedforge/workflows/supermimic_contacts.py",
        "src/embodiedforge/workflows/supermimic_backflip_quality.py",
        "src/embodiedforge/workflows/supermimic_reference_play.py",
        "src/embodiedforge/workflows/supermimic_reference_evaluate.py",
    )
    guards = {name: snapshot_file_hashes(workspace / name)[0] for name in sources}
    record = {
        "status": "preparing",
        "recipe": "reference",
        "action": options["action"],
        "official_weights_used": False,
        "run_contract": run_identity("supermimic", options["action"], "backflip"),
        "launcher_sources": guards,
        "worker_exit_observed": False,
    }
    try:
        with run_lifecycle(
            record,
            lambda current: write_json(output / "launcher.json", current),
            label=output / "launcher.json",
        ):
            config = output / "worker-config.json"
            write_json(config, options)
            command = [
                options["python"],
                "-I",
                str(workspace / "benchmarks/supermimic_reference.py"),
                "--config",
                str(config),
            ]
            record.update(status="running", command=command)
            write_json(output / "launcher.json", record)
            print(
                f"SuperMimic reference/mjlab {options['action']} | output: {output}",
                flush=True,
            )
            run_process(
                command, cwd=workspace, env=environment, log_path=output / "command.log"
            )
            record.update(worker_exit_observed=True, worker_returncode=0)
            worker = json.loads((output / "result.json").read_text())
            if worker["status"] != "complete":
                raise ValueError("Reference-recipe worker validation is incomplete")
            for name, digest in guards.items():
                if snapshot_file_hashes(workspace / name)[0] != digest:
                    raise ValueError(
                        f"Source changed during reference-recipe execution: {name}"
                    )
            if options["action"] == "train" and options["profile"] != "smoke":
                if (
                    not worker["finite_model_tensors"]
                    or not worker["checkpoint_reload_action_parity"]
                    or worker["policy_updates"] <= 0
                ):
                    raise ValueError(
                        "Cannot register an unverified reference-recipe model"
                    )
                target = backflip.task_directory(workspace) / "reference"
                target.mkdir(parents=True, exist_ok=True)
                write_json(
                    target / "current.json",
                    {
                        "task": "backflip",
                        "recipe": "reference",
                        "training_status": "complete",
                        "checkpoint": checkpoint_registry_path(
                            output / "logs/last.pt", workspace
                        ),
                        "checkpoint_sha256": worker["checkpoint_sha256"],
                        "iteration": worker["iteration"],
                        "skill_success": None,
                        "selection": "Actual last checkpoint of a completed full-width reference-recipe run; not selected by return",
                    },
                )
            return 0
    except subprocess.CalledProcessError as error:
        return error.returncode if error.returncode > 0 else 128 - error.returncode
    except KeyboardInterrupt as error:
        return 128 + getattr(error, "signum", signal.SIGINT)


def checkpoint_registry_path(checkpoint, workspace):
    """Keep external output directories usable by the same local registry."""
    checkpoint, workspace = Path(checkpoint).resolve(), Path(workspace).resolve()
    return str(
        checkpoint.relative_to(workspace)
        if checkpoint.is_relative_to(workspace)
        else checkpoint
    )


def preferred_recipe(workspace):
    """Select the explicitly activated algorithm family; registries remain separate."""
    marker = backflip.task_directory(workspace) / "reference/preferred.json"
    if not marker.exists():
        return "cuboi"
    record = json.loads(marker.read_text())
    if record.get("task") != "backflip" or record.get("recipe") != "reference":
        raise ValueError("Invalid preferred backflip recipe")
    return "reference"


def activate_preference(workspace, *, basis_report, checkpoint_sha256):
    """Activate the recipe after a caller has evaluated its actual current model."""
    current = latest_checkpoint(workspace)
    if current is None or current["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError(
            "Preferred-recipe activation requires the current checkpoint identity"
        )
    target = backflip.task_directory(workspace) / "reference/preferred.json"
    write_json(
        target,
        {
            "task": "backflip",
            "recipe": "reference",
            "basis_report": basis_report,
            "evaluated_checkpoint_sha256": checkpoint_sha256,
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "scope": "Prefer this algorithm family; subsequent train commands resume its latest completed model. This marker does not certify later models.",
        },
    )
