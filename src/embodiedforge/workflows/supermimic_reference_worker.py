"""Run the complete independent SuperMimic recipe in the dedicated SDK."""

import importlib.metadata
import json
import runpy
import sys
from pathlib import Path

from embodiedforge.runtime.artifacts import write_json
from embodiedforge.runtime.hashing import snapshot_file_hashes
from embodiedforge.runtime.provenance import validate_snapshot
from embodiedforge.workflows import supermimic_backflip as backflip
from embodiedforge.workflows.supermimic_catalog import REFERENCES
from embodiedforge.workflows.supermimic_reference import freeze_reference
from embodiedforge.workflows.supermimic_source import matches_version


def main(config):
    options = json.loads(Path(config).read_text())
    output, workspace = Path(options["output"]), Path(options["workspace"])
    record = {
        "status": "preparing",
        "recipe": "reference",
        "method": "SuperMimic",
        "backend": "mjlab",
        "official_weights_used": False,
    }
    write_json(output / "result.json", record)
    try:
        reference = REFERENCES["cuboi-reproduced"]
        versions = {
            name: importlib.metadata.version(name) for name in reference["dependencies"]
        }
        if sys.version_info[:2] != (3, 12) or any(
            not matches_version(versions[name], constraint)
            for name, constraint in reference["dependencies"].items()
        ):
            raise ValueError(
                "Reference/mjlab bridge requires the existing pinned SuperMimic SDK"
            )
        origin = json.loads(
            importlib.metadata.distribution("mjlab").read_text("direct_url.json")
        )
        if origin["vcs_info"]["commit_id"] != reference["framework"]["revision"]:
            raise ValueError("mjlab framework revision changed")
        data = backflip.task_directory(workspace) / "data"
        provenance = backflip.prepared(data)
        canonical = (data / "backflip.csv").read_bytes()
        if snapshot_file_hashes(data / "backflip.csv")[0] != backflip.CSV_SHA256:
            raise ValueError("Canonical backflip CSV changed")
        extended = options["action"] == "eval"
        lines = canonical.splitlines(keepends=True)
        if len(lines) != 321 or not canonical.endswith(b"\n"):
            raise ValueError("Expected321canonical50Hzframes")
        inputs = output / "inputs"
        inputs.mkdir()
        csv = inputs / "backflip.csv"
        csv.write_bytes(canonical + (lines[-1] * 250 if extended else b""))
        csv.chmod(0o444)
        csv_sha = snapshot_file_hashes(csv)[0]
        manifest = freeze_reference(workspace, output)
        sys.path.insert(0, str(output / "implementation"))
        import supermimic
        import torch
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

        from embodiedforge.workflows.supermimic_reference_evaluate import (
            validate_checkpoint,
        )

        if (
            Path(supermimic.__file__).resolve().parent
            != (output / "implementation/supermimic").resolve()
        ):
            raise ValueError(
                "SuperMimic import did not resolve to the frozen implementation"
            )
        if options["checkpoint"] is not None:
            if (
                snapshot_file_hashes(options["checkpoint"])[0]
                != options["checkpoint_sha256"]
            ):
                raise ValueError("Checkpoint changed before CPU validation")
            saved = torch.load(
                options["checkpoint"], map_location="cpu", weights_only=False
            )
            validate_checkpoint(
                saved,
                training=options["action"] == "train",
                iterations=options["iterations"],
            )
            del saved
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise ValueError("Reference recipe requires CUDA")
        assets = {
            str(p.relative_to(G1_XML.parent.parent)): snapshot_file_hashes(p)[0]
            for p in sorted(G1_XML.parent.parent.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        }
        record.update(
            environment={
                "python": sys.version,
                "packages": versions,
                "mjlab_origin": origin,
                "gpu": torch.cuda.get_device_name(0),
            },
            g1_assets=assets,
            motion_input_snapshot={
                "canonical_sha256": backflip.CSV_SHA256,
                "source_sha256": provenance["source_sha256"],
                "private_csv_sha256": csv_sha,
                "frames": 571 if extended else 321,
                "fps": 50,
                "read_only": True,
            },
            reference_adapter=manifest,
        )
        if (
            options["checkpoint"] is not None
            and snapshot_file_hashes(options["checkpoint"])[0]
            != options["checkpoint_sha256"]
        ):
            raise ValueError("Checkpoint changed before SDK execution")
        write_json(output / "result.json", record)
        if options["action"] == "train":
            smoke = options["profile"] == "smoke"
            script = output / "implementation/supermimic/train.py"
            if smoke:
                source = script.read_text().replace(
                    "torch.compile(", "(lambda function: function)("
                )
                script.write_text(source)
                manifest["adapted_manifest"]["files"]["train.py"] = (
                    snapshot_file_hashes(script)[0]
                )
                write_json(output / "reference-adapter.json", manifest)
            argv = [
                str(script),
                "--num_envs",
                str(options["num_envs"]),
                "--iterations",
                str(options["iterations"]),
                "--motion_dir",
                str(inputs),
                "--log_dir",
                str(output / "logs"),
                "--seed",
                str(options["seed"]),
                "--max_policy_minutes",
                str(options["hours"] * 60),
                "--wm_batch",
                str(32 if smoke else min(options["num_envs"], 1024)),
                "--pi_batch",
                str(8 if smoke else min(options["num_envs"], 1024)),
                "--warmup_steps",
                "20" if smoke else "64",
                "--wm_only_iters",
                "4" if smoke else "500",
                "--wm_batches",
                "2" if smoke else "10",
                "--wm_window",
                "2" if smoke else "8",
                "--horizon",
                "4" if smoke else "32",
                "--ckpt_every",
                "1000",
                "--log_every",
                "1" if smoke else "50",
            ]
            if options["checkpoint"]:
                argv += ["--resume", options["checkpoint"]]
            record.update(status="training", training_command=argv, smoke_only=smoke)
            write_json(output / "result.json", record)
            sys.argv = argv
            runpy.run_path(str(script), run_name="__main__")
            integrity = json.loads(
                (output / "logs/training-integrity.json").read_text()
            )
            if integrity["policy_updates"] <= 0:
                raise ValueError("No actual SuperMimic policy updates executed")
            record.update(
                integrity,
                checkpoint_sha256=snapshot_file_hashes(output / "logs/last.pt")[0],
            )
        else:
            from embodiedforge.workflows.supermimic_reference_evaluate import (
                evaluate,
                play,
            )

            execute = evaluate if options["action"] == "eval" else play
            record.update(
                execute(options, output, csv),
                checkpoint_sha256=options["checkpoint_sha256"],
            )
        validate_snapshot(
            output / "implementation/supermimic", manifest["adapted_manifest"]
        )
        if snapshot_file_hashes(csv)[0] != csv_sha:
            raise ValueError("Private backflip input changed during execution")
        if (
            options["checkpoint"] is not None
            and snapshot_file_hashes(options["checkpoint"])[0]
            != options["checkpoint_sha256"]
        ):
            raise ValueError("Resume/evaluation checkpoint changed during execution")
        for name, expected in assets.items():
            if snapshot_file_hashes(G1_XML.parent.parent / name)[0] != expected:
                raise ValueError(f"G1 asset changed during execution: {name}")
        record["status"] = "complete"
        print(
            json.dumps(
                {
                    "status": "complete",
                    "iteration": record["iteration"],
                    "recipe": "reference",
                    **(
                        {
                            "skill_success": record["skill_success"],
                            "physical_quality": record["physical_quality"],
                        }
                        if "physical_quality" in record
                        else {}
                    ),
                }
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "result.json", record)
