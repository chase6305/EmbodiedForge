"""Warm-start the separate +9-anchor experiment from a trusted local checkpoint.

Zero new input columns and Adam moments preserve the initial policy/critic
function within measured arithmetic tolerances. This changes the observation
protocol and is NOT a paper-observation checkpoint conversion for deployment.
"""

import argparse
import copy
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from embodiedforge.runtime.artifacts import write_json
    from embodiedforge.runtime.hashing import snapshot_file_hashes
    from embodiedforge.runtime.provenance import (
        snapshot_implementation,
        validate_snapshot,
    )
    from embodiedforge.supermimic import source_identity
    from embodiedforge.workflows.supermimic_catalog import REFERENCES

    ref = REFERENCES["cuboi-reproduced"]
    required = ref["source_files"] + [
        str(p.relative_to(args.root))
        for p in (args.root / "src/supermimic").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    source = source_identity(args.root, ref["revision"], required)
    frozen = args.output / "implementation/supermimic"
    manifest = snapshot_implementation(args.root / "src/supermimic", frozen)
    sys.path.insert(0, str(frozen.parent.resolve()))
    import torch
    from supermimic.policy.nets import Critic, CriticCfg, Policy, PolicyCfg

    source_sha = snapshot_file_hashes(args.checkpoint)[0]
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (
        original["config"]["env"]["anchor_obs"]
        or original["config"]["policy"]["goal_dim"] != 680
    ):
        raise ValueError("Expected the local paper-observation680-d goal checkpoint")
    checkpoint = copy.deepcopy(original)
    pcfg = PolicyCfg(**original["config"]["policy"])
    ccfg = CriticCfg(**original["config"]["critic"])
    old_policy = Policy(pcfg)
    old_critic = Critic(ccfg, pcfg=pcfg)
    old_policy.load_state_dict(original["policy"])
    old_critic.load_state_dict(original["critic"])

    def widen(value, left):
        assert value.ndim == 2 and 0 <= left <= value.shape[1]
        result = value.new_zeros((value.shape[0], value.shape[1] + 9))
        result[:, :left] = value[:, :left]
        result[:, left + 9 :] = value[:, left:]
        return result

    expanded = {}
    for section, opt_name, module, key, left in (
        ("policy", "policy_opt", old_policy, "encoders.motion.mlp.net.0.weight", 680),
        ("critic", "critic_opt", old_critic, "mlp.net.0.weight", 71 + 680),
    ):
        old_weight = checkpoint[section][key]
        checkpoint[section][key] = widen(old_weight, left)
        names = [name for name, _ in module.named_parameters()]
        opt = checkpoint[opt_name]
        ids = [item for group in opt["param_groups"] for item in group["params"]]
        assert len(ids) == len(names)
        state = opt["state"][ids[names.index(key)]]
        assert "exp_avg" in state and "exp_avg_sq" in state
        for name, value in list(state.items()):
            if isinstance(value, torch.Tensor) and value.shape == old_weight.shape:
                state[name] = widen(value, left)
                assert not torch.count_nonzero(state[name][:, left : left + 9])
        scale_key = "encoders.motion.scale" if section == "policy" else "goal_scale"
        scale = checkpoint[section][scale_key]
        checkpoint[section][scale_key] = torch.cat([scale, scale.new_ones(9)])
        expanded[section] = {
            "parameter": key,
            "old_shape": list(old_weight.shape),
            "new_shape": list(checkpoint[section][key].shape),
            "insert_at": left,
            "new_adam_moments_zero": True,
        }
    checkpoint["config"]["env"]["anchor_obs"] = True
    checkpoint["config"]["policy"]["goal_dim"] = 689
    checkpoint["config"]["policy_trainer"]["anchor_obs"] = True
    new_cfg = PolicyCfg(**checkpoint["config"]["policy"])
    new_policy, new_critic = Policy(new_cfg), Critic(ccfg, pcfg=new_cfg)
    new_policy.load_state_dict(checkpoint["policy"])
    new_critic.load_state_dict(checkpoint["critic"])
    for key, value in original["wm"].items():
        assert torch.equal(value, checkpoint["wm"][key])

    errors = {}
    for device, dtype, tolerance in (
        ("cpu", torch.float64, 1e-10),
        ("cuda", torch.float32, 2e-5),
    ):
        torch.manual_seed(123)
        op, oc = old_policy.to(device, dtype), old_critic.to(device, dtype)
        np, nc = new_policy.to(device, dtype), new_critic.to(device, dtype)
        obs = torch.randn(16, 930, device=device, dtype=dtype) * 0.1
        goal = torch.randn(16, 680, device=device, dtype=dtype) * 0.2
        augmented = torch.cat(
            [goal, torch.randn(16, 9, device=device, dtype=dtype)], dim=-1
        )
        feat = torch.randn(16, 71, device=device, dtype=dtype)
        xi = torch.randn(16, 33, device=device, dtype=dtype)
        root_error = torch.randn(16, 4, device=device, dtype=dtype)
        with torch.no_grad():
            pi_error = float((op(obs, goal) - np(obs, augmented)).abs().max())
            value_error = float(
                (oc(feat, goal, xi, root_error) - nc(feat, augmented, xi, root_error))
                .abs()
                .max()
            )
        assert pi_error <= tolerance and value_error <= tolerance, (
            device,
            pi_error,
            value_error,
        )
        errors[device] = {
            "policy_max_abs_error": pi_error,
            "critic_max_abs_error": value_error,
            "tolerance": tolerance,
        }
    target = args.output / "anchor-initial.pt"
    torch.save(checkpoint, target)
    assert snapshot_file_hashes(args.checkpoint)[0] == source_sha
    validate_snapshot(frozen, manifest)
    source_identity(args.root, ref["revision"], required)
    proof = {
        "status": "complete",
        "source": source,
        "source_checkpoint_sha256": source_sha,
        "output_checkpoint_sha256": snapshot_file_hashes(target)[0],
        "iteration": checkpoint["iteration"],
        "updates": checkpoint["num_updates"],
        "expanded": expanded,
        "fixed_input_errors": errors,
        "world_model_tensors_unchanged": True,
        "scope": "Separate anchor-observation experiment. Numerical fixed-input agreement is not trajectory or training equivalence; RNG/replay are not restored.",
    }
    write_json(args.output / "result.json", proof)
    print(json.dumps(proof), flush=True)


if __name__ == "__main__":
    main()
