"""Keep the new complete recipe isolated from the older checkpoint protocol."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from embodiedforge import supermimic_run as runner
from embodiedforge.workflows import supermimic_backflip as backflip
from embodiedforge.workflows import supermimic_reference as reference
from embodiedforge.workflows.supermimic_reference_evaluate import validate_checkpoint


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "strided"])
@pytest.mark.parametrize("count", [0, 1, 5])
def test_actual_state_owns_simulator_inputs_and_preserves_gradients(
    dtype_name, layout, count
):
    torch = pytest.importorskip("torch")
    from embodiedforge.workflows.supermimic_reference_evaluate import actual_state

    def value(width, offset):
        result = (
            torch.arange(count * width, dtype=getattr(torch, dtype_name)).reshape(
                count, width
            )
            + offset
        )
        if layout == "transpose":
            result = result.T.contiguous().T
        elif layout == "strided":
            padded = torch.zeros(count, width * 2, dtype=result.dtype)
            padded[:, ::2] = result
            result = padded[:, ::2]
        return result.detach().requires_grad_()

    qpos, qvel, origins = value(11, 0.25), value(10, -0.25), value(3, 0.125)
    originals = [tensor.detach().clone() for tensor in (qpos, qvel, origins)]
    env = SimpleNamespace(
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos, qvel=qvel)), origins=origins
    )
    result = actual_state(env)
    expected = torch.cat(originals[:2], -1)
    expected[:, :3] -= originals[2]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(result.sum(), (qpos, qvel, origins))
    for gradient, source, sign in zip(
        gradients, (qpos, qvel, origins), (1, 1, -1), strict=True
    ):
        torch.testing.assert_close(gradient, torch.full_like(source, sign))
    result.detach().fill_(-99)
    for tensor, original in zip((qpos, qvel, origins), originals, strict=True):
        torch.testing.assert_close(tensor.detach(), original, rtol=0, atol=0)


def registry(workspace, recipe):
    checkpoint = workspace / (recipe + ".pt")
    checkpoint.write_bytes(recipe.encode())
    directory = backflip.task_directory(workspace)
    if recipe == "reference":
        directory /= "reference"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "current.json").write_text(
        json.dumps(
            {
                "task": "backflip",
                "recipe": recipe,
                "training_status": "complete",
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }
        )
    )
    return checkpoint


def test_reference_recipe_never_resumes_the_older_cuboi_policy(tmp_path):
    registry(tmp_path, "cuboi")
    args = runner.parser().parse_args(
        ["train", "--task", "backflip", "--recipe", "reference"]
    )
    plan = runner.resolve_options(args, tmp_path)
    assert plan["fresh"] is True and plan["checkpoint"] is None
    assert plan["recipe"] == "reference"
    assert plan["motions"].endswith("backflip/data/train")


def test_reference_registry_is_separate_and_hash_checked(tmp_path):
    older = registry(tmp_path, "cuboi")
    newer = registry(tmp_path, "reference")
    args = runner.parser().parse_args(
        ["eval", "--task", "backflip", "--recipe", "reference"]
    )
    assert runner.resolve_options(args, tmp_path)["checkpoint"] == str(newer)
    args = runner.parser().parse_args(["eval", "--task", "backflip"])
    assert runner.resolve_options(args, tmp_path)["checkpoint"] == str(older)
    newer.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint changed"):
        reference.latest_checkpoint(tmp_path)
    assert backflip.latest_checkpoint(tmp_path)["checkpoint"] == str(older)


@pytest.mark.parametrize("extra", [[], ["--task", "backflip", "--anchor-obs"]])
def test_reference_protocol_rejects_incompatible_options_before_sdk_load(extra, capsys):
    assert runner.main(["train", "--recipe", "reference", *extra]) == 2
    assert "requires --task backflip" in capsys.readouterr().err


def test_reference_target_must_leave_policy_updates_after_warmup(tmp_path):
    args = runner.parser().parse_args(
        ["train", "--task", "backflip", "--recipe", "reference", "--iterations", "500"]
    )
    plan = runner.resolve_options(args, tmp_path)
    with pytest.raises(ValueError, match="policy updates"):
        reference.launch(plan, {})
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("source", ["no matching text", "target target"])
def test_pinned_adapter_refuses_missing_or_ambiguous_patch_targets(source):
    with pytest.raises(ValueError, match="match once"):
        reference.replace_once(source, "target", "replacement")


def checkpoint_metadata():
    return {
        "pi": {},
        "wm": {},
        "args": {"hist": 10, "goal_frames": 10},
        "iteration": 6000,
    }


def test_reference_checkpoint_layout_check_rejects_the_cuboi_format():
    with pytest.raises(ValueError, match="not a Cuboi"):
        validate_checkpoint({"config": {}, "policy": {}, "iteration": 6000})
    assert validate_checkpoint(checkpoint_metadata()) == {"hist": 10, "goal_frames": 10}


def test_small_inference_snapshot_cannot_be_used_as_a_training_resume():
    with pytest.raises(ValueError, match="full last.pt"):
        validate_checkpoint(checkpoint_metadata(), training=True, iterations=7000)


def test_full_resume_requires_a_later_global_target():
    saved = checkpoint_metadata()
    saved.update(critic={}, opt_wm={}, opt_pi={}, opt_v={})
    with pytest.raises(ValueError, match="global"):
        validate_checkpoint(saved, training=True, iterations=6000)
    assert validate_checkpoint(saved, training=True, iterations=6001) is saved["args"]


@pytest.mark.parametrize("field", ["hist", "goal_frames"])
def test_invalid_reference_history_or_goal_is_rejected(field):
    saved = checkpoint_metadata()
    saved["args"][field] = 0
    with pytest.raises(ValueError, match="history/goal"):
        validate_checkpoint(saved)


def test_preference_selects_recipe_but_keeps_explicit_legacy_access(tmp_path):
    old = registry(tmp_path, "cuboi")
    current = registry(tmp_path, "reference")
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    reference.activate_preference(
        tmp_path, basis_report="docs/experiments/example.json", checkpoint_sha256=digest
    )
    default = runner.resolve_options(
        runner.parser().parse_args(["train", "--task", "backflip"]), tmp_path
    )
    assert default["recipe"] == "reference" and default["checkpoint"] == str(current)
    assert default["num_envs"] == 1024
    explicit = runner.resolve_options(
        runner.parser().parse_args(["eval", "--task", "backflip", "--recipe", "cuboi"]),
        tmp_path,
    )
    assert explicit["recipe"] == "cuboi" and explicit["checkpoint"] == str(old)


def test_recipe_preference_is_not_a_model_selection_fallback(tmp_path):
    current = registry(tmp_path, "reference")
    digest = hashlib.sha256(current.read_bytes()).hexdigest()
    reference.activate_preference(
        tmp_path, basis_report="evaluated.json", checkpoint_sha256=digest
    )
    current.write_bytes(b"changed after activation")
    assert reference.preferred_recipe(tmp_path) == "reference"
    with pytest.raises(ValueError, match="checkpoint changed"):
        runner.resolve_options(
            runner.parser().parse_args(["eval", "--task", "backflip"]), tmp_path
        )


def test_preference_activation_requires_current_checkpoint_identity(tmp_path):
    registry(tmp_path, "reference")
    with pytest.raises(ValueError, match="identity"):
        reference.activate_preference(
            tmp_path, basis_report="evaluated.json", checkpoint_sha256="wrong"
        )
    assert not (backflip.task_directory(tmp_path) / "reference/preferred.json").exists()


def test_reference_registry_rejects_another_task(tmp_path):
    registry(tmp_path, "reference")
    path = backflip.task_directory(tmp_path) / "reference/current.json"
    data = json.loads(path.read_text())
    data["task"] = "lafan"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="registry"):
        reference.latest_checkpoint(tmp_path)


def test_registry_supports_outputs_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    checkpoint = tmp_path / "external-output/last.pt"
    assert reference.checkpoint_registry_path(checkpoint, workspace) == str(checkpoint)
    assert (
        reference.checkpoint_registry_path(workspace / "runs/last.pt", workspace)
        == "runs/last.pt"
    )


def test_auto_reference_preference_rejects_anchor_observation(tmp_path):
    checkpoint = registry(tmp_path, "reference")
    reference.activate_preference(
        tmp_path,
        basis_report="evaluated.json",
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="observation protocol"):
        runner.resolve_options(
            runner.parser().parse_args(["train", "--task", "backflip", "--anchor-obs"]),
            tmp_path,
        )


@pytest.mark.parametrize("action", ["eval", "play"])
def test_reference_single_episode_evaluation_rejects_ignored_batch_count(
    tmp_path, action
):
    plan = runner.resolve_options(
        runner.parser().parse_args(
            [action, "--task", "backflip", "--recipe", "reference", "--num-envs", "2"]
        ),
        tmp_path,
    )
    with pytest.raises(ValueError, match="one environment"):
        reference.launch(plan, {})
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("destination", ["cuboi", "reference"])
def test_registry_cannot_move_weights_across_recipe_namespaces(tmp_path, destination):
    registry(tmp_path, "cuboi")
    registry(tmp_path, "reference")
    root = backflip.task_directory(tmp_path)
    cuboi, full = root / "current.json", root / "reference/current.json"
    target, foreign = (cuboi, full) if destination == "cuboi" else (full, cuboi)
    target.write_bytes(foreign.read_bytes())
    with pytest.raises(ValueError, match="registry"):
        backflip.latest_checkpoint(tmp_path, recipe=destination)
