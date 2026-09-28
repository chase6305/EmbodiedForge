"""Workflow boundaries, terminal-frame evaluation, and optional SDK contracts."""

import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge import _weave_worker, weave
from embodiedforge._weave_evaluation import ClipEvaluation
from embodiedforge._weave_worker import (
    attach_evaluator,
    check_runtime,
    close_resource,
    evaluate_policy,
    learn_policy,
    load_checkpoint,
    synchronize_resume_optimizer,
)
from embodiedforge.recipes import snapshot_implementation, validate_snapshot


def test_worker_validates_implementation_before_loading_sdk(tmp_path, monkeypatch):
    package = tmp_path / "implementation" / "embodiedforge"
    manifest = snapshot_implementation(Path(_weave_worker.__file__).parent, package)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps({"output": str(tmp_path), "implementation": manifest})
    )

    def runtime_boundary(request):
        raise RuntimeError("Reached SDK runtime validation")

    monkeypatch.setattr(_weave_worker, "check_runtime", runtime_boundary)

    def run():
        monkeypatch.setattr(sys, "argv", ["worker", str(request_path)])
        _weave_worker.main()

    with pytest.raises(ValueError, match="must load its implementation snapshot"):
        run()
    monkeypatch.setattr(_weave_worker, "__file__", str(package / "_weave_worker.py"))
    with pytest.raises(RuntimeError, match="Reached SDK runtime validation"):
        run()
    (package / "_weave_evaluation.py").write_text("# changed after snapshot\n")
    with pytest.raises(ValueError, match="Weave implementation differs"):
        run()


def test_export_dependencies_are_checked_only_for_export(tmp_path, monkeypatch):
    versions = {
        "isaaclab": "2.3.0",
        "isaaclab-rl": "0.4.0",
        "rsl-rl-lib": "3.1.2",
        "torch": "2.10.0",
        "numpy": "1.26.4",
        "isaacsim": "5.1.0",
    }

    def version(name):
        if name not in versions:
            raise _weave_worker.importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(_weave_worker.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(_weave_worker.importlib.metadata, "version", version)
    monkeypatch.setattr(
        _weave_worker.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(tmp_path / name / "__init__.py")),
    )
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(optim=SimpleNamespace(Muon=object))
    )
    request = {"command": "train", "isaaclab_root": str(tmp_path)}
    for command in ("train", "evaluate"):
        request["command"] = command
        assert "onnx" not in check_runtime(request)
    request["command"] = "export"
    for name in ("onnx", "onnxscript"):
        with pytest.raises(RuntimeError, match=f"dependency: {name};"):
            check_runtime(request)
        versions[name] = "fixture"
    report = check_runtime(request)
    assert report["onnx"] == report["onnxscript"] == "fixture"


@pytest.mark.parametrize(
    ("error", "cleanup_error"),
    [(None, False), (None, True), (KeyboardInterrupt, True), (RuntimeError, True)],
)
def test_cleanup_attempts_all_resources_without_masking_failure(
    error, cleanup_error, capsys
):
    closed = []

    def resource(name):
        def close():
            closed.append(name)
            if cleanup_error:
                raise OSError(f"{name} close error")

        return SimpleNamespace(close=close)

    def learn(**kwargs):
        assert kwargs == {"num_learning_iterations": 20, "init_at_random_ep_len": True}
        if error:
            raise error("training stopped")

    runner = SimpleNamespace(learn=learn, writer=resource("training log"))
    env, app = resource("environment"), resource("simulation app")

    def run():
        try:
            try:
                learn_policy(runner, 20)
            finally:
                close_resource(env, "environment")
        finally:
            close_resource(app, "simulation app")

    expected = error or (OSError if cleanup_error else None)
    if expected:
        message = "training stopped" if error else "training log close error"
        with pytest.raises(expected, match=message):
            run()
        diagnostic = capsys.readouterr().err
        assert "Failed to close environment" in diagnostic
        assert "Failed to close simulation app" in diagnostic
        if error:
            assert "Failed to close training log" in diagnostic
    else:
        run()
        assert not capsys.readouterr().err
    assert closed == ["training log", "environment", "simulation app"]


def test_evaluation_hook_records_before_reset_without_observation_side_effects(
    monkeypatch,
):
    # Keep the test independent of the simulator; exercise the configured hook.
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.managers",
        SimpleNamespace(
            RewardTermCfg=SimpleNamespace,
            TerminationTermCfg=SimpleNamespace,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "g1_hoi_learning.tasks.hoi.mdp.terminations",
        SimpleNamespace(
            motion_clip_end=lambda env, command_name: None,
        ),
    )

    transfers = []

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            transfers.append(self.values.shape)
            return self

        def numpy(self):
            return self.values

        def new_zeros(self, size):
            return np.zeros(size, dtype=self.values.dtype)

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            stack=lambda tensors: Tensor(np.stack([t.values for t in tensors]))
        ),
    )
    motion = SimpleNamespace(
        metrics={
            "error_anchor_pos": Tensor([0.25, 0.5]),
            "error_object_pos": Tensor([1.25, 1.5]),
        },
        time_steps=Tensor([1, 1]),
        _update_metrics=lambda: None,
    )
    terms = {"object_pos": Tensor([True, False]), "clip_end": Tensor([True, True])}
    env = SimpleNamespace(
        num_envs=2,
        command_manager=SimpleNamespace(get_term=lambda _: motion),
        termination_manager=SimpleNamespace(
            terminated=terms["object_pos"],
            time_outs=terms["clip_end"],
            get_term=terms.__getitem__,
            active_terms=list(terms),
        ),
    )
    cfg = SimpleNamespace(
        rewards=SimpleNamespace(), terminations=SimpleNamespace(), recorders=None
    )
    collector = ClipEvaluation(["failed", "succeeded"], [2, 2])
    attach_evaluator(cfg, collector)
    assert cfg.recorders is None
    hook = cfg.rewards.ef_tracking
    assert hook.weight != 0  # The SDK skips zero-weight terms.
    assert np.array_equal(hook.func(env), np.zeros(2))
    assert transfers == [(2, 2), (4, 2), (2,)]
    report = collector.report()
    assert [row["status"] for row in report["clips"]] == ["failed", "success"]
    assert report["clips"][0]["metrics"] == {
        "error_anchor_pos": 0.25,
        "error_object_pos": 1.25,
    }
    assert report["clips"][0]["termination_terms"] == ["object_pos", "clip_end"]
    assert report["clips"][1]["termination_terms"] == ["clip_end"]
    # A later auto-reset episode cannot replace the terminal-frame statistics.
    motion.metrics["error_anchor_pos"] = Tensor([99.0, 99.0])
    motion.time_steps = Tensor([0, 0])
    hook.func(env)
    assert collector.report() == report


@pytest.mark.parametrize("resume", [True, False])
def test_checkpoint_maps_device_and_restores_training_state(resume, monkeypatch):
    # Torch load_state_dict replaces child group dictionaries; the wrapper's
    # old flattened list still points at the pre-load groups.
    children = [SimpleNamespace(param_groups=[{"lr": 2e-4}]) for _ in range(2)]
    optimizer = SimpleNamespace(optimizers=children, param_groups=[{"lr": 1e-4}])
    algorithm = SimpleNamespace(optimizer=optimizer, learning_rate=1e-4)
    runner = SimpleNamespace(alg=algorithm, current_learning_iteration=0)
    restored = []
    checkpoint = {
        "model_state_dict": {"weights": "policy"},
        "optimizer_state_dict": {"optimizers": [{}, {}]},
        "iter": 99,
    }

    def read(path, *, weights_only, map_location):
        assert path == "model_99.pt"
        assert weights_only is False
        assert map_location == "cpu"
        return checkpoint

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=read))

    # This test exercises load ordering and the SDK-free routing boundary. The
    # real Torch/Weave buffer contract is covered below in the optional SDK test.
    def validate(opt, state):
        assert opt is optimizer
        assert state is checkpoint["optimizer_state_dict"]
        assert not restored

    monkeypatch.setattr(_weave_worker, "validate_resume_optimizer", validate)
    algorithm.policy = SimpleNamespace(load_state_dict=restored.append)
    optimizer.load_state_dict = restored.append
    monkeypatch.setattr(
        _weave_worker,
        "validate_policy_state",
        lambda policy, state: None,
    )
    load_checkpoint(runner, "model_99.pt", resume=resume)
    if not resume:
        assert restored == [checkpoint["model_state_dict"]]
        assert runner.current_learning_iteration == 99
        assert algorithm.learning_rate == 1e-4
        assert optimizer.param_groups == [{"lr": 1e-4}]
        return
    assert runner.current_learning_iteration == 100
    assert algorithm.learning_rate == 2e-4
    assert restored == [
        checkpoint["model_state_dict"],
        checkpoint["optimizer_state_dict"],
    ]
    # Invalid resume metadata must fail before changing policy or optimizer state.
    restored.clear()
    checkpoint["iter"] = -1
    with pytest.raises(ValueError, match="saved iteration"):
        load_checkpoint(runner, "model_99.pt", resume=True)
    assert restored == []
    for group in optimizer.param_groups:
        group["lr"] = 3e-4
    assert all(child.param_groups[0]["lr"] == 3e-4 for child in children)
    children[0].param_groups[0]["lr"] = 0.0
    with pytest.raises(ValueError, match="learning rate"):
        synchronize_resume_optimizer(algorithm)


def test_upstream_checkpoint_rejects_damage_and_preserves_update(tmp_path):
    torch = pytest.importorskip("torch")
    root = os.environ.get("WEAVE_ROOT")
    if not root or not hasattr(torch.optim, "Muon"):
        pytest.skip("Set WEAVE_ROOT and use Torch >= 2.10 for real Muon/AdamW resume")
    weave.check_checkout(Path(root), weave.WEAVE_REVISION)
    # Import the pinned optimizer without loading IsaacLab task registration.
    spec = importlib.util.spec_from_file_location(
        "weave_optimizer_contract",
        Path(root) / "source/g1_hoi_learning/g1_hoi_learning/algorithms/optimizers.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def runner():
        policy = torch.nn.Linear(3, 4)
        optimizer = module.MuonAdamWWrapper([policy], lr=1e-4)
        return SimpleNamespace(
            alg=SimpleNamespace(policy=policy, optimizer=optimizer, learning_rate=1e-4),
            current_learning_iteration=0,
        )

    def update(run):
        run.alg.optimizer.zero_grad()
        sum(p.square().sum() for p in run.alg.policy.parameters()).backward()
        run.alg.optimizer.step()

    original = runner()
    update(original)
    for group in original.alg.optimizer.param_groups:
        group["lr"] = 2e-4
    base = copy.deepcopy(
        {
            "model_state_dict": original.alg.policy.state_dict(),
            "optimizer_state_dict": original.alg.optimizer.state_dict(),
            "iter": 99,
        }
    )
    path = tmp_path / "model_99.pt"
    torch.save(base, path)
    original.current_learning_iteration = 99
    report = _weave_worker.training_result(original, tmp_path, 80, 20)
    assert report["last_iteration"] == 99
    assert report["policy_parameters"] == 16
    assert report["policy_state_tensor_bytes"] == 64
    with pytest.raises(RuntimeError, match="requested iterations"):
        _weave_worker.training_result(original, tmp_path, 80, 21)
    restored = runner()
    load_checkpoint(restored, path, resume=True)
    assert restored.current_learning_iteration == 100
    assert restored.alg.learning_rate == 2e-4
    for child in restored.alg.optimizer.optimizers:
        assert any(
            child.param_groups[0] is g for g in restored.alg.optimizer.param_groups
        )
    update(original)
    update(restored)
    for expected, actual in zip(
        original.alg.policy.parameters(), restored.alg.policy.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    for damage in (
        "missing_child",
        "empty_muon",
        "empty_adamw",
        "shape",
        "variance",
        "lr",
        "options",
        "nan_weight",
        "weight_dtype",
    ):
        saved = copy.deepcopy(base)
        children = saved["optimizer_state_dict"]["optimizers"]
        if damage == "missing_child":
            children.pop()
        elif damage.startswith("empty_"):
            children[int(damage == "empty_adamw")]["state"].clear()
        elif damage == "shape":
            children[0]["state"][0]["momentum_buffer"] = torch.zeros(1)
        elif damage == "variance":
            children[1]["state"][0]["exp_avg_sq"].fill_(-1)
        elif damage == "lr":
            children[1]["param_groups"][0]["lr"] = 3e-4
        elif damage == "options":
            children[0]["param_groups"][0]["momentum"] = 0.5
        elif damage == "nan_weight":
            saved["model_state_dict"]["weight"].fill_(float("nan"))
        else:
            saved["model_state_dict"]["weight"] = saved["model_state_dict"][
                "weight"
            ].double()
        torch.save(saved, path)
        for resume in (False, True) if "weight" in damage else (True,):
            target = runner()
            before = copy.deepcopy(target.alg.policy.state_dict())
            with pytest.raises(ValueError, match="Weave"):
                load_checkpoint(target, path, resume=resume)
            assert target.current_learning_iteration == 0
            assert all(not child.state for child in target.alg.optimizer.optimizers)
            for key, value in target.alg.policy.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("error", [KeyboardInterrupt, RuntimeError])
def test_interrupted_evaluation_keeps_first_episode_report(
    tmp_path, error, monkeypatch, capsys
):
    collector = ClipEvaluation(["done", "pending"], [2, 4])

    def step(_):
        collector.update(
            {"error": np.array([1.0, 2.0])},
            [1, 1],
            np.array([False, False]),
            np.array([True, False]),
            np.array([True, False]),
            {"clip_end": [True, False]},
        )
        raise error("interrupted step")

    inputs = {"motions": [{"sha256": "reference"}], "checkpoint": {"sha256": "policy"}}
    env = SimpleNamespace(get_observations=lambda: None, step=step)
    app = SimpleNamespace(is_running=lambda: True)
    with pytest.raises(error, match="interrupted step"):
        evaluate_policy(lambda obs: obs, env, app, collector, tmp_path, inputs)
    report = json.loads((tmp_path / "result.json").read_text())
    assert report["status"] == "incomplete"
    assert report["inputs"] == inputs
    assert [clip["status"] for clip in report["clips"]] == ["success", "incomplete"]
    assert report["clips"][0]["metrics"] == {"error": 1.0}

    def failed_write(*_):
        raise OSError("report disk full")

    monkeypatch.setattr("embodiedforge._weave_worker.write_json", failed_write)
    with pytest.raises(error, match="interrupted step"):
        evaluate_policy(lambda obs: obs, env, app, collector, tmp_path, inputs)
    assert (
        "Failed to save evaluation report: report disk full" in capsys.readouterr().err
    )
    assert json.loads((tmp_path / "result.json").read_text()) == report


def test_evaluation_counts_terminal_failure_before_reset():
    evaluation = ClipEvaluation(["fails_at_end", "success", "timeout"], [2, 3, 9])
    evaluation.update(
        {"error_object_pos": np.array([2.0, 1.0, 4.0])},
        [1, 1, 1],
        np.array([True, False, False]),
        np.array([True, False, True]),
        np.array([True, False, False]),
        {
            "object_pos": [True, False, False],
            "clip_end": [True, False, False],
            "time_out": [False, False, True],
        },
    )
    # Auto-reset states must neither overwrite first-episode status nor add errors.
    evaluation.update(
        {"error_object_pos": np.array([np.nan, 3.0, np.nan])},
        [0, 2, 0],
        np.array([False, False, False]),
        np.array([False, True, False]),
        np.array([False, True, False]),
        {"clip_end": [False, True, False]},
    )
    report = evaluation.report()
    assert report["status"] == "complete"
    assert [row["status"] for row in report["clips"]] == [
        "failed",
        "success",
        "truncated",
    ]
    assert report["success_rate"] == pytest.approx(1 / 3)
    assert report["metrics_success"] == {"error_object_pos": 2.0}
    assert report["clips"][0]["termination_terms"] == ["object_pos", "clip_end"]
    json.dumps(report, allow_nan=False)


def test_incomplete_and_all_failed_reports_do_not_claim_success():
    evaluation = ClipEvaluation(["clip"], [2])
    assert evaluation.report()["status"] == "incomplete"
    before = evaluation.report()
    with pytest.raises(ValueError, match="Invalid evaluation metric"):
        evaluation.update(
            {"first": np.array([1.0]), "invalid": np.array([np.nan])},
            [0],
            np.array([False]),
            np.array([False]),
            np.array([False]),
            {},
        )
    assert evaluation.report() == before
    evaluation.update(
        {"error": np.array([1.0])},
        [0],
        np.array([True]),
        np.array([False]),
        np.array([False]),
        {"failure": [True]},
    )
    assert evaluation.report()["metrics_success"] == {"error": None}
    assert evaluation.report()["success_rate"] == 0


@pytest.mark.parametrize(
    "worker_error", [None, subprocess.CalledProcessError, KeyboardInterrupt]
)
def test_launcher_isolates_sdk_and_preserves_run_status(
    tmp_path, monkeypatch, worker_error, capsys
):
    python = tmp_path / "sdk/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    output = tmp_path / "run"
    source = tmp_path / "checkpoint.pt"
    source.write_bytes(b"original checkpoint")
    monkeypatch.setattr(weave, "check_checkout", lambda *_: None)
    monkeypatch.setattr(weave, "check_assets", lambda *_: None)
    monkeypatch.setattr(weave, "prepare_inputs", lambda *_: {"objects": ["box"]})

    def worker(command, *, cwd, env, log_path):
        request = json.loads((output / "request.json").read_text())
        copied = Path(request["checkpoint"]["path"])
        assert copied != source and copied.read_bytes() == source.read_bytes()
        assert request["checkpoint"]["sha256"] == weave.sha256(copied)
        before = copied.read_bytes()
        source.write_bytes(b"source changed after snapshot")
        assert copied.read_bytes() == before
        assert request["headless"] is True
        assert request["iterations"] == 200
        assert request["python"] == str(python)  # Preserve venv symlink path.
        assert command[-2] == str(python)
        assert env["CONDA_PREFIX"] == str(python.parent.parent)
        assert "inherited-secret-path" not in env["PYTHONPATH"]
        assert env["PYTHONPATH"].split(os.pathsep) == [
            str(output / "implementation"),
            str(tmp_path / "source/g1_hoi_learning"),
        ]
        package = output / "implementation" / "embodiedforge"
        assert not package.is_symlink()
        validate_snapshot(package, request["implementation"])
        if worker_error is None:
            imported = subprocess.check_output(
                [
                    sys.executable,
                    "-c",
                    "from embodiedforge import _weave_worker; print(_weave_worker.__file__)",
                ],
                cwd=cwd,
                env=env,
                text=True,
            ).strip()
            assert Path(imported) == package / "_weave_worker.py"
        assert env["CUDA_VISIBLE_DEVICES"] == "4"
        assert (
            not {
                "WORLD_SIZE",
                "LOCAL_WORLD_SIZE",
                "RANK",
                "LOCAL_RANK",
                "MASTER_ADDR",
                "MASTER_PORT",
            }
            & env.keys()
        )
        assert cwd == output and log_path == output / "worker.log"
        (output / "model_9.pt").write_bytes(b"earlier publication")
        (output / "model_100.pt").write_bytes(b"latest publication")
        (output / "model_101.pt.incomplete.tmp").write_bytes(b"unfinished")
        (output / "model_backup.pt").write_bytes(b"unrelated")
        if worker_error is KeyboardInterrupt:
            raise KeyboardInterrupt()
        if worker_error:
            raise subprocess.CalledProcessError(1, command)
        (output / "result.json").write_text(
            json.dumps(
                {"status": "complete", "checkpoint": str(output / "model_100.pt")}
            )
        )

    monkeypatch.setenv("PYTHONPATH", "inherited-secret-path")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    for key, value in {
        "WORLD_SIZE": "1",
        "LOCAL_WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(weave, "run_process", worker)
    args = weave.parser().parse_args(
        [
            "train",
            "--python",
            str(python),
            "--weave-root",
            str(tmp_path),
            "--isaaclab-root",
            str(tmp_path),
            "--motions",
            "reference.npz",
            "--output",
            str(output),
            "--iterations",
            "200",
            "--checkpoint",
            str(source),
            *(["--headless"] if worker_error else []),
        ]
    )
    if worker_error:
        with pytest.raises(worker_error):
            weave.launch(args)
    else:
        assert weave.launch(args) == output
    status = json.loads((output / "run.json").read_text())
    assert status["status"] == (
        "interrupted"
        if worker_error is KeyboardInterrupt
        else "failed"
        if worker_error
        else "complete"
    )
    assert status["latest_checkpoint"] == str(output / "model_100.pt")
    assert status["last_saved_iteration"] == 100
    assert status["checkpoint_validation"] == (
        "not_performed" if worker_error else "policy_and_optimizer"
    )
    with pytest.raises(FileExistsError):
        weave.launch(args)

    if worker_error is None:
        args.output = tmp_path / "preparing-interrupt"
        previous_handler = signal.getsignal(signal.SIGTERM)

        def interrupt_preparation(*unused):
            status = json.loads((args.output / "run.json").read_text())
            assert status["status"] == "preparing"
            assert (args.output / "request.json").is_file()
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        with monkeypatch.context() as patch:
            patch.setattr(weave, "prepare_inputs", interrupt_preparation)
            with pytest.raises(KeyboardInterrupt, match="signal"):
                weave.launch(args)
        status = json.loads((args.output / "run.json").read_text())
        assert status["status"] == "interrupted"
        assert status["latest_checkpoint"] is None
        assert signal.getsignal(signal.SIGTERM) is previous_handler

        original_copy = weave.shutil.copyfile
        for mutation in ("source", "snapshot"):
            args.output = tmp_path / f"changed-{mutation}"

            def changing_copy(src, dst, mutation=mutation):
                original_copy(src, dst)
                (src if mutation == "source" else dst).write_bytes(
                    f"changed during {mutation} copy".encode()
                )

            def unexpected_worker(*args, **kwargs):
                pytest.fail("Changed input must be rejected before starting the SDK")

            with monkeypatch.context() as patch:
                patch.setattr(weave.shutil, "copyfile", changing_copy)
                patch.setattr(weave, "run_process", unexpected_worker)
                with pytest.raises(ValueError, match="changed during snapshot"):
                    weave.launch(args)
            status = json.loads((args.output / "run.json").read_text())
            assert status["status"] == "failed"
            assert status["latest_checkpoint"] is None
            assert not (args.output / "result.json").exists()

    # Final status I/O must not replace a worker failure or user interruption.
    original_write = weave.write_json

    def failed_final_status(path, value):
        if path.name == "run.json" and value["status"] not in ("preparing", "running"):
            raise OSError("status disk full")
        original_write(path, value)

    monkeypatch.setattr(weave, "write_json", failed_final_status)
    output = tmp_path / "final-status-failure"
    args.output = output
    with pytest.raises(worker_error or OSError):
        weave.launch(args)
    assert json.loads((output / "run.json").read_text())["status"] == "running"
    assert (output / "model_100.pt").read_bytes() == b"latest publication"
    if worker_error:
        assert (
            "Failed to finalize Weave run: status disk full" in capsys.readouterr().err
        )

    # Reject a distributed launch before touching output or preparing data.
    args.output = tmp_path / "distributed"
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="one SDK worker"):
        weave.launch(args)
    assert not args.output.exists()


def test_asset_checks_follow_task_dependencies(tmp_path):
    root = tmp_path / "source/g1_hoi_learning/g1_hoi_learning"
    robot = root / "assets/unitree_g1"
    (robot / "meshes").mkdir(parents=True)
    mesh = robot / "meshes/leg.STL"
    mesh.write_bytes(b"mesh")
    (robot / "g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf").write_text(
        '<robot><link><visual><geometry><mesh filename="meshes/leg.STL"/>'
        "</geometry></visual></link></robot>"
    )
    (root / "objects/geometry").mkdir(parents=True)
    np.save(root / "objects/geometry/bps_128.npy", np.zeros((128, 3)))
    objects = root / "objects/smalltable"
    objects.mkdir()
    np.save(objects / "surface.npy", np.zeros((8, 3)))
    sdf = objects / "sdf_128.npz"
    np.savez(sdf, sdf=np.zeros((2, 2, 2)))
    usd = objects / "smalltable.usd"
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:123\n"
    usd.write_text(pointer)
    with pytest.raises(ValueError, match="Unresolved Git LFS"):
        weave.check_assets(tmp_path, {"smalltable"})
    usd.write_bytes(b"USD fixture")
    # This task uses a generated ground plane and the converted object USD.
    (root / "assets/environments").mkdir()
    (root / "assets/environments/default_environment.usd").write_text(pointer)
    (objects / "smalltable.obj").write_text(pointer)
    weave.check_assets(tmp_path, {"smalltable"})
    for required in (mesh, sdf):
        content = required.read_bytes()
        required.unlink()
        with pytest.raises(ValueError, match="Missing Weave asset") as error:
            weave.check_assets(tmp_path, {"smalltable"})
        assert str(required) in str(error.value)
        required.write_bytes(content)
    with pytest.raises(ValueError, match="Unknown Weave objects"):
        weave.check_assets(tmp_path, {"smalltabel"})
