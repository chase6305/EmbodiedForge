"""Learners and shared math can be used without a simulator or core trainer."""

import os
import subprocess
import sys
import textwrap

import pytest


def test_nested_checkpoint_tensor_count_preserves_metadata_semantics():
    torch = pytest.importorskip("torch")
    from embodiedforge.learning.checkpoints import finite_tensors

    value = {"weights": [torch.ones(2), (torch.zeros(1),)], "metadata": float("nan")}
    assert finite_tensors(value) == 2
    with pytest.raises(ValueError, match="Non-finite checkpoint scalar"):
        finite_tensors(value, check_scalars=True)
    value["metadata"] = {"nested": [(0.25,)]}
    assert finite_tensors(value, check_scalars=True) == 2
    value["weights"].append({"bad": torch.tensor(float("inf"))})
    with pytest.raises(ValueError, match="Non-finite checkpoint tensor"):
        finite_tensors(value)


def test_shared_advantage_and_checkpoint_imports_do_not_load_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from embodiedforge.learning import advantages, checkpoints; assert 'torch' not in sys.modules; assert 'embodiedforge.training' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_robot_learners_do_not_import_simulators_or_the_core_trainer():
    pytest.importorskip("torch")
    script = textwrap.dedent("""
        import importlib.abc
        import sys

        forbidden = {'mujoco', 'mjbatch', 'mujoco_menagerie', 'embodiedforge.training',
                     'embodiedforge.locomotion.go1', 'embodiedforge.locomotion.h1_native'}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in forbidden:
                    raise RuntimeError('Forbidden learner dependency: ' + fullname)
        sys.meta_path.insert(0, Block())
        import torch
        from embodiedforge.locomotion import go1_ppo, h1_ppo
        go1 = go1_ppo.ActorCritic()
        h1 = h1_ppo.Policy()
        assert go1.action_mean(torch.zeros(2, 50)).shape == (2, 12)
        assert h1.distribution(torch.zeros(2, 69)).mean.shape == (2, 19)
        assert not forbidden & sys.modules.keys()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_live_viewer_and_workflow_helpers_do_not_import_recipe_cli():
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        forbidden = {'embodiedforge.recipes', 'torch', 'mujoco', 'mjbatch', 'warp'}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in forbidden:
                    raise RuntimeError('Forbidden workflow dependency: ' + fullname)
        sys.meta_path.insert(0, Block())
        import embodiedforge.go1_live
        from embodiedforge.workflows import recipe_catalog, recipe_artifacts, recipe_progress, recipe_validation, go1_settings
        assert recipe_catalog.RECIPES['go1-joystick']['algorithm'] == 'PPO'
        assert go1_settings.resolve_go1_semantics(None) == 'transition-v2'
        assert not forbidden & sys.modules.keys()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_publication_and_weave_import_need_no_robot_domain_or_sdk(tmp_path):
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from pathlib import Path

        forbidden = {'embodiedforge.locomotion', 'torch', 'numpy', 'mujoco',
                     'mjlab', 'isaaclab', 'rsl_rl', 'g1_hoi_learning', 'warp'}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.')
                       for name in forbidden):
                    raise RuntimeError('Forbidden checkpoint dependency: ' + fullname)
        blocker = Block()
        sys.meta_path.insert(0, blocker)
        from embodiedforge.learning.checkpoints import atomic_save
        from embodiedforge import _weave_worker
        assert _weave_worker.atomic_save is atomic_save
        target = Path(sys.argv[1]) / 'model.pt'
        infos = object()
        def save(filename, metadata):
            assert metadata is infos
            Path(filename).write_bytes(b'complete checkpoint')
        atomic_save(save, target, infos)
        assert target.read_bytes() == b'complete checkpoint'
        assert not forbidden & sys.modules.keys()

        sys.meta_path.remove(blocker)
        from embodiedforge.locomotion.microduck.checkpoint import atomic_save as legacy
        assert legacy is atomic_save
        assert not {'torch', 'mujoco', 'mjlab', 'rsl_rl'} & sys.modules.keys()
    """)
    result = subprocess.run(
        [sys.executable, "-S", "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
