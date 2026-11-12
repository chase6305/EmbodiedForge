"""Core policy loading does not compose the simulation execution path."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest


def test_core_checkpoint_loaders_work_with_simulation_imports_blocked(tmp_path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("Requires optional torch")
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from dataclasses import asdict
        from pathlib import Path

        forbidden = {
            'embodiedforge.env', 'embodiedforge.backends',
            'embodiedforge.sensors', 'embodiedforge.tasks',
            'mujoco', 'mjbatch', 'newton', 'warp',
        }
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + '.')
                       for name in forbidden):
                    raise RuntimeError('Forbidden policy dependency: ' + fullname)
        sys.meta_path.insert(0, Block())

        import torch
        from embodiedforge import training
        from embodiedforge.core import Config

        torch.set_num_threads(1)
        policy = training.ActorCritic()
        optimizer = torch.optim.Adam(policy.parameters())
        # Populate every Adam slot without constructing an environment or rollout.
        sum(parameter.sum() for parameter in policy.parameters()).backward()
        optimizer.step()
        path = Path(sys.argv[1]) / 'checkpoint.pt'
        torch.save({
            'schema_version': 2,
            'updates': 1,
            'model': policy.state_dict(),
            'model_spec': policy.model_spec,
            'optimizer': optimizer.state_dict(),
            'env': Config(num_envs=2).to_dict(),
            'env_spec': {
                'schema_version': 1, 'done_handling': 'freeze',
                'task': 'reach', 'scene': 'point_mass', 'device': 'cpu',
                'num_envs': 2, 'control_dt': 0.02,
                'proprio_shape': [6], 'action_shape': [2],
                'action_units': 'newtons_xy', 'action_range': [-1.0, 1.0],
            },
            'ppo': asdict(training.PPOConfig()),
        }, path)
        bundle = training.load_checkpoint(path)
        loaded_policy = training.load_policy(path)
        resumed = training.load_training_checkpoint(path)
        assert isinstance(bundle.policy, training.ActorCritic)
        assert isinstance(loaded_policy, training.ActorCritic)
        assert resumed.bundle.sha256 == bundle.sha256
        assert resumed.updates == 1
        assert training.ActorCritic.__module__ == 'embodiedforge.training'
        observation = {'proprio': torch.zeros(2, 6).numpy()}
        expected = policy.act(observation)
        for loaded in (bundle.policy, loaded_policy, resumed.bundle.policy):
            assert (loaded.act(observation) == expected).all()
        assert not any(name == prefix or name.startswith(prefix + '.')
                       for name in sys.modules for prefix in forbidden)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
