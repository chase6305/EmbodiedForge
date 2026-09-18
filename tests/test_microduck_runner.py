"""CPU regression checks against the installed training SDK; no simulation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


@unittest.skipUnless(importlib.util.find_spec("mjlab"), "requires Microduck SDK")
class MicroduckRunnerTests(unittest.TestCase):
    def setUp(self):
        import torch

        from embodiedforge.locomotion.microduck.tasks import MicroduckOnPolicyRunner

        self.torch = torch
        self.runner = MicroduckOnPolicyRunner.__new__(MicroduckOnPolicyRunner)
        self.runner.env = SimpleNamespace(
            unwrapped=SimpleNamespace(common_step_counter=120)
        )
        self.runner.cfg = {"upload_model": False}
        self.runner.current_learning_iteration = 5

    def test_save_preserves_training_state_without_export(self):
        payload = {
            "actor_state_dict": {"weight": self.torch.ones(2)},
            "critic_state_dict": {"weight": self.torch.zeros(2)},
            "optimizer_state_dict": {"param_groups": [{"lr": 0.001}]},
        }
        self.runner.alg = Mock()
        self.runner.alg.save.side_effect = lambda: dict(payload)
        self.runner.alg.load.return_value = True
        self.runner.export_policy_to_onnx = Mock(
            side_effect=AssertionError("checkpoint save must not export ONNX")
        )
        for logger in ("tensorboard", "wandb"):
            with self.subTest(logger=logger), tempfile.TemporaryDirectory() as folder:
                self.runner.logger = SimpleNamespace(logger_type=logger)
                path = Path(folder) / "model_5.pt"
                self.runner.save(str(path), {"label": "training"})
                saved = self.torch.load(path, weights_only=True)
                self.assertEqual(saved["iter"], 5)
                self.assertEqual(
                    saved["infos"]["env_state"]["common_step_counter"], 120
                )
                self.assertEqual(
                    saved["optimizer_state_dict"], payload["optimizer_state_dict"]
                )
                for key in ("actor_state_dict", "critic_state_dict"):
                    self.assertTrue(
                        self.torch.equal(saved[key]["weight"], payload[key]["weight"])
                    )
                self.assertEqual(list(Path(folder).iterdir()), [path])
                self.runner.export_policy_to_onnx.assert_not_called()
                self.runner.current_learning_iteration = 0
                self.runner.env.unwrapped.common_step_counter = 0
                infos = self.runner.load(str(path), map_location="cpu")
                self.assertEqual(infos["label"], "training")
                self.assertEqual(self.runner.current_learning_iteration, 5)
                self.assertEqual(self.runner.env.unwrapped.common_step_counter, 120)

    def test_export_metadata_follows_action_order_and_preserves_precision(self):
        import numpy as np
        import onnx
        from mjlab.envs.mdp.actions import JointPositionAction
        from mjlab.rl import exporter_utils

        from embodiedforge.locomotion.microduck.export import policy_metadata

        # Task registration must leave the SDK's metadata function intact.
        self.assertEqual(
            exporter_utils.get_base_metadata.__module__, exporter_utils.__name__
        )
        action = JointPositionAction.__new__(JointPositionAction)
        action._target_names = ["right", "left"]
        action._target_ids = self.torch.tensor([2, 0])
        defaults = self.torch.tensor([[0.123456789, 9.0, -0.0873]])
        robot = SimpleNamespace(
            joint_names=["left", "passive_link", "right"],
            data=SimpleNamespace(default_joint_pos=defaults),
            spec=SimpleNamespace(
                actuators=[
                    SimpleNamespace(target="robot/left", id=0),
                    SimpleNamespace(target="robot/right", id=1),
                ]
            ),
        )
        env = SimpleNamespace(
            scene={"robot": robot},
            action_manager=SimpleNamespace(get_term=lambda name: action),
            sim=SimpleNamespace(
                mj_model=SimpleNamespace(
                    actuator_gainprm=np.array([[10.0], [30.0]]),
                    actuator_biasprm=np.array([[0, 0, -0.1], [0, 0, -0.3]]),
                )
            ),
            command_manager=SimpleNamespace(active_terms=["twist"]),
            observation_manager=SimpleNamespace(active_terms={"actor": ["joint_pos"]}),
        )
        for scale in (1.0, self.torch.tensor([[0.000012345, 0.23456789]])):
            action._scale = scale
            metadata = policy_metadata(env, "checkpoint.pt")
            self.assertEqual(metadata["joint_names"], ["right", "left"])
            self.assertEqual(metadata["joint_stiffness"], [30.0, 10.0])
            self.assertEqual(metadata["joint_damping"], [0.3, 0.1])
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.onnx"
                onnx.save(
                    onnx.helper.make_model(
                        onnx.helper.make_graph([], "metadata", [], [])
                    ),
                    path,
                )
                exporter_utils.attach_metadata_to_onnx(str(path), metadata)
                saved = {item.key: item.value for item in onnx.load(path).metadata_props}
            self.assertEqual(
                list(map(float, saved["default_joint_pos"].split(","))),
                defaults[0, [2, 0]].tolist(),
            )
            expected_scale = [scale] if isinstance(scale, float) else scale[0].tolist()
            self.assertEqual(
                list(map(float, saved["action_scale"].split(","))), expected_scale
            )

    def test_ppo_preserves_finite_targets_and_rejects_nonfinite_rollouts(self):
        from copy import deepcopy

        from rsl_rl.algorithms.ppo import PPO
        from rsl_rl.utils import resolve_callable

        from embodiedforge.locomotion.microduck.tasks import (
            MicroduckPPO,
            MicroduckRlCfg,
        )

        self.assertIs(
            resolve_callable(MicroduckRlCfg.algorithm.class_name), MicroduckPPO
        )
        self.assertEqual(PPO.compute_returns.__module__, "rsl_rl.algorithms.ppo")
        algorithm = MicroduckPPO.__new__(MicroduckPPO)
        algorithm.gamma, algorithm.lam = 0.99, 0.95
        algorithm.normalize_advantage_per_mini_batch = False
        algorithm.critic = lambda obs: self.torch.zeros(1, 1)
        algorithm.storage = SimpleNamespace(
            num_transitions_per_env=2,
            rewards=self.torch.tensor([[[0.5]], [[1.0]]]),
            values=self.torch.zeros(2, 1, 1),
            dones=self.torch.zeros(2, 1, 1),
            returns=self.torch.zeros(2, 1, 1),
        )
        reference = deepcopy(algorithm)
        PPO.compute_returns(reference, None)
        algorithm.compute_returns(None)
        for name in ("advantages", "returns"):
            self.assertTrue(
                self.torch.equal(
                    getattr(algorithm.storage, name), getattr(reference.storage, name)
                )
            )
        for value in (float("nan"), float("inf")):
            algorithm.storage.rewards[0] = value
            with self.assertRaisesRegex(RuntimeError, "Nonfinite PPO"):
                algorithm.compute_returns(None)

    def test_sdk_reward_handling_needs_no_task_patch(self):
        from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

        self.assertEqual(
            RewardManager.compute.__module__, "mjlab.managers.reward_manager"
        )
        values = self.torch.tensor([2.0, float("nan"), float("inf"), -float("inf")])
        manager = RewardManager(
            {"example": RewardTermCfg(func=lambda env: values, weight=2.0)},
            SimpleNamespace(num_envs=4, device="cpu", max_episode_length_s=2.0),
        )
        expected = self.torch.tensor([0.08, 0.0, 0.0, 0.0])
        for _ in range(2):
            self.torch.testing.assert_close(manager.compute(0.02), expected)
        self.torch.testing.assert_close(manager._episode_sums["example"], expected * 2)
        self.torch.testing.assert_close(manager._step_reward[:, 0], expected / 0.02)
        metrics = manager.reset()
        self.torch.testing.assert_close(
            metrics["Episode_Reward/example"], expected.mean()
        )
        self.assertEqual(manager._episode_sums["example"].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
