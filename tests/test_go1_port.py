"""Go1 contract checks, runnable in the isolated SDK with stdlib unittest.

Set EF_MJBATCH_REFERENCE to a pinned mjbatch checkout to also compare the port
against the original environment, rollout, GAE and one complete PPO update.
"""

import importlib
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np


class Go1PortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch

            from embodiedforge.locomotion import go1, go1_ppo
        except ImportError as exc:
            raise unittest.SkipTest(f"Optional Go1 SDK: {exc}") from exc
        cls.task, cls.ppo, cls.torch = go1, go1_ppo, torch
        torch.set_num_threads(2)

    def reference(self):
        root = os.environ.get("EF_MJBATCH_REFERENCE")
        if not root:
            self.skipTest("EF_MJBATCH_REFERENCE is not configured")
        sys.path.insert(0, str(Path(root) / "examples"))
        try:
            owner = importlib.import_module("go1_joystick")
        finally:
            sys.path.pop(0)
        from mjbatch import Batch

        owner.Batch = lambda model, count: Batch(model, count, num_threads=2)
        owner.DEVICE = owner.ACTOR = "cpu"
        return owner

    def test_instances_keep_separate_episode_limits_and_reject_invalid_actions(self):
        a = self.task.Go1(2, num_threads=1, episode_steps=1)
        b = self.task.Go1(3, num_threads=1, episode_steps=3)
        before = a.qpos.copy()
        for action in (
            np.zeros((1, 12)),
            np.full((2, 12), np.nan),
            np.ones((2, 12), dtype=complex),
            np.full((2, 12), "invalid"),
            np.full((2, 12), 1e100),
        ):
            with self.assertRaises(ValueError):
                a.step(action)
            np.testing.assert_array_equal(a.qpos, before)
        self.assertTrue(a.step(np.zeros((2, 12)))[1].all())
        self.assertFalse(b.step(np.zeros((3, 12)))[1].any())
        self.assertEqual(b.obs().shape, (3, 50))

    def snapshot(self, env):
        return {
            name: getattr(env, name).copy()
            for name in (
                "qpos",
                "qvel",
                "ctrl",
                "friction",
                "steps",
                "clock",
                "action",
                "down",
                "command",
                "until",
            )
        }

    def test_unsorted_partial_reset_preserves_order_and_other_environments(self):
        a = self.task.Go1(4, seed=6, num_threads=1)
        b = self.task.Go1(4, seed=6, num_threads=1)
        action = np.full((4, 12), 0.2, dtype=np.float32)
        a.step(action)
        b.step(action)
        before = self.snapshot(a)
        a.reset([3, 1])
        b.reset([1, 3])
        after, ordered = self.snapshot(a), self.snapshot(b)
        for name in before:
            np.testing.assert_array_equal(after[name][[0, 2]], before[name][[0, 2]])
            np.testing.assert_array_equal(after[name][[3, 1]], ordered[name][[1, 3]])
        np.testing.assert_array_equal(a.obs()[[3, 1]], b.obs()[[1, 3]])

    def test_empty_and_invalid_reset_do_not_modify_state_or_rng(self):
        env = self.task.Go1(4, seed=6, num_threads=1)
        env.step(np.ones((4, 12), dtype=np.float32) * 0.1)
        before, rng = self.snapshot(env), deepcopy(env.rng.bit_generator.state)
        env.reset([])
        for ids in (
            [1, 1],
            [-1],
            [4],
            [True],
            [1.0],
            [[1]],
            1,
            np.array([2**64 - 1], dtype=np.uint64),
        ):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                env.reset(ids)
        for keep in (-0.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                env.resample([0], keep=keep)
        for name, value in self.snapshot(env).items():
            np.testing.assert_array_equal(value, before[name])
        self.assertEqual(env.rng.bit_generator.state, rng)
        env.reset()
        self.assertTrue((env.steps == 0).all())

    def test_invalid_ppo_batch_is_rejected_before_optimizer_changes(self):
        torch, ppo = self.torch, self.ppo
        model = ppo.ActorCritic()
        opt = torch.optim.Adam(model.parameters(), lr=ppo.LR)
        before = deepcopy(model.state_dict())
        for kind in ("small", "uneven", "shape", "nan", "dtype", "advantage", "loss"):
            with self.subTest(kind=kind):
                n = 1 if kind == "small" else 5 if kind == "uneven" else 4
                batch = {
                    "obs": torch.zeros(2, n, 50),
                    "act": torch.zeros(2, n, 12),
                    "logp": torch.zeros(2, n),
                }
                adv, ret = torch.zeros(2, n), torch.ones(2, n)
                if kind == "shape":
                    batch["act"] = torch.zeros(2, n, 11)
                elif kind == "nan":
                    ret[0, 0] = float("nan")
                elif kind == "dtype":
                    batch["obs"] = batch["obs"].double()
                elif kind == "advantage":
                    adv = torch.zeros(n)
                elif kind == "loss":
                    ret.fill_(1e30)  # Finite input whose squared loss overflows.
                with self.assertRaises(ValueError):
                    ppo.update(model, opt, batch, adv, ret)
        self.assertFalse(opt.state)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_gae_rejects_inconsistent_shapes_and_non_boolean_alive(self):
        torch, ppo = self.torch, self.ppo
        for invalid in (
            torch.full((2, 4), 0.5),
            torch.zeros(2, 3),
            torch.full((2, 4), float("nan")),
        ):
            batch = {
                "rew": torch.zeros(2, 4),
                "val": torch.zeros(2, 4),
                "alive": invalid,
                "last_val": torch.zeros(4),
            }
            with self.assertRaises(ValueError):
                ppo.gae(batch)
        batch.update(rew=torch.ones(2, 4, dtype=torch.int64), alive=torch.ones(2, 4))
        with self.assertRaises(ValueError):
            ppo.gae(batch)

    def test_mirror_policy_equivariance_with_nontrivial_normalization(self):
        torch, ppo = self.torch, self.ppo
        torch.manual_seed(7)
        model = ppo.ActorCritic()
        model.absorb(torch.randn(100, 50) * 3 + 2)
        obs = torch.randn(8, 50)
        action = model(obs)[0]
        mirrored = model(obs[:, model.mirror] * model.sign)[0]
        torch.testing.assert_close(mirrored, action[:, model.legs] * model.leg_sign)

    def test_tracking_profiles_change_only_reward_weights(self):
        for profile, extra_track, extra_turn in (
            ("tracking-v1", 1.0, 0.0),
            ("tracking-turn-v1", 1.0, 0.5),
            ("tracking-moderate-v1", 1.5, 0.5),
            ("tracking-balanced-v1", 2.0, 0.5),
            ("tracking-strong-v1", 2.0, 1.0),
        ):
            with self.subTest(profile=profile):
                original = self.task.Go1(4, seed=4, num_threads=1)
                tracking = self.task.Go1(
                    4, seed=4, num_threads=1, reward_profile=profile
                )
                action = np.zeros((4, 12), dtype=np.float32)
                for _ in range(10):
                    old, new = original.step(action), tracking.step(action)
                    np.testing.assert_array_equal(original.obs(), tracking.obs())
                    for key in old[3]:
                        np.testing.assert_array_equal(old[3][key], new[3][key])
                    expected = (
                        sum(self.task.REWARD[k] * v for k, v in new[3].items())
                        + extra_track * new[3]["track"]
                        + extra_turn * new[3]["turn"]
                    )
                    np.testing.assert_allclose(
                        new[0], np.maximum(expected.astype(np.float32), 0), rtol=1e-6
                    )
                tracking.reward_weights["track"] = 99
                self.assertEqual(original.reward_weights["track"], 1)
                self.assertEqual(
                    self.task.REWARD_PROFILES[profile]["track"], 1 + extra_track
                )
        with self.assertRaisesRegex(ValueError, "reward profile"):
            self.task.Go1(1, reward_profile="unknown")

    def test_command_profiles_preserve_other_axes_rng_and_instance_defaults(self):
        original = self.task.Go1(32, seed=4, num_threads=1)
        lateral = self.task.Go1(32, seed=4, num_threads=1, command_profile="lateral-v1")
        counts = np.zeros(2, dtype=int)
        ids = np.arange(32)
        for _ in range(100):
            original.resample(ids, keep=0)
            lateral.resample(ids, keep=0)
            np.testing.assert_array_equal(
                original.command[:, [0, 2]], lateral.command[:, [0, 2]]
            )
            np.testing.assert_array_equal(original.until, lateral.until)
            enabled = original.command[:, 1] != 0
            np.testing.assert_array_equal(
                original.command[enabled, 1], lateral.command[enabled, 1]
            )
            counts += [enabled.sum(), (lateral.command[:, 1] != 0).sum()]
        self.assertTrue(0.20 < counts[0] / 3200 < 0.30)
        self.assertTrue(0.70 < counts[1] / 3200 < 0.80)
        self.assertEqual(
            original.rng.bit_generator.state, lateral.rng.bit_generator.state
        )
        # Evaluation supplies its own commands: the training distribution must
        # not change the physical initial state or deterministic responses.
        for _ in range(25):
            for env in (original, lateral):
                env.command[:] = [0, 0.3, 0]
                env.until[:] = 2
            old = original.step(np.zeros((32, 12), dtype=np.float32))
            new = lateral.step(np.zeros((32, 12), dtype=np.float32))
            np.testing.assert_array_equal(original.obs(), lateral.obs())
            np.testing.assert_array_equal(old[0], new[0])
        lateral.command_on[0] = 0
        self.assertEqual(original.command_on[0], 0.9)
        self.assertEqual(self.task.COMMAND_PROFILES["lateral-v1"]["on"][0], 0.9)
        with self.assertRaisesRegex(ValueError, "command profile"):
            self.task.Go1(1, command_profile="unknown")

    def test_whole_vector_stop_overrides_retention_only_for_selected_rows(self):
        env = self.task.Go1(4, seed=4, num_threads=1, command_profile="lateral-stop-v1")
        self.assertEqual(env.command_stop_probability, 0.1)
        original = self.task.Go1(4, seed=4, num_threads=1)
        for _ in range(25):
            for task in (env, original):
                task.command[:] = [0.5, 0.3, 0]
                task.until[:] = 2
                task.step(np.zeros((4, 12), dtype=np.float32))
            np.testing.assert_array_equal(env.obs(), original.obs())
        env.command[:] = [0.5, 0.3, 0.2]
        env.command_stop_probability = 1.0
        env.resample([3, 1], keep=1)
        np.testing.assert_array_equal(env.command[[1, 3]], np.zeros((2, 3)))
        np.testing.assert_array_equal(
            env.command[[0, 2]], np.array([[0.5, 0.3, 0.2]] * 2)
        )
        self.assertEqual(
            self.task.COMMAND_PROFILES["lateral-stop-v1"]["stop_probability"], 0.1
        )

    def test_stop_profiles_share_draws_and_only_increase_complete_stops(self):
        a = self.task.Go1(32, seed=4, num_threads=1, command_profile="lateral-stop-v1")
        b = self.task.Go1(32, seed=4, num_threads=1, command_profile="lateral-stop-v2")
        self.assertEqual(b.command_stop_probability, 0.2)
        counts = np.zeros(2, dtype=int)
        for _ in range(100):
            for env in (a, b):
                env.command[:] = [0.5, 0.3, 0.2]
                env.resample(np.arange(32), keep=1)
            stopped_a = (a.command == 0).all(1)
            stopped_b = (b.command == 0).all(1)
            self.assertTrue((~stopped_a | stopped_b).all())
            np.testing.assert_array_equal(a.command[~stopped_b], b.command[~stopped_b])
            np.testing.assert_array_equal(a.until, b.until)
            counts += [stopped_a.sum(), stopped_b.sum()]
        self.assertTrue(0.08 < counts[0] / 3200 < 0.12)
        self.assertTrue(0.17 < counts[1] / 3200 < 0.23)
        self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)
        for _ in range(25):
            for env in (a, b):
                env.command[:] = [0.5, 0.3, 0]
                env.until[:] = 2
                env.step(np.zeros((32, 12), dtype=np.float32))
            np.testing.assert_array_equal(a.obs(), b.obs())

    def test_command_switch_rewards_the_command_seen_by_the_action(self):
        fixed = self.task.Go1(4, seed=8, num_threads=1)
        control = self.task.Go1(4, seed=8, num_threads=1)
        legacy = self.task.Go1(4, seed=8, num_threads=1, semantics="upstream-v1")
        next_command = np.array([1.0, 0.0, 0.8])
        for env in (fixed, control, legacy):
            env.command[:] = 0
            env.until[:] = 1 if env is not control else 100
            env.clock[:] = 0.125

        def switch(env):
            def resample(ids, keep=0.5):
                env.command[ids] = next_command
                env.until[ids] = 20

            return resample

        fixed.resample, legacy.resample = switch(fixed), switch(legacy)
        np.testing.assert_array_equal(fixed.obs(), control.obs())
        action = np.zeros((4, 12), dtype=np.float32)
        corrected, expected, old = (
            fixed.step(action),
            control.step(action),
            legacy.step(action),
        )
        for key in expected[3]:
            np.testing.assert_array_equal(corrected[3][key], expected[3][key])
        np.testing.assert_array_equal(corrected[0], expected[0])
        self.assertGreater(np.max(np.abs(old[3]["track"] - corrected[3]["track"])), 0.5)
        np.testing.assert_array_equal(
            fixed.obs()[:, 45:48],
            np.broadcast_to(next_command.astype(np.float32), (4, 3)),
        )
        np.testing.assert_array_equal(fixed.obs(), legacy.obs())

    def test_gae_stops_at_reset_and_bootstraps_timeouts(self):
        torch, ppo = self.torch, self.ppo
        # Timeout bootstrap is folded into the reward by rollout; alive=0
        # prevents the following episode's large value from leaking backward.
        batch = {
            "rew": torch.tensor([[1.0 + ppo.GAMMA * 5], [2.0]]),
            "val": torch.tensor([[3.0], [100.0]]),
            "alive": torch.tensor([[0.0], [1.0]]),
            "last_val": torch.tensor([7.0]),
        }
        adv, ret = ppo.gae(batch)
        self.assertAlmostEqual(ret[0, 0].item(), 1 + ppo.GAMMA * 5, places=5)
        self.assertAlmostEqual(adv[1, 0].item(), 2 + ppo.GAMMA * 7 - 100, places=5)

    def test_action_only_path_matches_forward_and_skips_critic(self):
        torch, ppo = self.torch, self.ppo
        torch.manual_seed(14)
        net = ppo.ActorCritic()
        env = self.task.Go1(8, seed=4, num_threads=2)
        obs = torch.as_tensor(env.obs())
        calls = []
        hook = net.critic.register_forward_hook(lambda *args: calls.append(1))
        try:
            action = net.action_mean(obs)
            self.assertEqual(calls, [])
            expected, _ = net(obs)
            self.assertEqual(calls, [1])
            torch.testing.assert_close(action, expected, rtol=0, atol=0)
            other = deepcopy(net)
            net.zero_grad(set_to_none=True)
            action.square().sum().backward()
            other(obs)[0].square().sum().backward()
            for a, b in zip(
                net.actor.parameters(), other.actor.parameters(), strict=True
            ):
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in net.critic.parameters()))
        finally:
            hook.remove()

    def test_fast_command_profiles_change_interval_and_retention_only(self):
        slow = self.task.Go1(
            8, seed=4, num_threads=1, command_profile="lateral-stop-v1"
        )
        fast = self.task.Go1(
            8, seed=4, num_threads=1, command_profile="lateral-fast-v1"
        )
        full = self.task.Go1(
            8, seed=4, num_threads=1, command_profile="lateral-fast-full-v1"
        )
        override = self.task.Go1(
            8, seed=4, num_threads=1, command_profile="lateral-fast-v1"
        )
        moderate = self.task.Go1(
            8, seed=4, num_threads=1, command_profile="lateral-fast-moderate-v1"
        )
        ids = np.arange(8)
        self.assertEqual(slow.command_keep, 0.5)
        self.assertEqual(full.command_keep, 0)
        for _ in range(50):
            slow.resample(ids)
            fast.resample(ids)
            full.resample(ids)
            override.resample(ids, keep=0)
            moderate.resample(ids)
            np.testing.assert_array_equal(slow.command, fast.command)
            np.testing.assert_allclose(fast.until, slow.until * 0.4, rtol=1e-14)
            np.testing.assert_array_equal(full.command, override.command)
            np.testing.assert_array_equal(full.until, override.until)
            np.testing.assert_allclose(
                moderate.command,
                full.command * [0.8 / 1.5, 0.5 / 0.8, 1.0 / 1.2],
                rtol=1e-14,
            )
            np.testing.assert_array_equal(full.until, moderate.until)
        self.assertEqual(slow.rng.bit_generator.state, fast.rng.bit_generator.state)
        self.assertEqual(full.rng.bit_generator.state, override.rng.bit_generator.state)
        self.assertEqual(full.rng.bit_generator.state, moderate.rng.bit_generator.state)
        for _ in range(25):
            for env in (slow, fast, full, override, moderate):
                env.command[:] = [0.5, 0.3, 0]
                env.until[:] = 2
                env.step(np.zeros((8, 12), dtype=np.float32))
            for env in (fast, full, override, moderate):
                np.testing.assert_array_equal(slow.obs(), env.obs())

    def test_recording_policy_preserves_rollout_update_and_sampler_rng(self):
        torch, ppo = self.torch, self.ppo
        torch.manual_seed(18)
        a = ppo.ActorCritic()
        b = deepcopy(a)
        env_a = self.task.Go1(8, seed=4, num_threads=2)
        env_b = self.task.Go1(8, seed=4, num_threads=2)
        plain, stats_a = ppo.rollout(a, env_a, horizon=8)
        recorded, stats_b = ppo.rollout(b, env_b, horizon=8, record_policy=True)
        self.assertEqual(stats_a, stats_b)
        self.assertEqual(env_a.rng.bit_generator.state, env_b.rng.bit_generator.state)
        for key in plain:
            torch.testing.assert_close(plain[key], recorded[key], rtol=0, atol=0)
        with torch.no_grad():
            for step in range(8):
                torch.testing.assert_close(
                    recorded["policy_mean"][step],
                    b.action_mean(recorded["obs"][step]),
                    rtol=0,
                    atol=0,
                )
        torch.testing.assert_close(
            recorded["policy_log_std"], b.log_std, rtol=0, atol=0
        )
        opt_a = torch.optim.Adam(a.parameters(), lr=0.00003)
        opt_b = torch.optim.Adam(b.parameters(), lr=0.00003)
        torch.manual_seed(17)
        metrics_a = ppo.update(a, opt_a, plain, *ppo.gae(plain), diagnostics=True)
        rng = torch.get_rng_state()
        torch.manual_seed(17)
        metrics_b = ppo.update(b, opt_b, recorded, *ppo.gae(recorded), diagnostics=True)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key in a.state_dict():
            torch.testing.assert_close(
                a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0
            )
        for i, state in opt_a.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value, opt_b.state_dict()["state"][i][key], rtol=0, atol=0
                )
        for key in metrics_a:
            np.testing.assert_allclose(
                metrics_a[key], metrics_b[key], rtol=1e-5, atol=1e-6
            )

    def test_cached_kl_measures_the_sampling_policy_after_external_actor_drift(self):
        torch, ppo = self.torch, self.ppo
        torch.manual_seed(18)
        model = ppo.ActorCritic()
        env = self.task.Go1(8, seed=4, num_threads=2)
        batch, _ = ppo.rollout(model, env, horizon=8, record_policy=True)
        with torch.no_grad():
            model.actor[-1].bias.add_(0.15)
        other = deepcopy(model)
        plain = {k: v for k, v in batch.items() if not k.startswith("policy_")}
        adv, ret = ppo.gae(batch)
        cached = ppo.update(
            model,
            torch.optim.Adam(model.parameters(), lr=0),
            batch,
            adv,
            ret,
            diagnostics=True,
        )
        recomputed = ppo.update(
            other,
            torch.optim.Adam(other.parameters(), lr=0),
            plain,
            adv,
            ret,
            diagnostics=True,
        )
        # No optimizer step can change the policy at zero LR. Only the cached
        # distribution remembers that actions were sampled before the bias edit.
        self.assertGreater(cached["ppo_kl_before_normalization"], 0.1)
        self.assertEqual(recomputed["ppo_kl_before_normalization"], 0)

    def test_invalid_recorded_policy_is_rejected_before_optimizer_changes(self):
        torch, ppo = self.torch, self.ppo
        model = ppo.ActorCritic()
        env = self.task.Go1(8, seed=4, num_threads=2)
        batch, _ = ppo.rollout(model, env, horizon=8, record_policy=True)
        adv, ret = ppo.gae(batch)
        opt = torch.optim.Adam(model.parameters(), lr=0.00003)
        before = deepcopy(model.state_dict())
        for kind in ("missing_mean", "missing_std", "shape", "nan", "dtype"):
            bad = dict(batch)
            if kind == "missing_mean":
                del bad["policy_mean"]
            elif kind == "missing_std":
                del bad["policy_log_std"]
            elif kind == "shape":
                bad["policy_mean"] = bad["policy_mean"][0]
            elif kind == "nan":
                bad["policy_log_std"] = torch.full((12,), float("nan"))
            else:
                bad["policy_mean"] = bad["policy_mean"].double()
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "recorded policy"),
            ):
                ppo.update(model, opt, bad, adv, ret, diagnostics=True)
        self.assertEqual(opt.state_dict()["state"], {})
        for key in before:
            torch.testing.assert_close(
                before[key], model.state_dict()[key], rtol=0, atol=0
            )

    def test_policy_kl_matches_torch_distributions_and_is_directional(self):
        torch, ppo = self.torch, self.ppo
        old_mean = torch.zeros(3, 12, dtype=torch.float64)
        new_mean = torch.full_like(old_mean, 0.2)
        old_std = torch.ones(12, dtype=torch.float64)
        new_std = torch.linspace(0.5, 1.5, 12, dtype=torch.float64)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mean, old_std),
            torch.distributions.Normal(new_mean, new_std),
        ).sum(-1)
        actual = ppo.policy_kl(old_mean, old_std.log(), new_mean, new_std.log())
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(
            (ppo.policy_kl(old_mean, old_std.log(), old_mean, old_std.log()) == 0).all()
        )
        reverse = ppo.policy_kl(new_mean, new_std.log(), old_mean, old_std.log())
        self.assertFalse(torch.allclose(actual, reverse))

    def test_ppo_diagnostics_preserve_model_optimizer_and_rng(self):
        torch, ppo = self.torch, self.ppo
        torch.manual_seed(23)
        a = ppo.ActorCritic()
        b = deepcopy(a)
        env = self.task.Go1(8, seed=4, num_threads=2)
        batch, _ = ppo.rollout(a, env, horizon=8)
        adv, ret = ppo.gae(batch)
        opt_a = torch.optim.Adam(a.parameters(), lr=0.00005)
        opt_b = torch.optim.Adam(b.parameters(), lr=0.00005)
        torch.manual_seed(6)
        self.assertIsNone(ppo.update(a, opt_a, batch, adv, ret))
        expected_rng = torch.get_rng_state()
        torch.manual_seed(6)
        metrics = ppo.update(b, opt_b, batch, adv, ret, diagnostics=True)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key in a.state_dict():
            torch.testing.assert_close(
                a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0
            )
        for i, state in opt_a.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value, opt_b.state_dict()["state"][i][key], rtol=0, atol=0
                )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertGreaterEqual(metrics["ppo_max_kl"], metrics["ppo_kl"])
        self.assertGreaterEqual(metrics["ppo_normalizer_kl"], 0)
        self.assertGreaterEqual(metrics["ppo_clip_fraction"], 0)
        self.assertLessEqual(metrics["ppo_clip_fraction"], 1)
        self.assertGreater(metrics["ppo_mean_gradient_norm"], 0)

    def test_upstream_physics_observations_rewards_and_partial_reset(self):
        owner = self.reference()
        owner.EPISODE = 32
        a = owner.Go1(8, seed=4)
        b = self.task.Go1(
            8, seed=4, num_threads=2, episode_steps=32, semantics="upstream-v1"
        )
        rng = np.random.default_rng(9)
        for step in range(100):
            np.testing.assert_array_equal(a.obs(), b.obs())
            action = rng.uniform(-0.4, 0.4, (8, 12)).astype(np.float32)
            old, new = a.step(action), b.step(action)
            for left, right in zip(old[:3], new[:3], strict=True):
                np.testing.assert_array_equal(left, right)
            for key in old[3]:
                np.testing.assert_array_equal(old[3][key], new[3][key])
            np.testing.assert_array_equal(a.qpos, b.qpos)
            if step % 15 == 0:
                ids = np.array([1, 4, 7])
                a.reset(ids)
                b.reset(ids)

    def test_upstream_rollout_gae_and_optimizer_update(self):
        owner = self.reference()
        torch, ppo = self.torch, self.ppo
        owner.NUM_ENVS, owner.HORIZON, owner.EPISODE = 8, 8, 3
        a = owner.Go1(8, seed=4)
        b = self.task.Go1(
            8, seed=4, num_threads=2, episode_steps=3, semantics="upstream-v1"
        )
        torch.manual_seed(2)
        old = owner.ActorCritic()
        new = ppo.ActorCritic()
        new.load_state_dict(old.state_dict(), strict=True)
        old_batch, old_stats = owner.rollout(old, a)
        new_batch, new_stats = ppo.rollout(new, b, horizon=8)
        self.assertEqual(old_stats, new_stats)
        for key in old_batch:
            torch.testing.assert_close(old_batch[key], new_batch[key], rtol=0, atol=0)
        old_gae, new_gae = owner.gae(old_batch), ppo.gae(new_batch)
        for left, right in zip(old_gae, new_gae, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        old_opt = torch.optim.Adam(old.parameters(), lr=ppo.LR)
        new_opt = torch.optim.Adam(new.parameters(), lr=ppo.LR)
        torch.manual_seed(5)
        owner.update(old, old_opt, old_batch, *old_gae)
        torch.manual_seed(5)
        ppo.update(new, new_opt, new_batch, *new_gae)
        for key in old.state_dict():
            torch.testing.assert_close(
                old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0
            )
        for i, state in old_opt.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(
                    value, new_opt.state_dict()["state"][i][key], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
