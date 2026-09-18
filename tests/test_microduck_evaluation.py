"""Evaluation command invariants and action parity failure boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest

from embodiedforge._microduck_worker import (
    EvaluationMetrics,
    compare_policy_actions,
    configure_evaluation,
    verify_fixed_commands,
)


def test_parity_accepts_roundoff_but_rejects_wrong_finite_policy():
    reference = np.linspace(-1, 1, 14, dtype=np.float32).reshape(1, 14)
    assert compare_policy_actions(reference, reference + 1e-5) < 2e-5
    with pytest.raises(RuntimeError, match="action mismatch"):
        compare_policy_actions(reference, reference + 0.01)


@pytest.mark.parametrize(
    "candidate", [np.zeros((14,)), np.full((1, 14), np.nan), np.full((1, 14), np.inf)]
)
def test_parity_rejects_invalid_output(candidate):
    with pytest.raises(RuntimeError, match="finite"):
        compare_policy_actions(np.zeros((1, 14)), candidate)


def test_parity_uses_elementwise_tolerance():
    reference = np.zeros((1, 14))
    reference[0, 0] = 100
    candidate = reference.copy()
    candidate[0, 1] = 0.001  # A large action elsewhere cannot mask this mismatch.
    with pytest.raises(RuntimeError, match="action mismatch"):
        compare_policy_actions(reference, candidate)


def test_fixed_commands_preserve_physics_and_remove_command_curricula():
    cfg = SimpleNamespace(
        events={
            "push_robot": object(),
            "randomize_com": object(),
            "bam_startup": object(),
        },
        curriculum={
            name: object()
            for name in (
                "standing_envs",
                "head_pose_range",
                "body_pose_range",
                "com_range",
                "action_rate_weight",
            )
        },
        commands={
            "twist": SimpleNamespace(ranges=SimpleNamespace(heading=(-3.14, 3.14))),
            "head_pose": SimpleNamespace(ranges=((-1, 1),) * 4),
            "body_pose": SimpleNamespace(ranges=((-1, 1),) * 6),
        },
    )
    retained_event = cfg.events["randomize_com"]
    retained_curriculum = cfg.curriculum["com_range"]
    configure_evaluation(cfg, [-0.2, 0.1, -0.5], True)
    assert cfg.commands["twist"].ranges.lin_vel_x == (-0.2, -0.2)
    assert cfg.commands["twist"].ranges.ang_vel_z == (-0.5, -0.5)
    assert cfg.commands["twist"].rel_turn_in_place_envs == 0
    assert cfg.commands["twist"].heading_command is False
    assert cfg.commands["twist"].ranges.heading is None
    assert cfg.commands["head_pose"].ranges == ((0, 0),) * 4
    assert cfg.commands["body_pose"].ranges == ((0, 0),) * 6
    assert set(cfg.curriculum) == {"com_range", "action_rate_weight"}
    assert cfg.curriculum["com_range"] is retained_curriculum
    assert cfg.events["randomize_com"] is retained_event
    assert "bam_startup" in cfg.events and "push_robot" not in cfg.events


def test_default_evaluation_leaves_upstream_config_untouched():
    configure_evaluation(SimpleNamespace(), None, False)


def test_runtime_command_guard_detects_reset_or_curriculum_changes():
    torch = pytest.importorskip("torch")
    commands = {
        "twist": torch.tensor([[0.2, 0, 0], [0.2, 0, 0]]),
        "head_pose": torch.zeros(2, 4),
        "body_pose": torch.zeros(2, 6),
    }
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=commands.__getitem__)
    )
    verify_fixed_commands(env, [0.2, 0, 0])
    for value in (0.0, float("nan"), float("inf")):
        commands["twist"][1, 0] = value
        with pytest.raises(RuntimeError, match="changed: twist"):
            verify_fixed_commands(env, [0.2, 0, 0])
    commands["twist"][1, 0] = 0.2
    commands["head_pose"][0, 1] = 0.1
    commands["body_pose"][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="changed: head_pose"):
        verify_fixed_commands(env, [0.2, 0, 0])
    commands["head_pose"].zero_()
    with pytest.raises(RuntimeError, match="changed: body_pose"):
        verify_fixed_commands(env, [0.2, 0, 0])
    commands["body_pose"][0, 0] = 0.5e-6
    verify_fixed_commands(env, [0.2, 0, 0])


@pytest.fixture
def metric_env():
    torch = pytest.importorskip("torch")
    terms = {
        name: torch.zeros(2, dtype=torch.bool)
        for name in ("fell_over", "time_out", "nan_state")
    }
    data = SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(2, 3),
        root_link_ang_vel_b=torch.zeros(2, 3),
    )
    command = torch.zeros(2, 3)
    return SimpleNamespace(
        num_envs=2,
        device="cpu",
        cfg=SimpleNamespace(terminations=terms),
        scene={"robot": SimpleNamespace(data=data)},
        command_manager=SimpleNamespace(get_command=lambda _: command),
        termination_manager=SimpleNamespace(get_term=terms.__getitem__),
        reset_terminated=terms["fell_over"],
        reset_time_outs=terms["time_out"],
    )


def test_tracking_includes_terminal_state_and_original_command(metric_env):
    env = metric_env
    data = env.scene["robot"].data
    metric = EvaluationMetrics(None, env)
    # A terminal transition: the first env moves at 2 m/s against a 1 m/s command.
    data.root_link_lin_vel_b[0, 0] = 2
    env.command_manager.get_command("twist")[0, 0] = 1
    env.reset_terminated[0] = True
    metric(env)
    # The SDK now resets this env and resamples its command. These values must
    # not retroactively enter the terminal transition or the first-episode record.
    data.root_link_lin_vel_b.zero_()
    env.command_manager.get_command("twist")[0, 0] = -1
    env.reset_terminated.zero_()
    metric(env)
    report = metric.report(steps=2, step_dt=0.02)
    tracking = report["velocity_tracking"]
    assert tracking["samples"] == 4
    assert tracking["mean_command"] == [0, 0, 0]
    assert tracking["mean_actual"] == [0.5, 0, 0]
    assert tracking["mae"] == [0.5, 0, 0]
    assert tracking["rmse"] == pytest.approx([np.sqrt(0.5), 0, 0])
    per_env = tracking["per_environment"]
    assert per_env["samples_per_environment"] == 2
    assert per_env["mean_actual"] == [[1, 0, 0], [0, 0, 0]]
    assert per_env["rmse"] == [[1, 0, 0], [0, 0, 0]]
    assert report["termination_counts"]["fell_over"] == 1
    assert report["completed_episodes"] == 1
    assert report["mean_completed_episode_steps"] == 1
    assert report["unfinished_episode_steps"] == [1, 2]
    assert report["initial_episodes"]["duration_steps"] == [1, 2]
    assert report["initial_episodes"]["right_censored"] == [False, True]
    assert report["initial_episodes"]["survived_full_horizon_count"] == 1


def test_all_velocity_axes_and_units_are_independent(metric_env):
    env = metric_env
    data = env.scene["robot"].data
    data.root_link_lin_vel_b[:, 0] = -3
    data.root_link_lin_vel_b[:, 1] = 4
    data.root_link_lin_vel_b[:, 2] = 999  # Vertical speed is not yaw rate.
    data.root_link_ang_vel_b[:, 2] = -2
    metric = EvaluationMetrics(None, env)
    instantaneous_error = metric(env)
    assert instantaneous_error.tolist() == [5.0, 5.0]
    report = metric.report(steps=1, step_dt=0.02)["velocity_tracking"]
    assert report["mean_actual"] == [-3, 4, -2]
    assert report["mae"] == [3, 4, 2]
    assert report["rmse"] == [3, 4, 2]
    assert report["planar_velocity_rmse_m_s"] == 5


def test_per_environment_metrics_expose_errors_cancelled_by_batch_average(metric_env):
    env = metric_env
    env.scene["robot"].data.root_link_lin_vel_b[1, 0] = 2
    env.command_manager.get_command("twist")[:, 0] = 1
    metric = EvaluationMetrics(None, env)
    metric(env)
    metric(env)
    tracking = metric.report(steps=2, step_dt=0.02)["velocity_tracking"]
    assert tracking["mean_actual"] == tracking["mean_command"] == [1, 0, 0]
    detail = tracking["per_environment"]
    assert detail["mean_actual"] == [[0, 0, 0], [2, 0, 0]]
    assert detail["bias"] == [[-1, 0, 0], [1, 0, 0]]
    assert detail["planar_velocity_rmse_m_s"] == [1, 1]
    assert detail["rmse"] == [[1, 0, 0], [1, 0, 0]]


def test_first_episode_is_not_overwritten_by_later_failures(metric_env):
    env = metric_env
    metric = EvaluationMetrics(None, env)
    for step in range(4):
        env.reset_terminated[0] = step in (1, 3)
        metric(env)
    report = metric.report(steps=4, step_dt=0.02)
    first = report["initial_episodes"]
    assert first["duration_steps"] == [2, 4]
    assert first["duration_seconds"] == [0.04, 0.08]
    assert first["mean_observed_duration_seconds"] == pytest.approx(0.06)
    assert report["completed_episodes"] == 2
    assert report["unfinished_episode_steps"] == [0, 4]


def test_time_limits_are_censored_and_not_counted_as_surviving_longer(metric_env):
    env = metric_env
    metric = EvaluationMetrics(None, env)
    env.reset_time_outs[0] = True
    metric(env)
    env.reset_time_outs.zero_()
    metric(env)
    first = metric.report(steps=2, step_dt=0.02)["initial_episodes"]
    assert first["duration_steps"] == [1, 2]
    assert first["terminated"] == [False, False]
    assert first["time_limit_only"] == [True, False]
    assert first["right_censored"] == [True, True]
    assert first["survived_full_horizon_count"] == 1
    assert first["censored_before_horizon_count"] == 1


def test_overlapping_timeout_and_failure_is_a_failure(metric_env):
    env = metric_env
    metric = EvaluationMetrics(None, env)
    env.reset_terminated[0] = env.reset_time_outs[0] = True
    metric(env)
    report = metric.report(steps=1, step_dt=0.02)
    assert report["completed_episodes"] == 1
    assert report["initial_episodes"]["right_censored"] == [False, True]
    assert report["initial_episodes"]["time_limit_only"] == [False, False]


@pytest.mark.parametrize("invalid", ["velocity", "command", "nan_state"])
def test_nonfinite_terminal_metrics_cannot_be_hidden_by_reset(metric_env, invalid):
    env = metric_env
    metric = EvaluationMetrics(None, env)
    data = env.scene["robot"].data
    if invalid == "velocity":
        data.root_link_lin_vel_b[0, 0] = float("nan")
    elif invalid == "command":
        env.command_manager.get_command("twist")[0, 0] = float("inf")
    else:
        env.cfg.terminations["nan_state"][0] = True
    metric(env)
    data.root_link_lin_vel_b.zero_()
    env.command_manager.get_command("twist").zero_()
    env.cfg.terminations["nan_state"].zero_()
    with pytest.raises(RuntimeError, match="Nonfinite"):
        metric.report(steps=1, step_dt=0.02)


def test_missing_metric_callbacks_fail_instead_of_reporting_partial_data(metric_env):
    metric = EvaluationMetrics(None, metric_env)
    with pytest.raises(RuntimeError, match="callbacks"):
        metric.report(steps=1, step_dt=0.02)


@pytest.mark.parametrize("override", [None, 0, 123])
def test_checkpoint_curriculum_counter_is_set_before_wrapper_reset(
    monkeypatch, override
):
    import sys
    from dataclasses import dataclass

    from embodiedforge import _microduck_worker as worker

    @dataclass
    class AgentCfg:
        clip_actions: float | None = None

    cfg = SimpleNamespace(
        scene=SimpleNamespace(), sim=SimpleNamespace(nan_guard=SimpleNamespace())
    )
    env = SimpleNamespace(common_step_counter=0, close=lambda: None)
    monkeypatch.setattr(
        worker, "checkpoint_metadata", lambda _: {"common_step_counter": 48000}
    )

    def wrapper(instance, **kwargs):
        assert instance.common_step_counter == (48000 if override is None else override)
        return instance

    def restore_checkpoint(*args, **kwargs):
        assert kwargs["load_cfg"] == {"actor": True}
        assert kwargs["map_location"] == "cpu"
        env.common_step_counter = 48000

    runner = SimpleNamespace(
        load=restore_checkpoint, get_inference_policy=lambda **kw: None
    )
    modules = {
        "mjlab.envs": SimpleNamespace(ManagerBasedRlEnv=lambda **kw: env),
        "mjlab.rl": SimpleNamespace(
            MjlabOnPolicyRunner=lambda *a, **kw: runner, RslRlVecEnvWrapper=wrapper
        ),
        "mjlab.tasks.registry": SimpleNamespace(
            load_env_cfg=lambda *a, **kw: cfg,
            load_rl_cfg=lambda _: AgentCfg(),
            load_runner_cls=lambda _: None,
        ),
        "mjlab.utils.torch": SimpleNamespace(
            configure_torch_backends=lambda **kw: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    with worker.policy_environment(None, 2, 0, curriculum_step=override):
        assert env.common_step_counter == (48000 if override is None else override)
