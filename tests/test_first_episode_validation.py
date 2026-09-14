from copy import deepcopy

import numpy as np
import pytest

from embodiedforge._h1_metrics import FirstEpisodeMetrics, validate_first_episode


def unequal_episodes():
    metrics = FirstEpisodeMetrics(2, 4, 0.02)
    for step in range(4):
        metrics.update(
            np.array([[3.0, 4.0, 2.0], [0.0, 0.0, 0.0]]),
            np.zeros((2, 3)),
            np.array([step == 0, False]),
            np.zeros(2, bool),
        )
    return metrics.report()


def test_pooled_rmse_uses_frames_and_includes_fall_frame():
    report = unequal_episodes()
    validate_first_episode(report)
    assert report["per_env"]["observed_steps"] == [1, 4]
    assert report["planar_rmse"] == pytest.approx(np.sqrt(5))
    assert report["yaw_rmse"] == pytest.approx(np.sqrt(0.8))
    assert report["survival_fraction"] == 0.5
    assert report["mean_velocity"] == pytest.approx([0.6, 0.8, 0.4])
    report["planar_rmse"] = 2.5  # Averaging per-environment RMSE is not pooling.
    with pytest.raises(ValueError, match="planar_rmse"):
        validate_first_episode(report)


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_envs", True),
        ("steps_limit", 0),
        ("steps_executed", 5),
        ("dt", 0),
        ("dt", float("nan")),
        ("dt", True),
        ("survival_fraction", 1),
        ("fall_fraction", 0),
        ("truncation_fraction", 1),
        ("horizon_seconds", 100),
        ("mean_observed_seconds", 0.08),
        ("mean_velocity", [0, 0, 0]),
        ("yaw_rmse", float("nan")),
    ],
)
def test_corrupted_first_episode_summary_is_rejected(key, value):
    report = unequal_episodes()
    report[key] = value
    with pytest.raises(ValueError):
        validate_first_episode(report)


@pytest.mark.parametrize(
    "key,value",
    [
        ("observed_steps", [0, 4]),
        ("observed_steps", [1, 3]),
        ("observed_steps", [1.0, 4.0]),
        ("observed_steps", [1, 5]),
        ("terminated", [1, 0]),
        ("truncated", [False]),
        ("yaw_rmse", [float("nan"), 0]),
        ("planar_rmse", [-5, 0]),
        ("mean_velocity", [[3, 4, 2]]),
    ],
)
def test_corrupted_first_episode_rows_are_rejected(key, value):
    report = unequal_episodes()
    report["per_env"][key] = value
    with pytest.raises(ValueError):
        validate_first_episode(report)


def test_completed_truncation_and_early_all_fall_are_valid():
    for flag in ("terminated", "truncated"):
        metrics = FirstEpisodeMetrics(2, 4, 0.02)
        metrics.update(
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            np.full(2, flag == "terminated"),
            np.full(2, flag == "truncated"),
        )
        report = metrics.report()
        validate_first_episode(report)
        copy = deepcopy(report)
        copy["per_env"][flag] = [False, False]
        with pytest.raises(ValueError, match="Incomplete"):
            validate_first_episode(copy)
