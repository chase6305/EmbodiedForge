"""Compare saved ACT benchmark episodes without importing the training SDK."""

import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path

from .core import Config


def _load_report(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    report = json.loads(content)
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != 1
        or report.get("mode") != "paired_single_episode_per_environment"
    ):
        raise ValueError(
            "Comparison requires paired benchmark reports, not evaluate reports"
        )
    seeds = report.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Benchmark seeds must be nonnegative and unique")
    count, max_steps = report.get("num_envs"), report.get("max_steps")
    if any(type(value) is not int or value <= 0 for value in (count, max_steps)):
        raise ValueError("Benchmark num_envs and max_steps must be positive integers")
    rows = report.get("per_seed")
    if not isinstance(rows, list) or len(rows) != len(seeds):
        raise ValueError("Benchmark must contain exactly one row per seed")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Benchmark rows must be objects")
        seed = row.get("seed")
        if type(seed) is not int or seed not in seeds or seed in indexed:
            raise ValueError(
                "Benchmark rows have missing, duplicate or unexpected seeds"
            )
        config = row.get("environment_config")
        if not isinstance(config, dict) or set(config) != {
            f.name for f in fields(Config)
        }:
            raise ValueError(
                "Missing complete per-seed environment_config; rerun benchmark"
            )
        Config(**config)  # Validate without supplying defaults for missing fields.
        if (config["seed"], config["num_envs"], config["max_steps"]) != (
            seed,
            count,
            max_steps,
        ):
            raise ValueError(
                "Per-seed environment_config disagrees with benchmark settings"
            )
        for name in ("act", "expert"):
            policy = row.get(name, {})
            if not isinstance(policy, dict):
                raise ValueError(f"Benchmark {name} results must be an object")
            for key in ("episode_success", "episode_return", "episode_length"):
                values = policy.get(key)
                if not isinstance(values, list) or len(values) != count:
                    raise ValueError(
                        f"{name}.{key} must contain one value per environment"
                    )
            if any(type(value) is not bool for value in policy["episode_success"]):
                raise ValueError("episode_success must contain booleans")
            if any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in policy["episode_return"]
            ):
                raise ValueError("episode_return must contain finite numbers")
            if any(
                type(value) is not int or not 1 <= value <= max_steps
                for value in policy["episode_length"]
            ):
                raise ValueError("episode_length must be within the episode step limit")
            digest = policy.get("initial_observation_sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("Invalid initial observation hash")
        if (
            row["act"]["initial_observation_sha256"]
            != row["expert"]["initial_observation_sha256"]
        ):
            raise ValueError("ACT and expert initial observations differ")
        indexed[seed] = row
    source = {
        "path": str(path),
        "report_sha256": hashlib.sha256(content).hexdigest(),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "checkpoint_fingerprint": report.get("checkpoint_fingerprint"),
        "inference_settings": report.get("inference_settings"),
    }
    return report, indexed, source


def _summarize(episodes):
    count = len(episodes)
    baseline = sum(row["baseline_success"] for row in episodes)
    candidate = sum(row["candidate_success"] for row in episodes)
    return {
        "episodes": count,
        "baseline_successes": baseline,
        "candidate_successes": candidate,
        "baseline_success_rate": baseline / count,
        "candidate_success_rate": candidate / count,
        "success_rate_delta_percentage_points": 100 * (candidate - baseline) / count,
        "improved": sum(
            not row["baseline_success"] and row["candidate_success"] for row in episodes
        ),
        "regressed": sum(
            row["baseline_success"] and not row["candidate_success"] for row in episodes
        ),
        "both_succeeded": sum(
            row["baseline_success"] and row["candidate_success"] for row in episodes
        ),
        "both_failed": sum(
            not row["baseline_success"] and not row["candidate_success"]
            for row in episodes
        ),
        "mean_episode_return_delta": math.fsum(
            row["return_delta"] / count for row in episodes
        ),
        "mean_episode_length_delta": math.fsum(row["length_delta"] for row in episodes)
        / count,
    }


def compare_reports(baseline_path, candidate_path):
    """Pair by (seed, environment row); compute descriptive changes from raw episodes."""
    baseline, before, baseline_source = _load_report(baseline_path)
    candidate, after, candidate_source = _load_report(candidate_path)
    if set(before) != set(after):
        raise ValueError("Benchmark seed sets differ")
    if any(baseline[key] != candidate[key] for key in ("num_envs", "max_steps")):
        raise ValueError("Benchmark num_envs or max_steps differ")
    per_seed = []
    all_episodes = []
    for seed in sorted(before):
        left, right = before[seed], after[seed]
        if left["environment_config"] != right["environment_config"]:
            raise ValueError(
                f"Benchmark environment configurations differ for seed {seed}"
            )
        if (
            left["act"]["initial_observation_sha256"]
            != right["act"]["initial_observation_sha256"]
        ):
            raise ValueError(f"Benchmark initial observations differ for seed {seed}")
        for key in ("episode_success", "episode_return", "episode_length"):
            if left["expert"][key] != right["expert"][key]:
                raise ValueError(
                    f"Benchmark expert reference differs for seed {seed}: {key}"
                )
        episodes = [
            {
                "env_id": index,
                "baseline_success": left["act"]["episode_success"][index],
                "candidate_success": right["act"]["episode_success"][index],
                "return_delta": right["act"]["episode_return"][index]
                - left["act"]["episode_return"][index],
                "length_delta": right["act"]["episode_length"][index]
                - left["act"]["episode_length"][index],
            }
            for index in range(baseline["num_envs"])
        ]
        if any(not math.isfinite(row["return_delta"]) for row in episodes):
            raise ValueError("Episode return difference is not finite")
        per_seed.append(
            {"seed": seed, "summary": _summarize(episodes), "episodes": episodes}
        )
        all_episodes.extend(episodes)
    return {
        "schema_version": 1,
        "mode": "paired_benchmark_comparison",
        "scope": "descriptive_observed_episodes_not_statistical_significance",
        "delta_direction": "candidate_minus_baseline",
        "baseline": baseline_source,
        "candidate": candidate_source,
        "seeds": sorted(before),
        "num_envs": baseline["num_envs"],
        "summary": _summarize(all_episodes),
        "per_seed": per_seed,
    }
