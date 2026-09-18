"""Paired report comparisons must detect regressions and reject mismatched trials."""

import copy
import hashlib
import json
import subprocess
import sys
from argparse import Namespace

import pytest

from embodiedforge import Config
from embodiedforge._act_compare import compare_reports
from embodiedforge.act import compare


@pytest.fixture
def reports(tmp_path):
    def report(candidate):
        rows = []
        for seed in (2001, 2002):
            policy = {
                "episode_success": [False, True, True, False],
                "episode_return": [1.0, 2.0, 3.0, 4.0],
                "episode_length": [5, 6, 7, 8],
                "initial_observation_sha256": hashlib.sha256(
                    str(seed).encode()
                ).hexdigest(),
            }
            row = {
                "seed": seed,
                "environment_config": Config(
                    seed=seed, num_envs=4, max_steps=10
                ).to_dict(),
                "act": copy.deepcopy(policy),
                "expert": copy.deepcopy(policy),
            }
            if candidate:
                row["act"].update(
                    episode_success=[True, False, True, False],
                    episode_return=[3.0, 1.0, 3.0, 8.0],
                    episode_length=[4, 7, 7, 8],
                )
            rows.append(row)
        return {
            "schema_version": 1,
            "mode": "paired_single_episode_per_environment",
            "seeds": [2001, 2002],
            "num_envs": 4,
            "max_steps": 10,
            "per_seed": rows,
            # Deliberately stale summary: raw episode records are authoritative.
            "summary": {"act": {"success_rate": 1.0}},
        }

    paths = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for path, candidate in zip(paths, (False, True), strict=True):
        path.write_text(json.dumps(report(candidate)))
    return paths


def test_equal_success_rates_still_expose_regressed_episodes(reports):
    result = compare_reports(*reports)
    assert result["summary"] == {
        "episodes": 8,
        "baseline_successes": 4,
        "candidate_successes": 4,
        "baseline_success_rate": 0.5,
        "candidate_success_rate": 0.5,
        "success_rate_delta_percentage_points": 0.0,
        "improved": 2,
        "regressed": 2,
        "both_succeeded": 2,
        "both_failed": 2,
        "mean_episode_return_delta": 1.25,
        "mean_episode_length_delta": 0.0,
    }
    row = result["per_seed"][0]["episodes"][1]
    assert row == {
        "env_id": 1,
        "baseline_success": True,
        "candidate_success": False,
        "return_delta": -1.0,
        "length_delta": 1,
    }
    assert (
        result["baseline"]["report_sha256"]
        == hashlib.sha256(reports[0].read_bytes()).hexdigest()
    )


def test_seed_order_does_not_change_pairing(reports):
    original = compare_reports(*reports)
    path = reports[1]
    data = json.loads(path.read_text())
    data["seeds"].reverse()
    data["per_seed"].reverse()
    path.write_text(json.dumps(data))
    reordered = compare_reports(*reports)
    assert original["summary"] == reordered["summary"]
    assert original["per_seed"] == reordered["per_seed"]
    assert (
        original["candidate"]["report_sha256"]
        != reordered["candidate"]["report_sha256"]
    )


@pytest.mark.parametrize(
    "case, message",
    [
        ("mode", "paired benchmark"),
        ("duplicate_seed", "unique"),
        ("missing_row", "exactly one"),
        ("duplicate_row", "duplicate"),
        ("legacy_config", "rerun benchmark"),
        ("environment", "configurations differ"),
        ("seed_set", "seed sets differ"),
        ("max_steps", "max_steps differ"),
        ("num_envs", "num_envs or max_steps differ"),
        ("initial_observation", "initial observations differ"),
        ("expert", "expert reference differs"),
        ("length", "step limit"),
        ("nan", "finite"),
        ("boolean", "booleans"),
        ("missing_episode", "one value per environment"),
    ],
)
def test_incompatible_or_invalid_reports_are_rejected(reports, case, message):
    path = reports[1]
    data = json.loads(path.read_text())
    row = data["per_seed"][0]
    if case == "mode":
        data["mode"] = "fixed_steps"
    elif case == "duplicate_seed":
        data["seeds"] = [2001, 2001]
    elif case == "missing_row":
        data["per_seed"].pop()
    elif case == "duplicate_row":
        data["per_seed"][1] = copy.deepcopy(row)
    elif case == "legacy_config":
        del row["environment_config"]
    elif case == "environment":
        row["environment_config"]["physics_hz"] = 400
    elif case == "seed_set":
        data["seeds"][0] = 2003
        row["seed"] = 2003
        row["environment_config"]["seed"] = 2003
    elif case == "max_steps":
        data["max_steps"] = 20
        for item in data["per_seed"]:
            item["environment_config"]["max_steps"] = 20
    elif case == "num_envs":
        data["num_envs"] = 3
        for item in data["per_seed"]:
            item["environment_config"]["num_envs"] = 3
            for policy in ("act", "expert"):
                for key in ("episode_success", "episode_return", "episode_length"):
                    item[policy][key].pop()
    elif case == "initial_observation":
        for name in ("act", "expert"):
            row[name]["initial_observation_sha256"] = "f" * 64
    elif case == "expert":
        row["expert"]["episode_return"][0] += 0.5
    elif case == "length":
        row["act"]["episode_length"][0] = 11
    elif case == "nan":
        row["act"]["episode_return"][0] = float("nan")
    elif case == "boolean":
        row["act"]["episode_success"][0] = 1
    elif case == "missing_episode":
        row["act"]["episode_success"].pop()
    path.write_text(json.dumps(data))
    output = path.parent / "comparison.json"
    with pytest.raises(ValueError, match=message):
        compare(Namespace(baseline=reports[0], candidate=path, output=output))
    assert not output.exists()


def test_cli_compares_without_sdk_and_preserves_reports(reports, tmp_path):
    before = [path.read_bytes() for path in reports]
    output = tmp_path / "comparison.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "embodiedforge",
            "act",
            "compare",
            "--baseline",
            str(reports[0]),
            "--candidate",
            str(reports[1]),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == json.loads(output.read_text())
    assert before == [path.read_bytes() for path in reports]
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        compare(Namespace(baseline=reports[0], candidate=reports[1], output=output))
    assert output.read_bytes() == saved
