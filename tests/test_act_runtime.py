"""ACT provenance inspects the requested SDK entry points for each operation."""

import importlib.abc
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from embodiedforge import Config, VectorEnv, act


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    """Install only fake SDK entries, retaining real source-file provenance checks."""
    for name in tuple(sys.modules):
        if name == "lerobot" or name.startswith("lerobot."):
            monkeypatch.delitem(sys.modules, name)

    def module(name, source="", *, package=False):
        path = tmp_path.joinpath(*name.split("."))
        path = path / "__init__.py" if package else path.with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        value = ModuleType(name)
        value.__file__ = str(path)
        if package:
            value.__path__ = [str(path.parent)]
        exec(compile(source, str(path), "exec"), value.__dict__)
        monkeypatch.setitem(sys.modules, name, value)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(sys.modules[parent], child, value)
        return value

    for name in ("lerobot", "lerobot.policies", "lerobot.policies.act"):
        module(name, package=True)
    module("lerobot.policies.act.modeling_act", "class ACTPolicy: pass\n")
    module("lerobot.policies.factory", "def make_pre_post_processors(): pass\n")
    monkeypatch.setattr(
        act,
        "distribution",
        lambda name: SimpleNamespace(
            version="fixture",
            read_text=lambda filename: None,
            locate_file=lambda filename: tmp_path / filename,
        ),
    )

    def configure(*, training):
        if training:
            module("lerobot.datasets", package=True)
            module("lerobot.datasets.lerobot_dataset", "class LeRobotDataset: pass\n")
            module("lerobot.scripts", package=True)
            module("lerobot.scripts.lerobot_train")
        else:

            class BlockTraining(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.startswith(("lerobot.datasets", "lerobot.scripts")):
                        raise AssertionError(
                            "Unneeded training dependency: " + fullname
                        )

            monkeypatch.setattr(sys, "meta_path", [BlockTraining(), *sys.meta_path])

    return configure


@pytest.mark.parametrize("training", [True, False])
def test_runtime_report_checks_only_requested_sdk_entries(sdk, training):
    sdk(training=training)
    # The zero-argument public API still checks all doctor/training entries.
    report = act.runtime_report() if training else act.runtime_report(training=False)
    assert report["schema_version"] == 1
    assert report["requested_scope"] == ("training" if training else "inference")
    assert report["scope"] == "imported_entry_points_not_full_dependency_graph"
    assert set(report["components"]) == (
        {"policy", "trainer", "dataset", "processors"}
        if training
        else {"policy", "processors"}
    )
    assert report["installation"]["mode"] == "installed_package"
    assert report["adapter"]["project_namespace"]
    for entry in report["components"].values():
        assert len(entry["python_source_sha256"]) == 64


@pytest.mark.parametrize("operation", ["evaluate", "benchmark"])
def test_closed_loop_reports_request_inference_sdk_entries(
    tmp_path, monkeypatch, sdk, operation
):
    sdk(training=False)
    config = Config(num_envs=1, max_steps=2, channels=(), render="null")
    with VectorEnv(config) as env:
        data = {
            "config": config.to_dict(),
            "env_spec": json.loads(json.dumps(env.spec)),
        }

    class Policy:
        horizon = 1
        checkpoint_sha256 = "0" * 64
        checkpoint_fingerprint = {"sha256": "fixture"}
        clipped_values = 0
        total_values = 0

        def act(self, observation):
            return np.zeros((len(observation["proprio"]), 1, 2), dtype=np.float32)

        def inference_settings(self):
            return {"device": "fixture"}

    monkeypatch.setattr(
        act, "_evaluation_policy", lambda args: (data, tmp_path / "model", {}, Policy())
    )
    args = SimpleNamespace(num_envs=1, seed=1, steps=2, seeds=[1])
    report = getattr(act, operation)(args)
    assert report["runtime"]["requested_scope"] == "inference"
    assert set(report["runtime"]["components"]) == {"policy", "processors"}
    assert "lerobot.datasets" not in sys.modules
    assert "lerobot.scripts" not in sys.modules
