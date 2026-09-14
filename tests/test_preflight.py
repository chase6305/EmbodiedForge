"""Startup rejection must happen before task allocation or SDK construction."""

import json
import subprocess
import sys
from dataclasses import replace
from importlib import metadata

import pytest

from embodiedforge import Config, VectorEnv
from embodiedforge.backends import PHYSICS, RENDER, Registration, execution_plan
from embodiedforge.core import Capabilities, SceneSpec
from embodiedforge.diagnostics import installation_report
from embodiedforge.tasks import ReachTask


@pytest.mark.parametrize("change", [{"action_dim": 3}, {"action_units": "radians"}])
def test_control_mismatch_precedes_allocation(change):
    class IncompatibleTask(ReachTask):
        spec = replace(ReachTask.spec, **change)

        def build(self, count):
            pytest.fail("incompatible task must not allocate")

    with pytest.raises(ValueError, match="control mismatch"):
        VectorEnv(task=IncompatibleTask())


def test_undeclared_control_rejected(monkeypatch):
    def factory():
        pytest.fail("undeclared backend must not be constructed")

    monkeypatch.setitem(PHYSICS, "undeclared", Registration(factory, Capabilities()))
    with pytest.raises(ValueError, match="control mismatch"):
        execution_plan(Config(physics="undeclared"))


def test_unknown_pipeline_channel_rejected_even_if_renderer_advertises_it(monkeypatch):
    monkeypatch.setitem(
        RENDER,
        "normal",
        Registration(
            lambda: pytest.fail("renderer must not be constructed"),
            Capabilities(channels=frozenset({"normal"})),
        ),
    )
    with pytest.raises(ValueError, match="Sensor pipeline"):
        execution_plan(Config(render="normal", channels=("normal",)))


def test_required_renderer_entities_checked_before_task_build():
    class MissingTarget(ReachTask):
        scene = SceneSpec(entity_ids=("agent",))

        def build(self, count):
            pytest.fail("invalid scene must not allocate")

    with pytest.raises(ValueError, match="requires scene entities"):
        VectorEnv(Config(render="raster"), task=MissingTarget())


@pytest.mark.parametrize(
    "newton,warp,ok",
    [
        ("1.6.0rc1", "1.17.0", True),
        ("1.6.0", "1.17.0", False),
        (None, "1.17.0", False),
        ("1.6.0rc1", None, False),
    ],
)
def test_diagnostics_exact_version_and_missing_dependency(
    monkeypatch, newton, warp, ok
):
    versions = {"numpy": "2.2.6", "newton": newton, "warp-lang": warp}

    def version(name):
        if versions.get(name) is None:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(metadata, "version", version)
    report = installation_report(Config(physics="newton"))
    assert report["metadata_ok"] is ok
    assert report["packages"][-1]["name"] == "torch"
    assert report["packages"][-1]["required_for_simulation"] is False


def test_diagnostics_does_not_import_sdks():
    script = """
import json, sys
from embodiedforge import Config
from embodiedforge.diagnostics import installation_report
print(json.dumps(installation_report(Config(physics="newton"))))
assert not {"newton", "warp", "mujoco", "torch"}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout)["check_scope"] == "installation_metadata_only"


def test_doctor_cli_exit_status_matches_report():
    result = subprocess.run(
        [sys.executable, "-m", "embodiedforge", "doctor", "--physics", "numpy"],
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert result.returncode == (0 if report["metadata_ok"] else 1)
