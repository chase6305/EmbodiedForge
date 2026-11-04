"""Metadata and process tools run independently of the NumPy environment stack."""

import json
import os
import subprocess
import sys
import textwrap

import pytest


def isolated(script, *arguments):
    return subprocess.run(
        [sys.executable, "-S", "-c", textwrap.dedent(script), *map(str, arguments)],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )


def test_runtime_tools_and_cli_import_without_site_packages(tmp_path):
    isolated(
        """
        import sys
        from pathlib import Path
        import embodiedforge
        from embodiedforge import cli
        from embodiedforge.runtime import artifacts, hashing, implementation, process, provenance, records

        assert embodiedforge.__version__ == '0.1.0'
        assert set(embodiedforge.__all__) <= set(dir(embodiedforge))
        assert 'VectorEnv' not in vars(embodiedforge)
        with records.managed_run(Path(sys.argv[1]), records.run_identity('probe', 'test')):
            process.run_process([sys.executable, '-S', '-c', 'pass'],
                                cwd=Path(sys.argv[1]), env=None)
        assert records.read_run(sys.argv[1])['status'] == 'complete'
        assert not any(name == 'numpy' or name.startswith('embodiedforge.env')
                       or name == 'embodiedforge.core' for name in sys.modules)
        """,
        tmp_path,
    )


def test_runs_inspection_cli_needs_only_standard_library_and_keeps_manifest(tmp_path):
    manifest = tmp_path / "run.json"
    original = json.dumps(
        {
            "run_contract": {
                "schema_version": 1,
                "workflow": "fixture",
                "operation": "train",
                "task": None,
            },
            "status": "complete",
        }
    ).encode()
    manifest.write_bytes(original)
    result = isolated(
        """
        import runpy, sys
        sys.argv = ['embodiedforge', 'runs', 'inspect', '--run', sys.argv[1]]
        runpy.run_module('embodiedforge', run_name='__main__')
        assert 'numpy' not in sys.modules
        assert 'embodiedforge.env' not in sys.modules
        """,
        tmp_path,
    )
    assert json.loads(result.stdout)["workflow"] == "fixture"
    assert manifest.read_bytes() == original


def test_public_exports_keep_identity_pickle_and_submodule_imports():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
                import pickle
                import embodiedforge
                assert 'VectorEnv' not in vars(embodiedforge)
                from embodiedforge import Config
                assert 'VectorEnv' not in vars(embodiedforge)
                from embodiedforge.core import Config as DefinedConfig
                assert Config is DefinedConfig
                assert pickle.loads(pickle.dumps(Config())) == Config()
                from embodiedforge import VectorEnv
                from embodiedforge.env import VectorEnv as DefinedEnvironment
                assert VectorEnv is DefinedEnvironment
                assert embodiedforge.VectorEnv is VectorEnv
                namespace = {}
                exec('from embodiedforge import *', namespace)
                for name in embodiedforge.__all__:
                    assert namespace[name] is getattr(embodiedforge, name)
                try:
                    embodiedforge.unknown_public_name
                except AttributeError as error:
                    assert 'unknown_public_name' in str(error)
                else:
                    raise AssertionError('Unknown export must raise AttributeError')
            """),
        ],
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "workflow", ["recipes", "h1", "rlinf", "microduck", "gmr", "sonic", "supermimic"]
)
def test_launcher_help_does_not_load_the_core_or_numpy(workflow):
    result = isolated(
        """
        import runpy, sys
        sys.argv = ['embodiedforge', sys.argv[1], '--help']
        try:
            runpy.run_module('embodiedforge', run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('help must exit normally')
        assert 'numpy' not in sys.modules
        assert 'embodiedforge.env' not in sys.modules
        """,
        workflow,
    )
    assert "usage:" in result.stdout


@pytest.mark.parametrize(
    "workflow",
    [
        "runs",
        "microduck",
        "gmr",
        "sonic",
        "supermimic",
        "rlinf",
        "weave",
        "motion",
        "act",
        "h1-native",
        "live",
        "replay",
        "recipes",
        "h1",
    ],
)
def test_prefixed_core_options_never_run_a_different_environment(
    monkeypatch, capsys, workflow
):
    from embodiedforge import cli, env

    def unexpected_environment(*args, **kwargs):
        pytest.fail("A misordered SDK command must not start the core environment")

    monkeypatch.setattr(env, "VectorEnv", unexpected_environment)
    with pytest.raises(SystemExit) as error:
        cli.main(["--steps", "1", workflow])
    assert error.value.code == 2
    assert "must be the first argument" in capsys.readouterr().err
