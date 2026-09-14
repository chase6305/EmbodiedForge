"""Device diagnostics must retain warnings and contain native driver failures."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from embodiedforge import graphics


def test_missing_nvidia_registration_and_explicit_vendor_selection(
    tmp_path, monkeypatch
):
    mesa = tmp_path / "mesa.json"
    nvidia = tmp_path / "nvidia.json"
    for path, library in ((mesa, "libEGL_mesa.so.0"), (nvidia, "libEGL_nvidia.so.0")):
        path.write_text(json.dumps({"ICD": {"library_path": library}}))
    monkeypatch.setattr(
        graphics.ctypes.util, "find_library", lambda name: "libEGL_nvidia.so.0"
    )
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", str(mesa))
    report = graphics.egl_vendor_report()
    assert len(report["warnings"]) == 1
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", str(nvidia))
    report = graphics.egl_vendor_report()
    assert report["warnings"] == []
    assert report["vendor_files"] == [
        {"path": str(nvidia), "library_path": "libEGL_nvidia.so.0"}
    ]


@pytest.mark.parametrize(
    "content",
    ["not json", '{"ICD":{"library_path":null}}', '{"ICD":{"library_path":[]}}'],
)
def test_vendor_directory_override_and_invalid_registration(
    tmp_path, monkeypatch, content
):
    invalid = tmp_path / "10-broken.json"
    invalid.write_text(content)
    monkeypatch.delenv("__EGL_VENDOR_LIBRARY_FILENAMES", raising=False)
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_DIRS", str(tmp_path))
    monkeypatch.setattr(
        graphics.ctypes.util, "find_library", lambda name: "libEGL_nvidia.so.0"
    )
    report = graphics.egl_vendor_report()
    assert len(report["vendor_files"]) == 1
    assert report["vendor_files"][0]["error"]


@pytest.mark.parametrize("mode", ["timeout", "crash", "malformed", "missing-device"])
def test_probe_failure_is_structured_without_crashing_parent(monkeypatch, mode):
    monkeypatch.setattr(graphics, "egl_vendor_report", lambda: {})

    def run(*args, **kwargs):
        if mode == "timeout":
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return SimpleNamespace(
            returncode=-11 if mode == "crash" else 0,
            stdout="not json" if mode == "malformed" else "{}",
            stderr="driver warning",
        )

    monkeypatch.setattr(graphics.subprocess, "run", run)
    report = graphics.egl_render_report(timeout=0.1)
    assert not report["ok"] and report["error"]
    if mode != "timeout":
        assert report["stderr"] == "driver warning"


def test_success_reports_actual_device_without_mutating_caller_environment(monkeypatch):
    monkeypatch.setattr(graphics, "egl_vendor_report", lambda: {})
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", "/custom/mesa.json")

    def run(*args, **kwargs):
        assert kwargs["env"]["MUJOCO_GL"] == "egl"
        assert kwargs["env"]["__EGL_VENDOR_LIBRARY_FILENAMES"] == "/custom/mesa.json"
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "vendor": "AMD",
                    "renderer": "integrated",
                    "gl_version": "4.6",
                    "image_shape": [64, 64, 3],
                    "image_range": [0, 200],
                }
            ),
            stderr="warning retained",
        )

    monkeypatch.setattr(graphics.subprocess, "run", run)
    report = graphics.egl_render_report()
    assert report["ok"] and report["vendor"] == "AMD"
    assert report["stderr"] == "warning retained"
    assert graphics.os.environ["MUJOCO_GL"] == "glfw"


def test_import_diagnostics_does_not_load_graphics_sdks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from embodiedforge.graphics import egl_vendor_report, egl_render_report
assert not {'mujoco', 'OpenGL', 'warp', 'torch'}.intersection(sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_graphics_option_rejected_outside_doctor():
    result = subprocess.run(
        [sys.executable, "-m", "embodiedforge", "plan", "--graphics", "egl"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "only supported by doctor" in result.stderr


@pytest.mark.parametrize("ok", [True, False])
def test_doctor_graphics_exit_status_and_scope(monkeypatch, capsys, ok):
    from embodiedforge.cli import main

    monkeypatch.setattr(graphics, "egl_render_report", lambda: {"ok": ok})
    monkeypatch.setattr(sys, "argv", ["embodiedforge", "doctor", "--graphics", "egl"])
    with pytest.raises(SystemExit) as exit_status:
        main()
    assert exit_status.value.code == (0 if ok else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["check_scope"] == "installation_metadata_and_egl_framebuffer"
    assert report["graphics"]["ok"] is ok
