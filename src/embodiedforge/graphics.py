"""Explicit EGL rendering diagnostics, isolated from the caller's SDK state."""

import ctypes.util
import json
import os
import subprocess
import sys
from pathlib import Path


def egl_vendor_report() -> dict:
    """Inspect GLVND discovery inputs without changing the driver configuration."""
    explicit_files = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES")
    explicit_dirs = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
    if explicit_files is not None:
        paths = [Path(value) for value in explicit_files.split(os.pathsep) if value]
    else:
        directories = (
            explicit_dirs.split(os.pathsep)
            if explicit_dirs is not None
            else ["/etc/glvnd/egl_vendor.d", "/usr/share/glvnd/egl_vendor.d"]
        )
        paths = sorted(
            {
                path
                for directory in directories
                if directory
                for path in Path(directory).glob("*.json")
            }
        )
    files = []
    for path in paths:
        item = {"path": str(path)}
        try:
            data = json.loads(path.read_text())
            library = data["ICD"]["library_path"]
            if not isinstance(library, str) or not library:
                raise ValueError("ICD.library_path must be a string")
            item["library_path"] = library
        except (OSError, ValueError, KeyError, TypeError) as error:
            item["error"] = str(error)
        files.append(item)
    nvidia_library = ctypes.util.find_library("EGL_nvidia")
    warnings = []
    if nvidia_library and not any(
        "nvidia" in item.get("library_path", "").lower() for item in files
    ):
        warnings.append(
            "NVIDIA EGL library is installed but no NVIDIA vendor registration "
            "was found in the selected GLVND paths. Check driver package files "
            "and EGL vendor environment overrides."
        )
    return {
        "vendor_files": files,
        "nvidia_library": nvidia_library,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
                "PYOPENGL_PLATFORM",
                "__EGL_VENDOR_LIBRARY_FILENAMES",
                "__EGL_VENDOR_LIBRARY_DIRS",
            )
        },
        "warnings": warnings,
    }


def egl_render_report(timeout: float = 30) -> dict:
    """Render a tiny MuJoCo scene in a child; report actual GL vendor and errors.

    The requested probe selects EGL only in the child. Vendor selection overrides
    remain intact. A successful frame does not imply NVIDIA or RTX was selected.
    """
    report = {"check_scope": "egl_context_and_framebuffer", **egl_vendor_report()}
    environment = {**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    try:
        result = subprocess.run(
            [sys.executable, "-m", "embodiedforge.graphics"],
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {**report, "ok": False, "error": f"EGL probe timed out after {timeout}s"}
    except OSError as error:
        return {**report, "ok": False, "error": str(error)}
    report["returncode"] = result.returncode
    report["stderr"] = result.stderr[-8192:]
    if result.returncode:
        return {**report, "ok": False, "error": "EGL probe failed; inspect stderr"}
    try:
        frame = json.loads(result.stdout)
        if not isinstance(frame, dict) or not all(
            isinstance(frame.get(key), str) and frame[key]
            for key in ("vendor", "renderer", "gl_version")
        ):
            raise ValueError("Missing GL device information")
        report.update(frame)
    except (ValueError, TypeError) as error:
        return {**report, "ok": False, "error": f"Invalid EGL probe output: {error}"}
    return {**report, "ok": True}


def _render_probe() -> dict:
    import mujoco
    import numpy as np
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, GL_VERSION, glGetString

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <light pos="0 0 3"/>
      <geom type="plane" size="2 2 .1" rgba=".2 .3 .4 1"/>
      <geom type="sphere" pos="0 0 .2" size=".2" rgba=".8 .2 .1 1"/>
    </worldbody></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=64, width=64)
    try:
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, 0.2]
        camera.distance, camera.azimuth, camera.elevation = 2, 90, -30
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        if image.shape != (64, 64, 3) or not np.ptp(image):
            raise RuntimeError("EGL probe produced an invalid or uniform frame")
        return {
            "vendor": glGetString(GL_VENDOR).decode(),
            "renderer": glGetString(GL_RENDERER).decode(),
            "gl_version": glGetString(GL_VERSION).decode(),
            "image_shape": list(image.shape),
            "image_range": [int(image.min()), int(image.max())],
        }
    finally:
        renderer.close()


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    print(json.dumps(_render_probe()))
