"""External interpreter entrypoint. Run with -I to avoid stdlib shadowing."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    # Add the package parent, never the directory containing logging.py.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from embodiedforge.recipes import sha256, write_json

    standalone = request.get("standalone", False)
    if standalone and request["task"] != "go1-joystick":
        raise ValueError("Standalone execution supports Go1 only")
    name = "wuji_unilab" if request["task"].startswith("wuji-") else "mjbatch"
    package = importlib.import_module(name)
    path = Path(package.__file__).resolve()
    if not standalone and not path.is_relative_to(Path(request["project"]) / "src"):
        raise ValueError(f"{name} loaded from unexpected source: {path}")
    required = {"mujoco": "3.11.0", "torch": "2.9.0+cu128", "mjbatch": "0.1.0"}
    if standalone:
        required["torch"] = "2.9.0"
    if name == "wuji_unilab":
        required = {
            "unilab": "1.2.0",
            "unisim-core": "1.2.0",
            "unilab-rl": "1.2.0",
            "mujoco": "3.11.0",
            "mujoco-warp": "3.11.0",
            "warp-lang": "1.16.0",
            "torch": "2.8.0+cu128",
            "rsl-rl-lib": "5.0.1",
        }
    versions = {key: importlib.metadata.version(key) for key in required}
    checked = dict(versions)
    if standalone:
        checked["torch"] = checked["torch"].split("+")[0]
    if checked != required:
        raise ValueError(
            f"Recipe dependency mismatch: expected {required}, got {versions}"
        )
    versions["numpy"] = importlib.metadata.version("numpy")
    if name == "mjbatch":
        versions["mujoco-menagerie"] = importlib.metadata.version("mujoco-menagerie")
        if versions["mujoco-menagerie"] != "2026.9.0":
            raise ValueError("Expected the pinned Menagerie 2026.9.0 assets registry")
    module = importlib.import_module(
        "embodiedforge._wuji_recipe"
        if name == "wuji_unilab"
        else "embodiedforge._mjbatch_recipe"
    )
    adapter_path = Path(module.__file__).resolve()
    if not adapter_path.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError(
            f"Native adapter loaded outside the run implementation: {adapter_path}"
        )
    write_json(
        Path("runtime.json"),
        {
            "python": sys.version,
            "executable": sys.executable,
            "source_import": str(path),
            "adapter_import": str(adapter_path),
            "launch_mode": "installed_packages" if standalone else "sdk_checkout",
            "versions": versions,
        },
    )
    result = module.run(request)
    result["runtime"] = json.loads(Path("runtime.json").read_text())
    if "checkpoint" in result:
        result["checkpoint_sha256"] = sha256(Path(result["checkpoint"]))
    write_json(Path("result.json"), result)


if __name__ == "__main__":
    main()
