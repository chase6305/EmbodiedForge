"""Verify the published Go1 assets consistently for training and live policies."""

import json
from pathlib import Path


def recorded_go1_assets(directory, result):
    """Read new inline records or the assets.json emitted by older Go1 runs."""
    assets = result.get("assets")
    path = Path(directory) / "assets.json"
    if assets is None and path.is_file():
        assets = json.loads(path.read_text())
    if assets is None:
        return None
    if not isinstance(assets, dict):
        raise ValueError("Invalid Go1 asset record")
    assets = dict(assets)
    # Older records stored only the model path. Infer a cache only for the
    # published Menagerie directory layout, never an arbitrary local checkout.
    model_path = Path(assets.get("path", ""))
    robot, tree_id = assets.get("robot"), assets.get("tree_id")
    if (
        "cache_dir" not in assets
        and isinstance(robot, str)
        and isinstance(tree_id, str)
        and model_path.is_absolute()
        and model_path.parent.name == "models"
        and model_path.name == f"{robot}-{tree_id[:16]}"
    ):
        assets["cache_dir"] = str(model_path.parent.parent)
    return assets


def verified_go1_assets(expected=None):
    import mujoco_menagerie as menagerie

    robot = menagerie.get("unitree_go1")
    identity = {
        "robot": robot.name,
        "tree_id": robot.oid,
        "archive_sha256": robot.sha256,
    }
    if expected is not None and (
        not isinstance(expected, dict)
        or any(expected.get(key) != value for key, value in identity.items())
    ):
        raise ValueError("Go1 asset identity differs from the training run")
    cache = menagerie.Cache()
    # Cache.verify checks the published cache, not MENAGERIE_ROOT. Loading a
    # separate checkout while verifying the cache would verify the wrong files.
    if cache.root is not None:
        raise ValueError(
            "Verified Go1 runs do not support MENAGERIE_ROOT; unset it and use "
            "MENAGERIE_CACHE_DIR to select the published asset cache"
        )
    path = robot.path(cache=cache)
    if cache.verify(robot):
        raise ValueError("Go1 asset cache differs from its recorded hashes")
    return {
        **identity,
        "path": str(Path(path).resolve()),
        "cache_dir": str(cache.dir.resolve()),
    }


def reuse_asset_cache(environment, assets):
    """Keep explicit user choices; only reuse a recorded cache that still exists."""
    if not environment.get("MENAGERIE_CACHE_DIR") and isinstance(assets, dict):
        directory = assets.get("cache_dir")
        if (
            isinstance(directory, str)
            and Path(directory).is_absolute()
            and Path(directory).is_dir()
        ):
            environment["MENAGERIE_CACHE_DIR"] = directory
