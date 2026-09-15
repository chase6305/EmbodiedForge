"""Asset verification must apply to the directory actually used by the task."""

import sys
from types import SimpleNamespace

import pytest

from embodiedforge._go1_assets import reuse_asset_cache, verified_go1_assets


@pytest.fixture
def registry(monkeypatch, tmp_path):
    calls = []
    cache = SimpleNamespace(dir=tmp_path, root=None)
    robot = SimpleNamespace(name="unitree_go1", oid="tree", sha256="archive")

    def resolve(*, cache):
        calls.append(("resolve", cache.dir))
        return cache.dir / "models" / "go1"

    def verify(robot):
        calls.append(("verify", robot.name))
        return []

    robot.path = resolve
    cache.verify = verify
    monkeypatch.setitem(
        sys.modules,
        "mujoco_menagerie",
        SimpleNamespace(
            get=lambda name: robot,
            Cache=lambda: cache,
        ),
    )
    return cache, calls


def test_verifies_the_resolved_cache_and_records_its_identity(registry):
    cache, calls = registry
    result = verified_go1_assets()
    assert result["robot"] == "unitree_go1"
    assert result["tree_id"] == "tree"
    assert result["cache_dir"] == str(cache.dir)
    assert calls == [("resolve", cache.dir), ("verify", "unitree_go1")]
    assert verified_go1_assets(result) == result


@pytest.mark.parametrize(
    "change", [{"tree_id": "wrong"}, {"archive_sha256": "wrong"}, {}]
)
def test_changed_training_identity_is_rejected_before_loading(registry, change):
    _, calls = registry
    with pytest.raises(ValueError, match="identity"):
        verified_go1_assets(change)
    assert not calls


def test_local_root_cannot_bypass_verification_of_the_published_assets(registry):
    cache, calls = registry
    cache.root = cache.dir / "custom-checkout"
    with pytest.raises(ValueError, match="MENAGERIE_ROOT"):
        verified_go1_assets()
    assert not calls


def test_changed_cache_files_are_rejected(registry):
    cache, _ = registry
    cache.verify = lambda robot: ["go1.xml"]
    with pytest.raises(ValueError, match="recorded hashes"):
        verified_go1_assets()


def test_reuses_existing_cache_without_overriding_user_choice(tmp_path):
    assets = {"cache_dir": str(tmp_path)}
    environment = {}
    reuse_asset_cache(environment, assets)
    assert environment["MENAGERIE_CACHE_DIR"] == str(tmp_path)
    empty = {"MENAGERIE_CACHE_DIR": ""}
    reuse_asset_cache(empty, assets)
    assert empty == environment
    reuse_asset_cache(environment, {"cache_dir": "/other"})
    assert environment["MENAGERIE_CACHE_DIR"] == str(tmp_path)
    for value in (
        None,
        {},
        {"cache_dir": str(tmp_path / "missing")},
        {"cache_dir": "relative"},
    ):
        environment = {}
        reuse_asset_cache(environment, value)
        assert not environment


def test_legacy_asset_record_reuses_only_the_published_directory_layout(tmp_path):
    import json

    from embodiedforge._go1_assets import recorded_go1_assets

    model = tmp_path / "cache" / "models" / "unitree_go1-0123456789abcdef"
    model.mkdir(parents=True)
    assets = {
        "robot": "unitree_go1",
        "tree_id": "0123456789abcdef0000",
        "archive_sha256": "hash",
        "path": str(model),
    }
    (tmp_path / "assets.json").write_text(json.dumps(assets))
    record = recorded_go1_assets(tmp_path, {})
    assert record["cache_dir"] == str(tmp_path / "cache")
    assets["path"] = str(tmp_path / "custom" / "unitree_go1")
    (tmp_path / "assets.json").write_text(json.dumps(assets))
    assert "cache_dir" not in recorded_go1_assets(tmp_path, {})
    assert recorded_go1_assets(tmp_path / "missing", {}) is None
