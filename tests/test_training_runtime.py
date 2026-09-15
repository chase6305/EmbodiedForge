"""Provenance follows loaded objects, including external and native modules."""

import hashlib
import json
import math
from pathlib import Path

from embodiedforge._training_runtime import component, record_training_runtime


def test_component_records_the_loaded_source_and_native_module():
    result = component(record_training_runtime)
    assert result["symbol"] == "record_training_runtime"
    assert (
        result["python_source_sha256"]
        == hashlib.sha256(Path(result["file"]).read_bytes()).hexdigest()
    )
    assert result["project_namespace"]
    external = component(math)
    assert external["module"] == "math"
    assert external["python_source_sha256"] is None
    assert not external["project_namespace"]


def test_runtime_is_json_and_preserves_external_component_identity(tmp_path):
    result = record_training_runtime(
        tmp_path,
        environment=Path(),
        task=math,
        learner=record_training_runtime,
        physics_adapter=math,
        physics="test",
        core_vector_env=False,
        packages=["numpy", "embodiedforge-nonexistent-test-package"],
    )
    saved = json.loads((tmp_path / "training-runtime.json").read_text())
    assert saved == result
    assert not saved["environment"]["project_namespace"]
    assert saved["environment"]["module"] == "pathlib"
    assert saved["versions"]["numpy"]
    assert saved["versions"]["embodiedforge-nonexistent-test-package"] is None
    assert not (tmp_path / "training-runtime.json.tmp").exists()
