"""Checkpoint comparisons must not silently mix different evaluation code."""

import json

import pytest

from embodiedforge import microduck
from embodiedforge._microduck_reports import compare_sources

BASELINE = {"revision": "upstream-revision", "uv_lock_sha256": "upstream-lock"}
NATIVE = {
    **BASELINE,
    "kind": "repository_owned",
    "implementation": "embodiedforge.locomotion.microduck",
    "implementation_sha256": "a" * 64,
    "requirements_sha256": "b" * 64,
}


def test_equal_native_implementations_record_verified_fields():
    before, after = {**NATIVE, "repo": "/old"}, {**NATIVE, "repo": "/new"}
    result = compare_sources(before, after)
    assert result["level"] == "local_implementation_and_requirements"
    assert result["fields"]["implementation_sha256"] == NATIVE["implementation_sha256"]
    assert result["fields"]["requirements_sha256"] == NATIVE["requirements_sha256"]
    assert "repo" not in result["fields"]


@pytest.mark.parametrize(
    "key",
    [
        "revision",
        "uv_lock_sha256",
        "kind",
        "implementation",
        "implementation_sha256",
        "requirements_sha256",
    ],
)
@pytest.mark.parametrize("value", [None, "different"])
def test_changed_or_missing_identity_is_rejected(key, value):
    after = {**NATIVE, key: value}
    with pytest.raises(ValueError, match="Comparison"):
        compare_sources(NATIVE, after)


@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, 123, None])
def test_equal_invalid_hashes_do_not_claim_verification(value):
    source = {**NATIVE, "implementation_sha256": value}
    with pytest.raises(ValueError, match="implementation_sha256"):
        compare_sources(source, source)


def test_legacy_comparison_is_explicitly_limited_to_upstream_baseline():
    result = compare_sources(BASELINE, BASELINE)
    assert result == {"level": "upstream_baseline_only", "fields": BASELINE}
    for before, after in ((BASELINE, NATIVE), (NATIVE, BASELINE)):
        with pytest.raises(ValueError, match="implementation kinds"):
            compare_sources(before, after)


def test_missing_native_kind_cannot_downgrade_to_legacy_verification():
    source = {key: value for key, value in NATIVE.items() if key != "kind"}
    with pytest.raises(ValueError, match="missing its implementation kind"):
        compare_sources(source, source)


def test_comparison_rejects_changed_local_source_before_creating_output(
    tmp_path, monkeypatch
):
    from embodiedforge import _microduck_reports as reports

    after = {**NATIVE, "implementation_sha256": "c" * 64}
    monkeypatch.setattr(
        reports,
        "load_evaluation_run",
        lambda path: ([], {"source": NATIVE if path.name == "before" else after}, {}),
    )
    monkeypatch.setattr(
        reports,
        "compare_evaluations",
        lambda *a: pytest.fail("incomparable reports reached metric comparison"),
    )
    output = tmp_path / "comparison"
    with pytest.raises(SystemExit) as exc:
        microduck.main(
            [
                "compare",
                "--before",
                str(tmp_path / "before"),
                "--after",
                str(tmp_path / "after"),
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == 1 and not output.exists()


def test_success_records_source_check_in_both_outputs(tmp_path, monkeypatch):
    from embodiedforge import _microduck_reports as reports

    monkeypatch.setattr(
        reports, "load_evaluation_run", lambda path: ([], {"source": NATIVE}, {})
    )
    monkeypatch.setattr(reports, "compare_evaluations", lambda *a: {"metrics": {}})
    output = tmp_path / "comparison"
    microduck.main(
        [
            "compare",
            "--before",
            str(tmp_path / "before"),
            "--after",
            str(tmp_path / "after"),
            "--output",
            str(output),
        ]
    )
    manifest = json.loads((output / "run.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    assert manifest["status"] == "complete"
    assert (
        manifest["source_verification"]
        == comparison["source_verification"]
        == compare_sources(NATIVE, NATIVE)
    )
