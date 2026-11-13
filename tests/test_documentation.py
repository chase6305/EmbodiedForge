"""Repository documentation links survive moving reports into the archive."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = sorted((ROOT / "docs").rglob("*.md")) + [
    ROOT / "README.md",
    ROOT / "README.en.md",
]


@pytest.mark.parametrize(
    "document", DOCUMENTS, ids=lambda path: str(path.relative_to(ROOT))
)
def test_local_documentation_targets_exist(document):
    text = document.read_text()
    # Commands and example source strings are not Markdown navigation.
    text = re.sub(r"```.*?```|~~~.*?~~~", "", text, flags=re.S)
    for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        target = match[1].strip()
        if target.startswith("<"):
            target = target[1 : target.index(">")]
        else:
            target = target.split(' "', 1)[0]
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path or url.path.startswith("/"):
            continue
        resolved = (document.parent / unquote(url.path)).resolve()
        if resolved.is_relative_to(ROOT / "runs"):
            # Historical run artifacts are intentionally excluded from source releases.
            continue
        if (
            (ROOT / "PKG-INFO").is_file()
            and resolved.is_relative_to(ROOT / "docs/experiments/assets")
            and resolved.name.endswith(".tar.gz")
        ):
            # Source releases omit historical evidence blobs; repository links
            # remain checked above this packaged-source-only boundary.
            continue
        assert resolved.exists(), (
            f"{document.relative_to(ROOT)} links to missing {target}"
        )


def test_dated_experiment_files_are_separate_from_current_guides():
    dated = [
        path.name
        for path in (ROOT / "docs").iterdir()
        if path.is_file() and re.search(r"20\d{6}", path.name)
    ]
    assert not dated, f"Historical experiment files belong in docs/experiments: {dated}"
