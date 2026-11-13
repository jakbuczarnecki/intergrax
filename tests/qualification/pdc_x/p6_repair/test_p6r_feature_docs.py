# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 7 - the ARTIFACT_LIFECYCLE feature pair is registered and counted by the docs-pair checker."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
FEATURE = "ARTIFACT_LIFECYCLE"
README = REPO_ROOT / "docs/project/capabilities/README.md"
HUB = REPO_ROOT / "docs/project/architecture/intergrax_runtime_architecture.md"


def test_the_feature_pair_files_exist() -> None:
    assert (REPO_ROOT / f"docs/project/capabilities/architecture/{FEATURE}.md").is_file()
    assert (REPO_ROOT / f"docs/project/capabilities/plan/{FEATURE}.md").is_file()


def test_capabilities_readme_has_a_canonical_index_row_linking_both_docs() -> None:
    text = README.read_text(encoding="utf-8")
    start = text.index("## Current multi-layer features")
    section = text[start : text.index("**Satellites", start)]
    row = next(line for line in section.splitlines() if line.startswith(f"| `{FEATURE}` |"))
    assert f"architecture/{FEATURE}.md" in row
    assert f"plan/{FEATURE}.md" in row


def test_runtime_hub_lists_the_feature_pair() -> None:
    text = HUB.read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith(f"| `{FEATURE}` |"))
    assert f"../capabilities/architecture/{FEATURE}.md" in row
    assert f"../capabilities/plan/{FEATURE}.md" in row
    assert re.search(rf"\| \*\*FEATURE\*\* .*`{FEATURE}`", text)


def test_the_docs_pair_checker_counts_the_feature() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/docs/check_docs_domain_pairs.py")],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check_docs_domain_pairs: OK" in result.stdout
    count = int(re.search(r"(\d+) feature pairs", result.stdout).group(1))  # type: ignore[union-attr]
    index = README.read_text(encoding="utf-8")
    assert f"`{FEATURE}`" in index
    assert count >= 5
