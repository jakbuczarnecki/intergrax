# © Artur Czarnecki. All rights reserved.

"""GOV-FINAL-4 — Governance E2E qualification documentation regression gates."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
QUAL_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "GOVERNANCE_FINAL_E2E_QUALIFICATION.md"
)
ARCH_GOVERNED = _REPO_ROOT / "docs" / "project" / "architecture" / "GOVERNED_EXECUTION.md"
_CATALOG = _REPO_ROOT / "tests" / "qualification" / "governance" / "catalog.py"

_FORBIDDEN_FULL_CLAIMS = (
    "Full Governance Plane enterprise certified: YES",
    "FULL GOVERNANCE ENTERPRISE CERTIFIED",
    "Governance Layer enterprise certified",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _head_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()


def test_gov_final_4_qualification_artifact_exists() -> None:
    assert QUAL_DOC.is_file()


def test_gov_final_4_qualification_doc_has_required_sections() -> None:
    text = _read(QUAL_DOC)
    for heading in (
        "## Baseline",
        "## Scenario matrix",
        "## Strategy qualification matrix",
        "## Failure matrix",
        "## Pluginability matrix",
        "## Remaining gaps",
        "## Claim boundary",
        "## Test commands",
    ):
        assert heading in text
    assert re.search(r"baseline SHA:\s*`[0-9a-f]{40}`", text, re.IGNORECASE)


def test_gov_final_4_qualification_doc_baseline_is_contained_in_history() -> None:
    doc = _read(QUAL_DOC)
    match = re.search(r"baseline SHA:\s*`([0-9a-f]{40})`", doc, re.IGNORECASE)
    assert match, "qualification doc must record baseline SHA"
    baseline = match.group(1)
    head = _head_sha()
    if baseline == head:
        return
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, head],
        cwd=_REPO_ROOT,
        check=True,
    )


def test_gov_final_4_catalog_module_present() -> None:
    assert _CATALOG.is_file()
    assert "GOV_FINAL_4_SCENARIO_CATALOG" in _read(_CATALOG)


def test_gov_final_4_architecture_pointer_without_full_certification_claim() -> None:
    arch = _read(ARCH_GOVERNED)
    assert "GOV-FINAL-4" in arch
    assert "GOVERNANCE_FINAL_E2E_QUALIFICATION.md" in arch
    for phrase in _FORBIDDEN_FULL_CLAIMS:
        assert phrase not in arch


def test_gov_final_4_no_false_full_enterprise_claim_in_qualification_doc() -> None:
    text = _read(QUAL_DOC)
    claim_section = text.split("## Claim boundary", 1)[-1]
    assert "Full Governance Plane enterprise certified: NO" in claim_section
    for phrase in _FORBIDDEN_FULL_CLAIMS:
        assert phrase not in text
