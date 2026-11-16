# © Artur Czarnecki. All rights reserved.

"""Acceptance: the ARTIFACT_LIFECYCLE feature doc pair is published and names the decisions."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
ARCH = REPO_ROOT / "docs/project/capabilities/architecture/ARTIFACT_LIFECYCLE.md"
PLAN = REPO_ROOT / "docs/project/capabilities/plan/ARTIFACT_LIFECYCLE.md"


def test_feature_doc_pair_exists() -> None:
    assert ARCH.is_file() and PLAN.is_file()


@pytest.mark.parametrize(
    "needle",
    [
        "ADR-PDC-08",
        "Collaborative Work",
        "UCL",
        "tool runtime",
        "governed access port",
        "no global",
        "legal hold",
        "OD-PDC-SCOPE-04",
    ],
)
def test_architecture_doc_states_the_decisions(needle: str) -> None:
    assert needle.lower() in ARCH.read_text(encoding="utf-8").lower()


def test_pair_links_to_each_other_and_is_a_feature_not_a_domain() -> None:
    arch = ARCH.read_text(encoding="utf-8")
    plan = PLAN.read_text(encoding="utf-8")
    assert "plan/ARTIFACT_LIFECYCLE.md" in arch
    assert "architecture/ARTIFACT_LIFECYCLE.md" in plan
    assert "feature" in arch.lower() and "not a new domain" in arch.lower()
    for requirement in ("PDC-D8-R001", "PDC-D8-R018", "PDC-D6-R022"):
        assert requirement in plan
