# © Artur Czarnecki. All rights reserved.

"""ME-18-DOC-Q1 — Capability Marketplace canonical architecture documentation gates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = REPO_ROOT / "docs" / "project" / "architecture" / "CAPABILITY_MARKETPLACE_ENGINE.md"

_OBSOLETE_DOC_PHRASES = (
    "no dedicated recommendation SPI (ME-RB1-006)",
    "**Gap** — no dedicated recommendation",
)

_REQUIRED_CONTRACT_MARKERS = (
    "CapabilityCatalogSource",
    "MarketplaceMetadataSource",
    "MarketplaceListingProjection",
    "CapabilitySearchStrategy",
    "CapabilityRanker",
    "CapabilityGovernanceEvaluator",
    "CapabilityRecommendationStrategy",
    "MarketplaceDiagnosticObserver",
    "CapabilityCatalogSnapshotCache",
    "MarketplaceLifecycleHandoffHandler",
)


def _read_doc() -> str:
    assert DOC_PATH.is_file(), "canonical Capability Marketplace architecture doc must exist"
    return DOC_PATH.read_text(encoding="utf-8")


def test_canonical_marketplace_architecture_doc_exists() -> None:
    assert DOC_PATH.is_file()


def test_canonical_doc_contains_mermaid_architecture_diagram() -> None:
    text = _read_doc()
    assert "```mermaid" in text
    assert "flowchart" in text


def test_canonical_doc_describes_one_common_marketplace_for_verticals() -> None:
    text = _read_doc()
    assert "ONE COMMON CAPABILITY MARKETPLACE ENGINE" in text
    lowered = text.lower()
    assert "agent" in lowered and "tool" in lowered and "skill" in lowered
    assert "vertical" in lowered


def test_canonical_doc_describes_contract_variation_points() -> None:
    text = _read_doc()
    for marker in _REQUIRED_CONTRACT_MARKERS:
        assert marker in text, f"missing variation-point contract marker: {marker}"


def test_canonical_doc_describes_lifecycle_handoff_boundary() -> None:
    text = _read_doc()
    assert "MarketplaceLifecycleHandoff" in text or "lifecycle handoff" in text.lower()
    assert "HANDOFF_ACCEPTED" in text or "handoff accepted" in text.lower()


def test_canonical_doc_marketplace_is_not_execution_engine() -> None:
    text = _read_doc()
    lowered = text.lower()
    assert "not" in lowered and "execution" in lowered
    assert re.search(r"marketplace\s+.*execution", lowered) is not None or (
        "Machine API != Execution Engine" in text
    )


def test_canonical_doc_marketplace_is_not_trust_authority() -> None:
    text = _read_doc()
    lowered = text.lower()
    assert "trust authority" in lowered
    assert "not" in text and "trust authority" in text


def test_canonical_doc_governance_before_recommendation() -> None:
    text = _read_doc()
    assert re.search(
        r"governance\s+narrowing\s*→\s*recommendation",
        text,
        flags=re.IGNORECASE,
    ), "canonical pipeline must place governance narrowing before recommendation"


def test_canonical_doc_excludes_obsolete_me_rb1_recommendation_gap() -> None:
    text = _read_doc()
    for phrase in _OBSOLETE_DOC_PHRASES:
        assert phrase not in text, f"obsolete marketplace doc phrase still present: {phrase!r}"
