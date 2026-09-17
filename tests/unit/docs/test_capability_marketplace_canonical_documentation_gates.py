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

_PLUGIN_VARIATION_POINT_MARKERS = (
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

_MERMAID_ARCHITECTURE_MARKERS = (
    "CapabilityCatalogSource",
    "FederatedCapabilityCatalog",
    "MarketplaceCatalogService",
    "Visibility filtering",
    "CapabilitySearchStrategy",
    "CapabilityRanker",
    "CapabilityGovernanceEvaluator",
    "CapabilityRecommendationStrategy",
    "Explicit selection",
    "Typed lifecycle handoff",
    "Agent Distribution",
    "Tool domain",
    "Skill domain",
    "Execution Engine",
)

_SECTION_VISUAL = "### 3.1 Canonical end-to-end architecture (visual)"
_SECTION_PLUGIN = "## 10. Plugin architecture"
_SECTION_HANDOFF = "## 11. Lifecycle handoff (ME-RB4)"


def _read_doc() -> str:
    assert DOC_PATH.is_file(), "canonical Capability Marketplace architecture doc must exist"
    return DOC_PATH.read_text(encoding="utf-8")


def _heading_level(heading_line: str) -> int:
    stripped = heading_line.lstrip()
    if not stripped.startswith("#"):
        return 0
    level = 0
    for ch in stripped:
        if ch == "#":
            level += 1
        else:
            break
    return level


def _section(text: str, heading_line: str) -> str:
    """Body from ``heading_line`` until the next Markdown heading of the same or higher level."""
    pos = text.find(heading_line)
    assert pos != -1, f"missing section heading: {heading_line!r}"
    body_start = pos + len(heading_line)
    if body_start < len(text) and text[body_start] == "\n":
        body_start += 1
    level = _heading_level(heading_line)
    scan = body_start
    while scan < len(text):
        if text[scan] != "\n":
            scan += 1
            continue
        line_start = scan + 1
        if line_start >= len(text) or text[line_start] != "#":
            scan += 1
            continue
        next_level = _heading_level(text[line_start : text.find("\n", line_start)])
        if next_level > 0 and next_level <= level:
            return text[body_start:scan]
        scan += 1
    return text[body_start:]


def _doc_preamble(text: str) -> str:
    """Introductory canon before the first ``##`` section (one common engine model)."""
    marker = "\n## "
    pos = text.find(marker)
    assert pos != -1, "canonical doc must contain at least one ## section"
    return text[:pos]


def test_canonical_marketplace_architecture_doc_exists() -> None:
    assert DOC_PATH.is_file()


def test_canonical_doc_contains_mermaid_architecture_diagram() -> None:
    visual = _section(_read_doc(), _SECTION_VISUAL)
    assert "```mermaid" in visual
    assert "flowchart" in visual
    for marker in _MERMAID_ARCHITECTURE_MARKERS:
        assert marker in visual, f"missing canonical architecture element in §3.1 diagram: {marker}"


def test_canonical_doc_describes_one_common_marketplace_for_verticals() -> None:
    preamble = _doc_preamble(_read_doc())
    assert "ONE COMMON CAPABILITY MARKETPLACE ENGINE" in preamble
    assert "AGENT" in preamble and "TOOL" in preamble and "SKILL" in preamble
    assert "vertical" in preamble.lower()
    vertical_count = preamble.lower().count("vertical")
    assert vertical_count >= 3, (
        "canonical preamble must name Agent, Tool, and Skill each as a vertical of one engine"
    )


def test_canonical_doc_describes_contract_variation_points() -> None:
    plugin = _section(_read_doc(), _SECTION_PLUGIN)
    for marker in _PLUGIN_VARIATION_POINT_MARKERS:
        assert marker in plugin, f"missing variation-point contract in §10 plugin matrix: {marker}"
    assert re.search(
        r"defaults\s+are\s+.*not.*\s+the\s+platform\s+contract",
        plugin,
        flags=re.IGNORECASE,
    ), "§10 must state that defaults are not the platform contract"


def test_canonical_doc_describes_lifecycle_handoff_boundary() -> None:
    handoff = _section(_read_doc(), _SECTION_HANDOFF)
    assert "typed lifecycle handoff" in handoff.lower() or "LIFECYCLE HANDOFF" in handoff
    assert "DOMAIN AUTHORITY" in handoff or "domain authority" in handoff.lower()
    assert re.search(
        r"HANDOFF_ACCEPTED[`\s]*≠\s*installed\s*≠\s*active\s*≠\s*routable\s*≠\s*executed",
        handoff,
        flags=re.IGNORECASE,
    ) or re.search(
        r"HANDOFF_ACCEPTED[`\s]*!=\s*installed\s*!=\s*active\s*!=\s*routable\s*!=\s*executed",
        handoff,
        flags=re.IGNORECASE,
    ), (
        "handoff § must preserve HANDOFF_ACCEPTED ≠ installed ≠ active ≠ routable ≠ executed "
        "invariant (or ASCII != chain)"
    )


def test_canonical_doc_marketplace_is_not_execution_engine() -> None:
    visual = _section(_read_doc(), _SECTION_VISUAL)
    assert "Execution Engine (execution only — outside Marketplace)" in visual
    assert re.search(
        r"\*\*Not Marketplace:\*\*.*runtime registry mutation",
        visual,
        flags=re.IGNORECASE | re.DOTALL,
    ), "§3.1 must exclude Marketplace from runtime registry mutation (execution-adjacent)"


def test_canonical_doc_marketplace_is_not_trust_authority() -> None:
    visual = _section(_read_doc(), _SECTION_VISUAL)
    assert re.search(
        r"\*\*Not Marketplace:\*\*\s*trust authority",
        visual,
        flags=re.IGNORECASE,
    ), "§3.1 must state Marketplace is not trust authority"


def test_canonical_doc_governance_before_recommendation() -> None:
    visual = _section(_read_doc(), _SECTION_VISUAL)
    assert re.search(
        r"governance\s+narrowing\s*→\s*recommendation",
        visual,
        flags=re.IGNORECASE,
    ), "canonical §3.1 pipeline must place governance narrowing before recommendation"


def test_canonical_doc_excludes_obsolete_me_rb1_recommendation_gap() -> None:
    text = _read_doc()
    for phrase in _OBSOLETE_DOC_PHRASES:
        assert phrase not in text, f"obsolete marketplace doc phrase still present: {phrase!r}"
