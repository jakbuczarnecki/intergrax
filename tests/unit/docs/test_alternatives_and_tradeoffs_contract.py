# © Artur Czarnecki. All rights reserved.

"""AUD-COMP-1A: public alternatives/trade-offs contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
ALTERNATIVES_PATH = REPO_ROOT / "docs" / "project" / "overview" / "ALTERNATIVES_AND_TRADEOFFS.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize(text: str) -> str:
    return re.sub(r"[*_`]", "", text).lower()


@pytest.fixture(scope="module")
def alternatives_text() -> str:
    return _read(ALTERNATIVES_PATH)


def test_alternatives_document_exists() -> None:
    assert ALTERNATIVES_PATH.is_file()


def test_agentcore_section_present(alternatives_text: str) -> None:
    assert "## Amazon Bedrock AgentCore" in alternatives_text


def test_agentcore_may_be_better_choice(alternatives_text: str) -> None:
    normalized = _normalize(alternatives_text)
    section_heading = "## amazon bedrock agentcore"
    agentcore_start = normalized.index(section_heading)
    agentcore_section = normalized[agentcore_start:]
    next_section = agentcore_section.find("\n## ", len(section_heading))
    bounded = agentcore_section if next_section == -1 else agentcore_section[:next_section]
    assert "may be the better choice" in bounded or "better choice" in bounded


def test_agentcore_acknowledges_gateway_policy_authorization(alternatives_text: str) -> None:
    normalized = _normalize(alternatives_text)
    section_heading = "## amazon bedrock agentcore"
    agentcore_start = normalized.index(section_heading)
    agentcore_section = normalized[agentcore_start:]
    next_section = agentcore_section.find("\n## ", len(section_heading))
    bounded = agentcore_section if next_section == -1 else agentcore_section[:next_section]
    assert "gateway" in bounded
    assert "cedar" in bounded or "policy" in bounded
    assert "outside agent code" in bounded or "gateway boundary" in bounded


def test_cross_product_reuse_remains_hypothesis(alternatives_text: str) -> None:
    normalized = _normalize(alternatives_text)
    assert "hypothesis" in normalized
    assert "cross-product reuse" in normalized or "cross-product" in normalized
    forbidden_proven = (
        "cross-product reuse is proven",
        "cross-product reuse proven",
        "proven cross-product reuse",
        "measured delivery acceleration is established",
    )
    for phrase in forbidden_proven:
        assert phrase not in normalized, f"Alternatives doc claims proven reuse: {phrase!r}"


def test_shared_positioning_framing_across_sections(alternatives_text: str) -> None:
    """POS-1: common framing derives comparisons from operating-model responsibility."""
    normalized = _normalize(alternatives_text)
    assert "not whether the competitor has tracing" in normalized or (
        "question is not whether the competitor has tracing" in normalized
    )
    assert "shared application operating model" in normalized
    assert "feature matrix" in normalized
    assert "where architectural responsibility sits" in normalized or (
        "architectural responsibility" in normalized
    )
    assert "not treated as intergrax differentiators by default" in normalized or (
        "not" in normalized and "differentiators by default" in normalized
    )


def test_agentcore_aws_primary_sources_linked(alternatives_text: str) -> None:
    for url in (
        "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html",
        "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html",
        "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html",
        "https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html",
    ):
        assert url in alternatives_text, f"Missing AWS primary source link: {url}"


def test_agentcore_verification_date_distinct(alternatives_text: str) -> None:
    assert "verified 2026-08-23" in alternatives_text.lower() or "verified **2026-08-23**" in alternatives_text
    assert "2026-08-18" in alternatives_text


def _bounded_section(text: str, heading: str) -> str:
    normalized = _normalize(text)
    section_heading = f"## {heading}"
    section_start = normalized.index(section_heading)
    section_body = normalized[section_start:]
    next_section = section_body.find("\n## ", len(section_heading))
    return section_body if next_section == -1 else section_body[:next_section]


_CREWAI_STANDARD_HEADINGS = (
    "best fit / strengths",
    "choose it when",
    "what it already solves",
    "responsibilities / questions your team still needs to settle",
    "how intergrax approaches responsibility differently",
    "current intergrax evidence boundary",
)

_CREWAI_FORBIDDEN_STRAWMAN_PHRASES = (
    "crewai has no persistence",
    "crewai has no observability",
    "crewai is only a multi-agent framework",
    "crewai requires langchain",
    "crewai has no hitl",
    "crewai has no state management",
    "crewai is cloud-only",
)


def test_crewai_section_present(alternatives_text: str) -> None:
    assert "## CrewAI" in alternatives_text


def test_crewai_standard_structure(alternatives_text: str) -> None:
    section = _bounded_section(alternatives_text, "crewai")
    for heading in _CREWAI_STANDARD_HEADINGS:
        assert heading in section, f"Missing CrewAI subsection: {heading}"


def test_crewai_required_fairness_claims(alternatives_text: str) -> None:
    section = _bounded_section(alternatives_text, "crewai")
    assert "may be the better choice" in section
    assert "flows" in section
    assert "persistence" in section or "persist" in section
    assert "state" in section
    assert "enterprise" in section or "amp" in section


def test_crewai_no_strawman_claims(alternatives_text: str) -> None:
    section = _bounded_section(alternatives_text, "crewai")
    for phrase in _CREWAI_FORBIDDEN_STRAWMAN_PHRASES:
        assert phrase not in section, f"CrewAI section contains strawman: {phrase!r}"


def test_crewai_evidence_boundary_conservative(alternatives_text: str) -> None:
    section = _bounded_section(alternatives_text, "crewai")
    assert "active r&d" in section or "active r & d" in section
    assert "bounded" in section
    assert "hypothesis" in section
    forbidden_superiority = (
        "production maturity superiority",
        "more mature than crewai",
        "broader production maturity than crewai",
        "ecosystem superiority",
    )
    for phrase in forbidden_superiority:
        assert phrase not in section, f"CrewAI evidence boundary overclaims: {phrase!r}"


def test_crewai_verification_date_distinct(alternatives_text: str) -> None:
    assert "verified 2026-08-24" in alternatives_text.lower() or "verified **2026-08-24**" in alternatives_text
