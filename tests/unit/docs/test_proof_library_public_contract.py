# © Artur Czarnecki. All rights reserved.

"""Public Proof Library landing page contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROOF_LIBRARY_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "PROOF_LIBRARY.md"
_SCENARIO_DESIGN_PATH = (
    REPO_ROOT
    / "platform_proofs"
    / "scenarios"
    / "ai_incident_investigation"
    / "README.md"
)


@pytest.fixture(scope="module")
def proof_library_text() -> str:
    return PROOF_LIBRARY_PATH.read_text(encoding="utf-8")


def test_problem_first_framing(proof_library_text: str) -> None:
    normalized = re.sub(r"[*_`]", "", proof_library_text).lower()
    assert "real problems. executable evidence." in normalized
    assert "difficult real-world" in normalized or "difficult system" in normalized
    assert "executable" in normalized
    assert "falsif" in normalized


def test_scenario_not_product_boundary(proof_library_text: str) -> None:
    normalized = re.sub(r"[*_`]", "", proof_library_text).lower()
    assert "scenario proofs are not products" in normalized
    assert "real-user validation" in normalized
    assert "commercial validation" in normalized
    assert "production readiness" in normalized


def test_challenge_intergrax_route(proof_library_text: str) -> None:
    assert "Challenge Intergrax" in proof_library_text
    assert "scenario_proposal.yml" in proof_library_text


def test_catalog_truth(proof_library_text: str) -> None:
    normalized = re.sub(r"[*_`]", "", proof_library_text).lower()
    assert "no accepted scenario proofs are published yet" in normalized
    assert "in development" in normalized
    assert "ai_incident_investigation" in proof_library_text
    featured_section = proof_library_text.split("## D. Featured scenario in development", 1)[1]
    featured_normalized = re.sub(r"[*_`]", "", featured_section[:800]).lower()
    assert "not accepted proof evidence" in featured_normalized or "in development" in featured_normalized
    assert "verdict: pass" not in featured_normalized


def test_premium_structure(proof_library_text: str) -> None:
    for heading in (
        "## A. What is a Scenario Proof?",
        "## B. What makes a scenario worth publishing?",
        "## C. Scenario catalog",
        "## D. Featured scenario in development",
        "## E. How to read a proof",
        "## F. PASS / FAIL / UNRESOLVED semantics",
        "## G. Challenge Intergrax",
        "## H. Proof Library vs evidence dashboard",
        "## I. Related routes",
    ):
        assert heading in proof_library_text, f"Missing section: {heading}"


def test_visual_integration(proof_library_text: str) -> None:
    assert "intergrax-scenarios-overview-light.png" in proof_library_text
    assert "scenario-ai-incident-investigation-light.png" in proof_library_text
    preview_refs = (
        "fullsize/intergrax-scenarios-overview.md",
        "fullsize/scenario-ai-incident-investigation.md",
    )
    for ref in preview_refs:
        assert ref in proof_library_text


def test_proofs_dashboard_distinction(proof_library_text: str) -> None:
    assert "PROOFS.md" in proof_library_text
    assert "evidence dashboard" in proof_library_text.lower()


def test_scenario_design_link_exists() -> None:
    assert _SCENARIO_DESIGN_PATH.is_file()
