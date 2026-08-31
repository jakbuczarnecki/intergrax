# © Artur Czarnecki. All rights reserved.

"""Public Proof Library landing page contract tests."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]
PROOF_LIBRARY_PATH = REPO_ROOT / "docs" / "project" / "proofs" / "PROOF_LIBRARY.md"
_FEATURED_SECTION_HEADING = "## B. Featured scenarios in development"
_CATALOG_SECTION_HEADING = "## A. Scenario catalog"
_METHODOLOGY_SECTION_HEADING = "## C. What is a Scenario Proof?"
_SCENARIO_DESIGN_PATH = (
    REPO_ROOT
    / "platform_proofs"
    / "scenarios"
    / "ai_incident_investigation"
    / "README.md"
)
_INDIRECT_PROMPT_INJECTION_DESIGN_PATH = (
    REPO_ROOT
    / "platform_proofs"
    / "scenarios"
    / "indirect_prompt_injection"
    / "README.md"
)

_STALE_INCIDENT_SCENARIO_LOGISTICS_MARKERS = (
    "warehouse overload",
    "warehouse",
    "parcel",
    "sorter",
    "logistics operator",
    "heavy parcel",
)

_STALE_FULL2_PENDING_MARKERS = (
    "unresolved path pending",
    "pending full-2",
    "full-2) and public proof publication not started",
    "full-2 unresolved path and public publication not started",
)

_PUBLIC_INCIDENT_STATUS_PROJECTION_PATHS = (
    REPO_ROOT / "README.md",
    PROOF_LIBRARY_PATH,
    REPO_ROOT / "platform_proofs/scenarios/ai_incident_investigation/SCENARIO_SPEC.md",
    _SCENARIO_DESIGN_PATH,
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


def test_scenario_proof_positive_before_boundary(proof_library_text: str) -> None:
    """Positive Scenario Proof capability precedes the detailed validation boundary."""
    normalized = re.sub(r"[*_`]", "", proof_library_text).lower()
    boundary_idx = normalized.index("scenario proofs are not products")
    positive_markers = ("falsif", "stress-test", "inspectable evidence", "reproduced")
    earliest_positive = min(
        normalized.index(marker) for marker in positive_markers if marker in normalized
    )
    assert earliest_positive < boundary_idx


def test_challenge_intergrax_route(proof_library_text: str) -> None:
    assert "Challenge Intergrax" in proof_library_text
    assert "scenario_proposal.yml" in proof_library_text


def _featured_section(proof_library_text: str) -> str:
    section = proof_library_text.split(_FEATURED_SECTION_HEADING, 1)[1]
    return section.split(_METHODOLOGY_SECTION_HEADING, 1)[0]


def test_featured_incident_scenario_no_stale_logistics_framing(
    proof_library_text: str,
) -> None:
    """Featured AI Incident Investigation prose must not regress to logistics fixture wording."""
    featured_section = _featured_section(proof_library_text)
    incident_section = featured_section.split(
        "### Indirect Prompt Injection with Governed Action Prevention", 1
    )[0]
    normalized = re.sub(r"[*_`]", "", incident_section).lower()
    for marker in _STALE_INCIDENT_SCENARIO_LOGISTICS_MARKERS:
        assert marker not in normalized, (
            f"Stale logistics framing in featured incident scenario: {marker!r}"
        )
    assert "workload overload" in normalized


def test_catalog_truth(proof_library_text: str) -> None:
    normalized = re.sub(r"[*_`]", "", proof_library_text).lower()
    assert "no accepted scenario proofs are published yet" in normalized
    assert "in development" in normalized
    assert "ai_incident_investigation" in proof_library_text
    assert "indirect_prompt_injection" in proof_library_text
    featured_section = _featured_section(proof_library_text)
    featured_normalized = re.sub(r"[*_`]", "", featured_section[:1200]).lower()
    assert "not accepted proof evidence" in featured_normalized or "in development" in featured_normalized
    assert "verdict: pass" not in featured_normalized


def _readme_incident_scenario_section(readme_text: str) -> str:
    start = readme_text.index("## Real problems. Executable evidence.")
    end = readme_text.index("## Local Knowledge Workspace (LKW)")
    return readme_text[start:end]


def _projection_text_for_stale_guard(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if path.name == "README.md" and path.parent == REPO_ROOT:
        return _readme_incident_scenario_section(text)
    return text


def test_incident_scenario_public_projections_no_stale_full2_pending() -> None:
    """Public status projections must not regress FULL-2 to pending/not-started."""
    for path in _PUBLIC_INCIDENT_STATUS_PROJECTION_PATHS:
        projection = _projection_text_for_stale_guard(path)
        normalized = re.sub(r"[*_`]", "", projection).lower()
        for marker in _STALE_FULL2_PENDING_MARKERS:
            assert marker not in normalized, (
                f"{path.relative_to(REPO_ROOT)}: stale FULL-2 pending marker {marker!r}"
            )
        assert "full-2" in normalized, (
            f"{path.relative_to(REPO_ROOT)}: missing FULL-2 implemented projection"
        )
        assert "implemented" in normalized
        assert (
            "not yet accepted" in normalized
            or "not yet established" in normalized
            or "not accepted" in normalized
            or "not yet published" in normalized
            or "publication is still pending" in normalized
        )


def test_incident_scenario_projection_semantics(proof_library_text: str) -> None:
    """AI Incident Investigation first-contact status matches canonical scenario semantics."""
    normalized = re.sub(r"[*_`]", "", proof_library_text).lower()
    assert "full-1" in normalized
    assert "full-2" in normalized
    assert "implemented" in normalized
    assert "executable" in normalized
    assert (
        "public scenario proof not yet accepted" in normalized
        or "public proof" in normalized
        or "not accepted" in normalized
    )
    assert "no executable proof yet" not in normalized
    assert "design accepted for implementation" not in normalized
    featured_section = _featured_section(proof_library_text)
    incident_section = featured_section.split(
        "### Indirect Prompt Injection with Governed Action Prevention", 1
    )[0]
    featured_normalized = re.sub(r"[*_`]", "", incident_section).lower()
    assert "no report, evidence bundle, or reproduction path exists yet" not in featured_normalized
    assert (
        "no accepted published evidence bundle" in featured_normalized
        or "not accepted for public publication" in featured_normalized
        or "no accepted published" in featured_normalized
    )


def test_premium_structure(proof_library_text: str) -> None:
    for heading in (
        _CATALOG_SECTION_HEADING,
        _FEATURED_SECTION_HEADING,
        _METHODOLOGY_SECTION_HEADING,
        "## D. What makes a scenario worth publishing?",
        "## E. How to read a proof",
        "## F. PASS / FAIL / UNRESOLVED semantics",
        "## G. Challenge Intergrax",
        "## H. Proof Library vs evidence dashboard",
        "## I. Related routes",
    ):
        assert heading in proof_library_text, f"Missing section: {heading}"


def test_scenario_catalog_precedes_methodology(proof_library_text: str) -> None:
    catalog_idx = proof_library_text.index(_CATALOG_SECTION_HEADING)
    methodology_idx = proof_library_text.index(_METHODOLOGY_SECTION_HEADING)
    assert catalog_idx < methodology_idx


def test_catalog_thumbnail_previews(proof_library_text: str) -> None:
    catalog_section = proof_library_text.split(_CATALOG_SECTION_HEADING, 1)[1].split(
        _FEATURED_SECTION_HEADING, 1
    )[0]
    assert "ai_incident_investigation/assets/proof-story-light.svg" in catalog_section
    assert "indirect_prompt_injection/assets/scenario-overview.png" in catalog_section
    assert "| Preview | Scenario | What makes it hard | Status |" in catalog_section


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


def test_indirect_prompt_injection_featured_projection(proof_library_text: str) -> None:
    """Indirect prompt injection featured block must not claim verified proof."""
    featured_section = _featured_section(proof_library_text)
    injection_section = featured_section.split(
        "### Indirect Prompt Injection with Governed Action Prevention", 1
    )[1]
    normalized = re.sub(r"[*_`]", "", injection_section).lower()
    assert "implementation initialized" in normalized
    assert "no verified proof run yet" in normalized
    assert "indirect_prompt_injection/assets/scenario-overview.png" in injection_section
    assert "verdict: pass" not in normalized
    assert "production ready" not in normalized


def test_scenario_design_link_exists() -> None:
    assert _SCENARIO_DESIGN_PATH.is_file()
    assert _INDIRECT_PROMPT_INJECTION_DESIGN_PATH.is_file()
