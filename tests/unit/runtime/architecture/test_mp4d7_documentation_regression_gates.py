# © Artur Czarnecki. All rights reserved.

"""MP-4D7 — documentation regression gates (semantic invariants across MP-4 SSOT docs)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

CANONICAL_MP4_DOC = _REPO_ROOT / "docs" / "project" / "architecture" / "DECISION_APPROVAL_GOVERNANCE.md"
MAINTAINER_PLAN_DOC = (
    _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "DECISION_APPROVAL_GOVERNANCE.md"
)
MULTIPLAYER_ARCH_DOC = (
    _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
)
MULTIPLAYER_PLAN_DOC = _REPO_ROOT / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md"

_STATUS_DOCS: dict[str, Path] = {
    "canonical": CANONICAL_MP4_DOC,
    "maintainer_plan": MAINTAINER_PLAN_DOC,
    "multiplayer_architecture": MULTIPLAYER_ARCH_DOC,
    "multiplayer_plan": MULTIPLAYER_PLAN_DOC,
}

_D7_GATE_MODULE = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "architecture"
    / "test_mp4d7_documentation_regression_gates.py"
)

_MP4_PLUGINABILITY_CONTRACTS = (
    "CollaborativeDecisionBindingRepository",
    "DecisionHumanReviewPort",
    "DecisionAuthorizationEvaluator",
    "ExecutionContinuationPort",
    "ExecutionContinuationStateStore",
    "FunctionalEvidencePersistence",
    "ExecutionReconstructionReader",
)

_COMPETING_MP4_SSOT_CLAIMS = (
    "canonical MP-4 integration architecture entry point (SSOT)",
    "This file is the canonical MP-4 integration architecture entry point",
)


def _read_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _assert_contains_all(text: str, phrases: tuple[str, ...], context: str) -> None:
    missing = [phrase for phrase in phrases if phrase not in text]
    assert not missing, f"{context}: missing semantic markers: {missing}"


def _assert_contains_any(text: str, phrases: tuple[str, ...], context: str) -> None:
    if any(phrase in text for phrase in phrases):
        return
    assert False, f"{context}: expected at least one of {phrases!r}"


def _canonical_d7_closed(text: str) -> bool:
    return "MP-4D7 — CLOSED" in text or bool(
        re.search(r"\*\*MP-4D7\*\*\s*\|\s*\*\*CLOSED\*\*", text)
    )


def _canonical_d8_next(text: str) -> bool:
    return "MP-4D8 — NEXT" in text or bool(
        re.search(r"\*\*MP-4D8\*\*\s*\|\s*\*\*NEXT\*\*", text)
    ) or bool(re.search(r"MP-4D8[^|\n]{0,24}\bNEXT\b", text))


def test_mp4d7_status_documents_are_synchronized() -> None:
    texts = {name: _read_doc(path) for name, path in _STATUS_DOCS.items()}
    canonical = texts["canonical"]

    assert _canonical_d7_closed(canonical), (
        "MP-4 status drift: canonical SSOT must mark MP-4D7 CLOSED after D7 close"
    )
    assert _canonical_d8_next(canonical), (
        "MP-4 status drift: canonical SSOT must mark MP-4D8 as NEXT active documentation stage"
    )
    _assert_contains_any(
        canonical,
        ("MP-4 — FORMALLY CLOSED", "MP-4 implementation — FORMALLY CLOSED"),
        "MP-4 implementation closure",
    )
    assert "MP-4R0…MP-4R8 CLOSED" in canonical or "MP-4R0…MP-4R8 **CLOSED**" in canonical, (
        "MP-4 status drift: canonical SSOT must keep MP-4R program formally closed"
    )

    for name, text in texts.items():
        if name == "canonical":
            continue
        assert "FORMALLY CLOSED" in text, (
            f"MP-4 status drift: {name} disagrees with canonical implementation closure"
        )
        assert "MP-4D7 — NEXT" not in text and "**MP-4D7** — Documentation regression gates (**NEXT**)" not in text, (
            f"MP-4 status drift: {name} still marks MP-4D7 NEXT while canonical SSOT has D7 CLOSED"
        )
        assert _canonical_d7_closed(text) or "MP-4D1–D7 CLOSED" in text or "MP-4D1–D6 CLOSED · MP-4D7 CLOSED" in text, (
            f"MP-4 status drift: {name} does not reflect MP-4D7 CLOSED"
        )
        assert "MP-4 implementation IN PROGRESS" not in text and "MP-4 — IN PROGRESS" not in text, (
            f"MP-4 status drift: {name} contradicts formally closed MP-4 implementation"
        )


def test_mp4d7_canonical_ssot_entry_point_is_preserved() -> None:
    canonical = _read_doc(CANONICAL_MP4_DOC)
    _assert_contains_all(
        canonical,
        (
            "canonical MP-4 integration architecture entry point (SSOT)",
            "not a second architecture SSOT",
        ),
        "MP-4 canonical SSOT declaration",
    )

    for name, path in (
        ("multiplayer_architecture", MULTIPLAYER_ARCH_DOC),
        ("multiplayer_plan", MULTIPLAYER_PLAN_DOC),
        ("maintainer_plan", MAINTAINER_PLAN_DOC),
    ):
        text = _read_doc(path)
        for claim in _COMPETING_MP4_SSOT_CLAIMS:
            assert claim not in text, (
                f"MP-4 SSOT regression: {name} declares a competing canonical architecture entry point"
            )
        assert "DECISION_APPROVAL_GOVERNANCE" in text, (
            f"MP-4 SSOT regression: {name} must reference canonical DECISION_APPROVAL_GOVERNANCE hub"
        )


def test_mp4d7_canonical_authority_boundaries_are_preserved() -> None:
    canonical = _read_doc(CANONICAL_MP4_DOC)
    _assert_contains_all(
        canonical,
        (
            "Decision System owns Decision truth and lifecycle",
            "does **not** own Decision, Governance, Execution lifecycle",
            "Human APPROVED ≠ Governance ALLOW",
            "post-human Governance re-evaluation",
            "association truth",
            "DecisionProposalRef",
            "WHETHER the operation is permitted",
            "ALLOW",
            "DENY",
            "REQUIRE_HUMAN",
            "DecisionExecutionAuthorization",
            "must not** start or resume solely",
            "ExecutionContinuationPort",
            "Execution Engine owns semantics",
            "Nexus:** internal orchestration",
            "not** a public MP-4 integration surface",
            "Factual reconstruction** rebuilds **facts**",
            "Diagnostics** **interprets**",
            "does **not** authorize execution",
        ),
        "MP-4 authority boundary",
    )

    multiplayer_arch = _read_doc(MULTIPLAYER_ARCH_DOC)
    forbidden = (
        "Multiplayer owns Decision lifecycle",
        "Multiplayer owns continuation lifecycle",
        "Human APPROVED = automatic ALLOW",
    )
    for phrase in forbidden:
        assert phrase not in multiplayer_arch, (
            f"MP-4 authority regression: capability architecture states forbidden authority model ({phrase!r})"
        )


def test_mp4d7_contract_first_pluginability_contracts_are_documented() -> None:
    canonical = _read_doc(CANONICAL_MP4_DOC)
    _assert_contains_all(
        canonical,
        ("PLATFORM OPERATES ON CONTRACTS, NOT IMPLEMENTATIONS", *_MP4_PLUGINABILITY_CONTRACTS),
        "MP-4 contract-first pluginability",
    )
    assert "Enterprise Boundary & Pluginability Certification (MP-4D6)" in canonical, (
        "MP-4 contract-first regression: MP-4D6 boundary certification section missing from SSOT"
    )


def test_mp4d7_proof_and_provider_qualification_boundaries_are_preserved() -> None:
    canonical = _read_doc(CANONICAL_MP4_DOC)
    _assert_contains_all(
        canonical,
        (
            "PostgreSQLCollaborativeDecisionBindingRepository",
            "**does not** qualify governance",
            "R7 test-composition E2E",
            "Does not claim:** full production-deployment E2E",
            "Invariant → proof",
            "Provider → durability → qualification",
            "Contract boundary → external replacement",
            "NOT QUALIFIED** (MP-4-scoped)",
            "**CERTIFIED**",
        ),
        "MP-4 qualification boundary",
    )
    _assert_contains_any(
        canonical,
        (
            "R7 E2E uses test composition",
            "test composition",
            "in-memory binding",
        ),
        "MP-4 R7 qualification scope",
    )
    assert "PostgreSQL proves full MP-4 production E2E" not in canonical, (
        "MP-4 qualification regression: canonical SSOT overstates PostgreSQL binding proof as full MP-4 E2E"
    )


def test_mp4d7_evidence_reconstruction_diagnostics_separation_is_preserved() -> None:
    canonical = _read_doc(CANONICAL_MP4_DOC)
    _assert_contains_all(
        canonical,
        (
            "Evidence Plane",
            "no duplicate Evidence store",
            "Evidence facts → reconstruction → diagnostics interpretation",
            "Association truth is **not** reconstructed from Evidence facts as authority",
        ),
        "MP-4 evidence / reconstruction / diagnostics separation",
    )

    assert "Documentation Regression Gates (MP-4D7)" in canonical, (
        "MP-4D7 documentation: canonical SSOT must document regression gate scope"
    )
    assert str(_D7_GATE_MODULE.relative_to(_REPO_ROOT)).replace("\\", "/") in canonical, (
        "MP-4D7 documentation: canonical SSOT must point maintainers to D7 gate module path"
    )
