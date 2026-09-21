# © Artur Czarnecki. All rights reserved.

"""GR-10-R15-R2 — MSE ADR lineage SSOT + AGENT_DECISION delegated ownership gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_R9_NEXT_REMEDIATION,
    Gr10Applicability,
    Gr10CoverageStatus,
    Gr10EvidenceCertificationRequirement,
    gr10_orchestration_gep_semantics,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_CATALOG = Path(__file__).resolve().parent / "catalog.py"
_CATALOG_SOURCE = _CATALOG.read_text(encoding="utf-8")


def test_gr10_r15_r2_mse_lineage_adr_gr10_002_not_evidence_003() -> None:
    assert (
        "MeaningfulSideEffectAuthorizationPort (ADR-GR-10-002)" in _CATALOG_SOURCE
    )
    assert "ADR-GR-10-002 rejected design" in _CATALOG_SOURCE
    assert "MeaningfulSideEffectAuthorizationPort (ADR-GR-10-003)" not in _CATALOG_SOURCE
    assert "(ADR-GR-10-003 rejected design)" not in _CATALOG_SOURCE


def test_gr10_r15_r2_gr10_vs_gr13_evidence_scope_still_adr_gr10_003() -> None:
    assert "DEFERRED_TO_GR13 (ADR-GR-10-003)" in _CATALOG_SOURCE
    assert _CATALOG_SOURCE.count("ADR-GR-10-003") >= 3


def test_gr10_r15_r2_agent_decision_delegated_ownership_not_orchestration_evidence_gap() -> None:
    sem = gr10_orchestration_gep_semantics("AGENT_DECISION")
    assert sem.applicability is Gr10Applicability.APPLICABLE
    assert sem.coverage is Gr10CoverageStatus.QUALIFIED
    assert sem.gr8_evidence_applicability is Gr10Applicability.NOT_APPLICABLE
    assert (
        sem.gr10_evidence_requirement is Gr10EvidenceCertificationRequirement.NOT_APPLICABLE
    )
    assert (
        sem.gr13_evidence_requirement is Gr10EvidenceCertificationRequirement.NOT_APPLICABLE
    )
    assert sem.gr8_evidence_coverage is Gr10CoverageStatus.NOT_APPLICABLE
    assert "UAEP" in sem.canonical_owner
    assert "graph" in sem.production_path.lower()
    assert "AGENTIC" in sem.reason or "UAEP" in sem.reason
    assert "GR-10-R15-R2" in sem.reason


def test_gr10_r15_r2_r9_historical_blocker_uses_mse_adr_not_evidence_adr() -> None:
    assert "ADR-GR-10-002" in GR10_R9_NEXT_REMEDIATION.exact_blocker
    assert "ADR-GR-10-003" not in GR10_R9_NEXT_REMEDIATION.exact_blocker
