# © Artur Czarnecki. All rights reserved.

"""GR-10-R15-R1 — GR-8 evidence applicability vs GR-10/GR-13 certification scope."""

from __future__ import annotations

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_ORCHESTRATION_GEP_SEMANTICS,
    GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY,
    GR10_R15_R1_NEXT_REMEDIATION,
    GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED,
    Gr10Applicability,
    Gr10CoverageStatus,
    Gr10EvidenceCertificationRequirement,
    gr10_orchestration_gep_semantics,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ORCHESTRATION_GEPS = {row.gep for row in GR10_ORCHESTRATION_GEP_SEMANTICS}


def test_gr10_r15_r1_deferred_inventory_matches_semantics() -> None:
    assert {row.gep for row in GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED} == {
        row.gep
        for row in GR10_ORCHESTRATION_GEP_SEMANTICS
        if row.gr10_evidence_requirement is Gr10EvidenceCertificationRequirement.DEFERRED_TO_GR13
    }
    assert len(GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED) == 5


def test_gr10_r15_r1_no_false_not_applicable_on_governance_geps() -> None:
    """AGENT_DECISION excluded: policy row applies but orchestration graph routing is not GR-8 AGENT_DECISION spine."""
    genuine_gr8_na = frozenset({"AGENT_DECISION"})
    for sem in GR10_ORCHESTRATION_GEP_SEMANTICS:
        if sem.applicability is not Gr10Applicability.APPLICABLE:
            continue
        if sem.gep in genuine_gr8_na:
            continue
        assert sem.gr8_evidence_applicability is not Gr10Applicability.NOT_APPLICABLE, sem.gep


def test_gr10_r15_r1_deferred_rows_use_typed_deferral_not_na() -> None:
    for sem in GR10_ORCHESTRATION_GEP_SEMANTICS:
        if sem.gr10_evidence_requirement is not Gr10EvidenceCertificationRequirement.DEFERRED_TO_GR13:
            continue
        assert sem.gr8_evidence_applicability is Gr10Applicability.APPLICABLE, sem.gep
        assert (
            sem.gr13_evidence_requirement is Gr10EvidenceCertificationRequirement.REQUIRED_IN_GR13
        ), sem.gep
        assert sem.gr8_evidence_coverage is Gr10CoverageStatus.GAP, sem.gep


def test_gr10_r15_r1_required_in_gr10_rows_qualified_coverage() -> None:
    required = [
        row
        for row in GR10_ORCHESTRATION_GEP_SEMANTICS
        if row.gr10_evidence_requirement is Gr10EvidenceCertificationRequirement.REQUIRED_IN_GR10
    ]
    assert {row.gep for row in required} == {
        "ROOT_EXECUTION_ADMISSION",
        "MEANINGFUL_SIDE_EFFECT",
        "TOOL_INVOCATION_AUTHORIZATION",
    }
    for row in required:
        assert row.gr8_evidence_coverage is Gr10CoverageStatus.QUALIFIED, row.gep


def test_gr10_r15_r1_agent_decision_and_interrupt_evidence_genuinely_na() -> None:
    for gep in ("AGENT_DECISION", "INTERRUPT"):
        sem = gr10_orchestration_gep_semantics(gep)
        assert sem.gr8_evidence_applicability is Gr10Applicability.NOT_APPLICABLE
        assert (
            sem.gr10_evidence_requirement is Gr10EvidenceCertificationRequirement.NOT_APPLICABLE
        )
        assert (
            sem.gr13_evidence_requirement is Gr10EvidenceCertificationRequirement.NOT_APPLICABLE
        )


def test_gr10_r15_r1_r14_inventory_gr10_scope_qualified_deferred_partitioned() -> None:
    gr10_required_paths = {
        "Root execution admission",
        "Meaningful side effect authorization (production orchestration)",
        "Canonical inner guard DENY",
        "Decision-requirement DENY",
        "Decision-bound ALLOW",
        "Post-HITL fresh ALLOW",
        "Stale / cross-run / cross-resource / cross-digest grant",
    }
    deferred_paths = {"PRE_OUTPUT evaluation point", "POST_RUN evaluation point"}
    for row in GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY:
        if row.path in gr10_required_paths:
            assert row.status == "QUALIFIED", row.path
        elif row.path in deferred_paths:
            assert row.status == "DEFERRED_TO_GR13", row.path
        else:
            pytest.fail(f"unexpected inventory path: {row.path}")


def test_gr10_r15_r1_next_remediation_agentic_closure() -> None:
    assert GR10_R15_R1_NEXT_REMEDIATION.strategy == "AGENTIC"
    assert "GR-10" in GR10_R15_R1_NEXT_REMEDIATION.task_name
