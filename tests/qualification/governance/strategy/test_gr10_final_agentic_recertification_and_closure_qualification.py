# © Artur Czarnecki. All rights reserved.

"""GR-10 — AGENTIC final recertification and whole GR-10 formal closure gates."""

from __future__ import annotations

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_AGENTIC_CAPABILITY_SEMANTICS,
    GR10_AGENTIC_FORMAL_CLOSURE,
    GR10_AGENTIC_LEGAL_PRODUCTION_PATHS,
    GR10_INFERENCE_FORMAL_CLOSURE,
    GR10_ORCHESTRATION_CAPABILITY_SEMANTICS,
    GR10_ORCHESTRATION_FORMAL_CLOSURE,
    GR10_OVERALL_FORMAL_CLOSURE,
    GR10_POST_CLOSURE_NEXT_REMEDIATION,
    GR10_R15_R1_NEXT_REMEDIATION,
    Gr10CoverageStatus,
)
from tests.qualification.governance.strategy.gr10_inference_current_doc_ssot import (
    gr10_architecture_remaining_gaps_slice,
    gr10_maintainer_roadmap_slice,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_gr10_final_closure_status_ssot() -> None:
    assert GR10_INFERENCE_FORMAL_CLOSURE.status == "CLOSED"
    assert "FINAL CLOSED" in GR10_ORCHESTRATION_FORMAL_CLOSURE.status
    assert "FINAL CLOSED" in GR10_AGENTIC_FORMAL_CLOSURE.status
    assert "FINAL CLOSED" in GR10_OVERALL_FORMAL_CLOSURE.status
    assert "GR-13" in GR10_AGENTIC_FORMAL_CLOSURE.gr13_deferred_note
    assert "GR-13" in GR10_OVERALL_FORMAL_CLOSURE.gr13_deferred_note


def test_gr10_final_post_closure_next_is_gr12() -> None:
    assert GR10_POST_CLOSURE_NEXT_REMEDIATION.task_name.startswith("GR-12")
    assert GR10_POST_CLOSURE_NEXT_REMEDIATION.strategy == "PLATFORM"


def test_gr10_final_r15_remediation_records_prior_closure_task() -> None:
    assert "Recertification" in GR10_R15_R1_NEXT_REMEDIATION.task_name
    assert "Closure" in GR10_R15_R1_NEXT_REMEDIATION.task_name


def test_gr10_final_agentic_legal_paths_uaep_canonical_acp_explicit() -> None:
    by_id = {row.path_id: row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS}
    assert by_id["P-UAEP"].status == "QUALIFIED"
    assert by_id["P-ACP-SESSION"].status == "EXPLICIT_NON_CANONICAL"


def test_gr10_final_agentic_capabilities_only_governance_evidence_partial() -> None:
    partial = [
        row.capability
        for row in GR10_AGENTIC_CAPABILITY_SEMANTICS
        if row.coverage is Gr10CoverageStatus.PARTIAL
    ]
    assert partial == ["Governance Evidence"]
    assert not any(
        row.coverage is Gr10CoverageStatus.GAP for row in GR10_AGENTIC_CAPABILITY_SEMANTICS
    )


def test_gr10_final_orchestration_no_partial_capability_rows() -> None:
    partial = [
        row.capability
        for row in GR10_ORCHESTRATION_CAPABILITY_SEMANTICS
        if row.coverage is Gr10CoverageStatus.PARTIAL
    ]
    assert partial == []


def test_gr10_final_docs_record_gr10_closure() -> None:
    arch = gr10_architecture_remaining_gaps_slice()
    plan = gr10_maintainer_roadmap_slice()
    assert "FINAL CLOSED" in arch and "GR-10" in arch
    assert "GR-12" in arch or "GR-12" in plan
    assert "GR-13" in arch or "deferred" in arch.lower()
