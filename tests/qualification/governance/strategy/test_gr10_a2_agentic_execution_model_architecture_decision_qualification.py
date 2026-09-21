# © Artur Czarnecki. All rights reserved.

"""GR-10-A2 — AGENTIC execution model architecture decision gates (UAEP vs ACP session)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.governance.strategy.catalog import (
    GR10_A2_ADR_REFERENCE,
    GR10_A2_ARCHITECTURE_DECISION,
    GR10_A2_CHECKPOINT_SESSION_COUPLING,
    GR10_A3_NEXT_REMEDIATION,
    GR10_AGENTIC_LEGAL_PRODUCTION_PATHS,
    GR10_R15_R1_NEXT_REMEDIATION,
    Gr10AgenticExecutionArchitectureDecision,
    Gr10CheckpointSessionCouplingStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ADR = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "technical"
    / "adr"
    / "entries"
    / "2026-09-21"
    / "ADR-GR-10-004-agentic-execution-model-uaep-canonical.md"
)
_ENRICHER = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "acp_checkpoint_task_enricher.py"
)


def test_gr10_a2_ssot_decision_uaep_canonical_acp_explicit() -> None:
    assert (
        GR10_A2_ARCHITECTURE_DECISION
        is Gr10AgenticExecutionArchitectureDecision.UAEP_CANONICAL_ACP_EXPLICIT
    )


def test_gr10_a2_checkpoint_session_coupling_decoupled_retains_explicit_contract() -> None:
    assert (
        GR10_A2_CHECKPOINT_SESSION_COUPLING
        is Gr10CheckpointSessionCouplingStatus.RETAINED_AS_CONTRACT
    )


def test_gr10_a2_adr_file_exists_and_records_decision() -> None:
    text = _ADR.read_text(encoding="utf-8-sig")
    assert GR10_A2_ADR_REFERENCE in text
    assert "UAEP_CANONICAL_ACP_EXPLICIT" in text
    assert "DUAL_CANONICAL_AGENTIC_EXECUTION" in text


def test_gr10_a2_p_uaep_sole_qualified_canonical_path() -> None:
    qualified = [row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.status == "QUALIFIED"]
    assert len(qualified) == 1
    assert qualified[0].path_id == "P-UAEP"


def test_gr10_a2_p_acp_session_explicit_non_canonical_not_second_spine() -> None:
    acp = next(row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.path_id == "P-ACP-SESSION")
    assert acp.status == "EXPLICIT_NON_CANONICAL"
    assert "non-canonical" in acp.canonical_governance_owner.lower()


def test_gr10_a2_next_remediation_after_a3_is_final_recertification() -> None:
    assert GR10_A3_NEXT_REMEDIATION is GR10_R15_R1_NEXT_REMEDIATION
    assert "GR-10" in GR10_A3_NEXT_REMEDIATION.task_name


def test_gr10_a2_enricher_does_not_set_session_enabled() -> None:
    source = _ENRICHER.read_text(encoding="utf-8-sig")
    assert "SESSION_ENABLED" not in source
    adr = _ADR.read_text(encoding="utf-8-sig")
    assert "make_acp_checkpoint_task_enricher" in adr
    assert "DEPRECATED_MIGRATION_TARGET" in adr or "deprecated" in adr.lower()
