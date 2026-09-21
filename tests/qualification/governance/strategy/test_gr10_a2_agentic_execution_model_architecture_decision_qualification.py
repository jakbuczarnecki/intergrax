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


def test_gr10_a2_checkpoint_session_coupling_is_migration_target() -> None:
    assert (
        GR10_A2_CHECKPOINT_SESSION_COUPLING
        is Gr10CheckpointSessionCouplingStatus.DEPRECATED_MIGRATION_TARGET
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


def test_gr10_a2_p_acp_session_migration_required_not_second_canonical() -> None:
    acp = next(row for row in GR10_AGENTIC_LEGAL_PRODUCTION_PATHS if row.path_id == "P-ACP-SESSION")
    assert acp.status == "ARCHITECTURAL_MIGRATION_REQUIRED"
    assert "not canonical" in acp.canonical_governance_owner.lower()


def test_gr10_a2_next_remediation_is_a3_implementation() -> None:
    assert "GR-10-A3" in GR10_A3_NEXT_REMEDIATION.task_name
    assert "SESSION_ENABLED" in GR10_A3_NEXT_REMEDIATION.exact_blocker


def test_gr10_a2_enricher_coupling_documented_pending_a3_runtime() -> None:
    """Runtime unchanged in A2; SSOT + ADR forbid treating enricher as execution-mode owner."""
    source = _ENRICHER.read_text(encoding="utf-8-sig")
    assert "SESSION_ENABLED" in source
    adr = _ADR.read_text(encoding="utf-8-sig")
    assert "make_acp_checkpoint_task_enricher" in adr
    assert "DEPRECATED_MIGRATION_TARGET" in adr or "deprecated" in adr.lower()
