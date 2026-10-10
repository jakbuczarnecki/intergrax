# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R2-P4-R2-R1-R1 P4 integrity qualification matrix (blocker 34)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _IntegrityMatrixRow:
    scenario: str
    expected: str
    test_id: str


P4_INTEGRITY_QUALIFICATION_MATRIX: tuple[_IntegrityMatrixRow, ...] = (
    _IntegrityMatrixRow(
        "tenant mismatch in provenance record",
        "reconstruction fails closed",
        "test_tenant_mismatch_in_record_fails_closed",
    ),
    _IntegrityMatrixRow(
        "ExecutionId mismatch in provenance record",
        "reconstruction fails closed",
        "test_execution_id_mismatch_fails_closed",
    ),
    _IntegrityMatrixRow(
        "document pin row key subject mismatch",
        "read fails closed",
        "test_document_provenance_row_key_subject_mismatch_fail_closed",
    ),
    _IntegrityMatrixRow(
        "invalid/non-UTC requirement recovery staging",
        "staging construction rejected",
        "test_integrity_matrix_timezone_invalid_staging_rejected",
    ),
    _IntegrityMatrixRow(
        "required staging absent on first CONFIGURED_ADOPTED pin",
        "pin reconcile rejected",
        "test_integrity_matrix_first_pin_requires_candidate_staging",
    ),
    _IntegrityMatrixRow(
        "legacy configured pin without staging on retry",
        "pin reconcile rejected",
        "test_integrity_matrix_legacy_pin_without_staging_fails_reconcile",
    ),
    _IntegrityMatrixRow(
        "corrupt provenance codec payload",
        "read/pin fails closed",
        "test_opportunity_unsupported_schema_and_corrupt_record",
    ),
    _IntegrityMatrixRow(
        "requirement spine event without matching pin",
        "reconstruction fails closed",
        "test_required_provenance_missing_fails_closed",
    ),
    _IntegrityMatrixRow(
        "same EventId + changed payload",
        "persistence conflict rejected",
        "test_conflicting_payload_rejected",
    ),
    _IntegrityMatrixRow(
        "same EventId + changed timestamp",
        "persistence conflict rejected",
        "test_conflicting_payload_rejected",
    ),
    _IntegrityMatrixRow(
        "same EventId + changed tenant",
        "cross-tenant event id rejected",
        "test_cross_tenant_same_event_id_rejected",
    ),
    _IntegrityMatrixRow(
        "same EventId + changed task/run/attempt correlation",
        "persistence conflict rejected",
        "test_conflicting_task_rejected",
    ),
    _IntegrityMatrixRow(
        "semantic configured provenance slice invalid",
        "validator rejects record",
        "test_provenance_reject_configured_lookalike",
    ),
    _IntegrityMatrixRow(
        "mandatory spine persistence unavailable",
        "commit returns PERSISTENCE_UNAVAILABLE; no durable fact",
        "test_requirement_persistence_failure_fail_closed",
    ),
    _IntegrityMatrixRow(
        "malformed requirement event payload at reconstruction",
        "spine payload policy rejects invalid fact",
        "test_configured_adopted_requires_slice",
    ),
    _IntegrityMatrixRow(
        "cross-tenant reconstruction scope",
        "wrong tenant lookup empty / fail closed",
        "test_wrong_tenant_lookup_returns_none",
    ),
    _IntegrityMatrixRow(
        "historical current-state mutation",
        "historical projection unchanged after restart",
        "test_historical_restart_ignores_changed_current_configuration_state",
    ),
    _IntegrityMatrixRow(
        "active execution tenant vs configured invocation tenant",
        "staging rejected; no pin/spine/I/O",
        "test_active_execution_tenant_mismatch_rejects_pin_spine_and_io",
    ),
    _IntegrityMatrixRow(
        "KV pin storage tenant partition isolation",
        "foreign tenant read empty",
        "test_cross_tenant_pin_scope_isolation",
    ),
)


def test_p4_integrity_matrix_has_explicit_evidence_for_each_row() -> None:
    assert len(P4_INTEGRITY_QUALIFICATION_MATRIX) >= 17
    for row in P4_INTEGRITY_QUALIFICATION_MATRIX:
        assert row.scenario.strip()
        assert row.expected.strip()
        assert row.test_id.startswith("test_")
