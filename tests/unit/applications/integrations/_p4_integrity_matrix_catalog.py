# © Artur Czarnecki. All rights reserved.

"""P4 integrity qualification matrix catalog (blocker 34 SSOT)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IntegrityMatrixRow:
    scenario: str
    expected: str
    test_module: str
    test_id: str
    status: str


def _row(
    scenario: str,
    expected: str,
    test_module: str,
    test_id: str,
) -> IntegrityMatrixRow:
    return IntegrityMatrixRow(
        scenario=scenario,
        expected=expected,
        test_module=test_module,
        test_id=test_id,
        status="PASS",
    )


P4_INTEGRITY_QUALIFICATION_MATRIX: tuple[IntegrityMatrixRow, ...] = (
    _row(
        "tenant mismatch in provenance record",
        "reconstruction fails closed",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_tenant_mismatch_in_record_fails_closed",
    ),
    _row(
        "ExecutionId mismatch in provenance record",
        "reconstruction fails closed",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_execution_id_mismatch_fails_closed",
    ),
    _row(
        "document pin row key subject mismatch",
        "read fails closed",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p2_persistence.py",
        "test_document_provenance_row_key_subject_mismatch_fail_closed",
    ),
    _row(
        "invalid/non-UTC requirement recovery staging",
        "staging construction rejected",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_integrity_matrix_timezone_invalid_staging_rejected",
    ),
    _row(
        "required staging absent on first CONFIGURED_ADOPTED pin",
        "pin reconcile rejected",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_integrity_matrix_first_pin_requires_candidate_staging",
    ),
    _row(
        "legacy configured pin without staging on retry",
        "pin reconcile rejected",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_integrity_matrix_legacy_pin_without_staging_fails_reconcile",
    ),
    _row(
        "corrupt provenance codec payload",
        "read/pin fails closed",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p2_persistence.py",
        "test_opportunity_unsupported_schema_and_corrupt_record",
    ),
    _row(
        "requirement spine event without matching pin",
        "reconstruction fails closed",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_required_provenance_missing_fails_closed",
    ),
    _row(
        "same EventId + changed payload",
        "persistence conflict rejected",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_conflicting_event_payload_rejected",
    ),
    _row(
        "same EventId + changed timestamp",
        "persistence conflict rejected",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_conflicting_timestamp_rejected",
    ),
    _row(
        "same EventId + changed tenant",
        "cross-tenant event id rejected",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_cross_tenant_same_event_id_rejected",
    ),
    _row(
        "same EventId + changed TaskId correlation",
        "persistence conflict rejected",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_conflicting_task_rejected",
    ),
    _row(
        "same EventId + changed RunId correlation",
        "persistence conflict rejected",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_conflicting_run_rejected",
    ),
    _row(
        "same EventId + changed AttemptId correlation",
        "persistence conflict rejected",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_conflicting_attempt_rejected",
    ),
    _row(
        "semantic configured provenance slice invalid",
        "validator rejects record",
        "tests/unit/contracts/test_execution_integration_configuration_provenance.py",
        "test_provenance_reject_configured_lookalike",
    ),
    _row(
        "mandatory spine persistence unavailable",
        "commit returns PERSISTENCE_UNAVAILABLE; no durable fact",
        "tests/unit/runtime/execution/test_trace_x_p5_r2_p4_r2_requirement_spine.py",
        "test_requirement_persistence_failure_fail_closed",
    ),
    _row(
        "malformed INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED payload at reconstruction",
        "requirement spine payload policy rejects invalid fact",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_requirement_spine_malformed_payload_reconstruction_fails_closed",
    ),
    _row(
        "cross-tenant reconstruction scope",
        "wrong tenant lookup empty / fail closed",
        "tests/unit/runtime/events/test_event_id_persistence_semantics.py",
        "test_wrong_tenant_lookup_returns_none",
    ),
    _row(
        "historical current-state mutation",
        "historical projection unchanged after restart",
        "tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py",
        "test_historical_restart_ignores_changed_current_configuration_state",
    ),
    _row(
        "active execution tenant vs configured invocation tenant",
        "staging rejected; no pin/spine/I/O",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_active_execution_tenant_mismatch_rejects_pin_spine_and_io",
    ),
    _row(
        "KV pin storage tenant partition isolation",
        "foreign tenant read empty",
        "tests/unit/applications/integrations/test_trace_x_p5_r2_p4_r2_r1_production_requirement_wiring.py",
        "test_cross_tenant_pin_scope_isolation",
    ),
)
