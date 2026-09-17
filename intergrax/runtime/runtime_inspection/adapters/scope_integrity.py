# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Validate runtime spine events against inspection execution scope."""

from __future__ import annotations

from intergrax.contracts.execution_scope_identity import ExecutionScopeIdentity
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_inspection.errors import (
    RuntimeInspectionError,
    RuntimeInspectionErrorCode,
    RuntimeInspectionTenantBoundaryError,
)
from intergrax.contracts.runtime_inspection.sources import RuntimeInspectionExecutionScope


def validate_domain_record_scope(
    scope: RuntimeInspectionExecutionScope,
    record: ExecutionScopeIdentity,
    *,
    source_id: str,
    record_label: str,
) -> None:
    if not record.tenant_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            f"{record_label} missing tenant provenance",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if record.tenant_id != scope.tenant_id:
        raise RuntimeInspectionTenantBoundaryError(execution_id=scope.execution_id)
    if record.task_id != scope.task_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            f"{record_label} task_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if record.run_id != scope.run_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            f"{record_label} run_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if scope.attempt_id is not None and record.attempt_id != scope.attempt_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            f"{record_label} attempt_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if record.execution_id != scope.execution_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            f"{record_label} execution_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )


def validate_spine_event_scope(
    scope: RuntimeInspectionExecutionScope,
    event: RuntimeEvent,
    *,
    source_id: str,
) -> None:
    if event.tenant_id is None or not event.tenant_id.strip():
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "spine event missing tenant provenance",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if event.tenant_id != scope.tenant_id:
        raise RuntimeInspectionTenantBoundaryError(execution_id=scope.execution_id)
    if event.task_id != scope.task_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "spine event task_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if event.run_id != scope.run_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "spine event run_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if scope.attempt_id is not None and event.attempt_id != scope.attempt_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "spine event attempt_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )
    if event.execution_id != scope.execution_id:
        raise RuntimeInspectionError(
            RuntimeInspectionErrorCode.SOURCE_INTEGRITY,
            "spine event execution_id mismatch",
            execution_id=scope.execution_id,
            source_id=source_id,
        )


__all__ = ["validate_domain_record_scope", "validate_spine_event_scope"]
