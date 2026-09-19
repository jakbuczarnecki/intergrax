# © Artur Czarnecki. All rights reserved.

"""Builders mapping memory domain results to diagnostic events (MEM-ENT-12)."""

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import (
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_execution_task_id,
)
from intergrax.memory.contracts.memory_control import MemoryControlScopeRef
from intergrax.memory.contracts.memory_lifecycle import (
    MemoryLifecycleDisposition,
    MemoryLifecycleOutcome,
    MemoryProjectionFailureCategory,
    MemoryProjectionOperationEvidence,
    MemoryReconciliationDisposition,
    MemoryReconciliationOutcome,
)
from intergrax.memory.contracts.memory_observability import (
    MemoryDiagnosticComponent,
    MemoryDiagnosticCounts,
    MemoryDiagnosticEvent,
    MemoryDiagnosticExecutionCorrelation,
    MemoryDiagnosticFailureClass,
    MemoryDiagnosticOperation,
    MemoryDiagnosticOutcome,
    MemoryDiagnosticPhase,
)
from intergrax.memory.contracts.memory_security_governance import (
    MemoryGovernanceDecision,
    MemoryGovernanceEvaluationRequest,
    MemoryGovernanceOutcome,
    MemoryGovernanceReasonCode,
)
from intergrax.memory.contracts.long_horizon_memory import CompactionResult
from intergrax.memory.memory_diagnostic_emitter import MemoryDiagnosticEmitter
from intergrax.memory.contracts.memory_lifecycle import MemoryProjectionFailureEvidence

__all__ = [
    "emit_control_plane_terminal",
    "emit_governance_diagnostic",
    "emit_lifecycle_terminal",
    "emit_reconciliation_terminal",
    "emit_compaction_terminal",
    "emit_entity_projection_terminal",
    "emit_procedural_terminal",
    "optional_active_execution_correlation",
    "resolve_diagnostic_execution_correlation",
    "scope_identity_fields",
    "governance_diagnostic_outcome",
    "governance_failure_class",
]


def optional_active_execution_correlation() -> MemoryDiagnosticExecutionCorrelation | None:
    """Read bound canonical execution identity; never mints identifiers."""
    bound = peek_active_execution_identity()
    if bound is None:
        return None
    run_id, attempt_id = bound
    return MemoryDiagnosticExecutionCorrelation(
        task_id=peek_active_execution_task_id(),
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=peek_active_execution_id(),
    )


def resolve_diagnostic_execution_correlation(
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None,
) -> MemoryDiagnosticExecutionCorrelation:
    if execution_correlation is not None:
        return execution_correlation
    active = optional_active_execution_correlation()
    if active is not None:
        return active
    return MemoryDiagnosticExecutionCorrelation()


def _execution_correlation_fields(
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None,
) -> MemoryDiagnosticExecutionCorrelation:
    return resolve_diagnostic_execution_correlation(execution_correlation)


def _event_execution_identity_kwargs(
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None,
) -> MemoryDiagnosticExecutionCorrelation:
    return _execution_correlation_fields(execution_correlation)


def scope_identity_fields(
    identity: RequestIdentity,
    scope: MemoryControlScopeRef | None,
) -> tuple[str | None, str | None, str | None]:
    tenant_id = identity.tenant_id
    user_id = identity.user_id
    workspace_id = None
    if scope is not None:
        user_id = scope.user_id or user_id
    return tenant_id, user_id, workspace_id


def governance_diagnostic_outcome(decision: MemoryGovernanceDecision) -> MemoryDiagnosticOutcome:
    if decision.reason_code is MemoryGovernanceReasonCode.POLICY_MISSING:
        return MemoryDiagnosticOutcome.FAILED
    if decision.reason_code is MemoryGovernanceReasonCode.POLICY_FAILURE:
        return MemoryDiagnosticOutcome.FAILED
    if decision.outcome is MemoryGovernanceOutcome.DENY:
        return MemoryDiagnosticOutcome.DENIED
    if decision.outcome is MemoryGovernanceOutcome.REQUIRE_REVIEW:
        return MemoryDiagnosticOutcome.REVIEW_REQUIRED
    return MemoryDiagnosticOutcome.SUCCESS


def governance_failure_class(
    decision: MemoryGovernanceDecision,
) -> MemoryDiagnosticFailureClass | None:
    code = decision.reason_code
    if code in {
        MemoryGovernanceReasonCode.POLICY_FAILURE,
        MemoryGovernanceReasonCode.POLICY_MISSING,
    }:
        return MemoryDiagnosticFailureClass.POLICY
    if code is MemoryGovernanceReasonCode.AUTHORIZATION_DENY:
        return MemoryDiagnosticFailureClass.AUTHORIZATION
    if code in {
        MemoryGovernanceReasonCode.GOVERNANCE_DENY,
        MemoryGovernanceReasonCode.SENSITIVE_DATA_POLICY,
        MemoryGovernanceReasonCode.RETENTION_BLOCK,
    }:
        return MemoryDiagnosticFailureClass.POLICY
    if code is MemoryGovernanceReasonCode.CROSS_SCOPE:
        return MemoryDiagnosticFailureClass.AUTHORIZATION
    if decision.outcome is MemoryGovernanceOutcome.DENY:
        return MemoryDiagnosticFailureClass.POLICY
    return None


def emit_control_plane_terminal(
    emitter: MemoryDiagnosticEmitter,
    *,
    identity: RequestIdentity,
    scope: MemoryControlScopeRef,
    operation: MemoryDiagnosticOperation,
    outcome: MemoryDiagnosticOutcome,
    memory_id: str | None = None,
    revision: int | None = None,
    failure_class: MemoryDiagnosticFailureClass | None = None,
    duration_seconds: float | None = None,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    tenant_id, user_id, workspace_id = scope_identity_fields(identity, scope)
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=operation,
            phase=MemoryDiagnosticPhase.TERMINAL,
            outcome=outcome,
            component=MemoryDiagnosticComponent.CONTROL_PLANE,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
            memory_id=memory_id,
            revision=revision,
            failure_class=failure_class,
            duration_seconds=duration_seconds,
        )
    )


def emit_entity_projection_terminal(
    emitter: MemoryDiagnosticEmitter,
    *,
    tenant_id: str,
    user_id: str | None,
    workspace_id: str | None,
    operation: MemoryDiagnosticOperation,
    outcome: MemoryDiagnosticOutcome,
    memory_id: str | None = None,
    revision: int | None = None,
    projection_id: str | None = None,
    failure_class: MemoryDiagnosticFailureClass | None = None,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=operation,
            phase=MemoryDiagnosticPhase.PROJECTION,
            outcome=outcome,
            component=MemoryDiagnosticComponent.ENTITY_TEMPORAL,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
            memory_id=memory_id,
            revision=revision,
            projection_id=projection_id,
            failure_class=failure_class,
        )
    )


def emit_procedural_terminal(
    emitter: MemoryDiagnosticEmitter,
    *,
    identity: RequestIdentity,
    tenant_id: str,
    user_id: str | None,
    workspace_id: str | None,
    operation: MemoryDiagnosticOperation,
    outcome: MemoryDiagnosticOutcome,
    memory_id: str | None = None,
    failure_class: MemoryDiagnosticFailureClass | None = None,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=operation,
            phase=MemoryDiagnosticPhase.TERMINAL,
            outcome=outcome,
            component=MemoryDiagnosticComponent.PROCEDURAL,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
            memory_id=memory_id,
            failure_class=failure_class,
        )
    )


def emit_governance_diagnostic(
    emitter: MemoryDiagnosticEmitter,
    request: MemoryGovernanceEvaluationRequest,
    decision: MemoryGovernanceDecision,
    *,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    tenant_id, user_id, workspace_id = scope_identity_fields(
        request.context.identity,
        request.context.scope,
    )
    memory_id = decision.subject_memory_id
    if memory_id is None and request.proposed_record is not None:
        memory_id = request.proposed_record.memory_id
    elif memory_id is None and request.target is not None:
        memory_id = request.target.memory_id
    revision = None
    if request.proposed_record is not None:
        revision = request.proposed_record.revision
    elif request.target is not None:
        revision = request.target.revision
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=MemoryDiagnosticOperation.GOVERNANCE_EVALUATE,
            phase=MemoryDiagnosticPhase.GOVERNANCE,
            outcome=governance_diagnostic_outcome(decision),
            component=MemoryDiagnosticComponent.GOVERNANCE,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
            memory_id=memory_id,
            revision=revision,
            governance_operation=request.context.operation,
            policy_id=decision.policy_id,
            reason_code=decision.reason_code,
            failure_class=governance_failure_class(decision),
        )
    )


def _projection_failure_class(
    failure: MemoryProjectionFailureEvidence | None,
) -> MemoryDiagnosticFailureClass | None:
    if failure is None:
        return None
    if failure.category is MemoryProjectionFailureCategory.PERMANENT:
        return MemoryDiagnosticFailureClass.PROJECTION
    return MemoryDiagnosticFailureClass.PROJECTION


def emit_lifecycle_terminal(
    emitter: MemoryDiagnosticEmitter,
    *,
    user_id: str,
    tenant_id: str | None,
    outcome: MemoryLifecycleOutcome,
    duration_seconds: float | None = None,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    if outcome.disposition is MemoryLifecycleDisposition.PARTIAL_PROJECTION_FAILURE:
        diagnostic_outcome = MemoryDiagnosticOutcome.PARTIAL
    elif outcome.disposition is MemoryLifecycleDisposition.PRIMARY_FAILED:
        diagnostic_outcome = MemoryDiagnosticOutcome.FAILED
    else:
        diagnostic_outcome = MemoryDiagnosticOutcome.SUCCESS
    memory_id = outcome.memory_entity_ids[0] if outcome.memory_entity_ids else None
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=_lifecycle_operation_to_diagnostic(outcome.operation),
            phase=MemoryDiagnosticPhase.TERMINAL,
            outcome=diagnostic_outcome,
            component=MemoryDiagnosticComponent.LIFECYCLE,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            memory_id=memory_id,
            duration_seconds=duration_seconds,
        )
    )
    for evidence in outcome.projection_evidence:
        if evidence.succeeded:
            continue
        emitter.emit(
            MemoryDiagnosticEvent(
                event_id=emitter.new_event_id(),
                reference_time_iso=emitter.reference_time_iso(),
                operation=_projection_op_to_diagnostic(evidence.operation),
                phase=MemoryDiagnosticPhase.PROJECTION,
                outcome=MemoryDiagnosticOutcome.FAILED,
                component=MemoryDiagnosticComponent.LIFECYCLE,
                task_id=execution.task_id,
                run_id=execution.run_id,
                attempt_id=execution.attempt_id,
                execution_id=execution.execution_id,
                tenant_id=tenant_id,
                user_id=user_id,
                memory_id=memory_id,
                projection_id=evidence.projection_id,
                failure_class=_projection_failure_class(evidence.failure),
            )
        )


def _lifecycle_operation_to_diagnostic(operation: object) -> MemoryDiagnosticOperation:
    from intergrax.memory.contracts.memory_lifecycle import MemoryLifecycleOperation

    mapping = {
        MemoryLifecycleOperation.WRITE: MemoryDiagnosticOperation.REMEMBER,
        MemoryLifecycleOperation.UPDATE: MemoryDiagnosticOperation.UPDATE,
        MemoryLifecycleOperation.DELETE_ENTRY: MemoryDiagnosticOperation.DELETE,
        MemoryLifecycleOperation.CLEAR: MemoryDiagnosticOperation.DELETE,
        MemoryLifecycleOperation.DELETE_PROFILE: MemoryDiagnosticOperation.DELETE,
    }
    if isinstance(operation, MemoryLifecycleOperation):
        return mapping.get(operation, MemoryDiagnosticOperation.UPDATE)
    return MemoryDiagnosticOperation.UPDATE


def _projection_op_to_diagnostic(operation: object) -> MemoryDiagnosticOperation:
    from intergrax.memory.contracts.memory_lifecycle import MemoryProjectionOperation

    if operation is MemoryProjectionOperation.DELETE:
        return MemoryDiagnosticOperation.PROJECTION_DELETE
    return MemoryDiagnosticOperation.PROJECTION_WRITE


def emit_reconciliation_terminal(
    emitter: MemoryDiagnosticEmitter,
    *,
    tenant_id: str | None,
    user_id: str,
    outcome: MemoryReconciliationOutcome,
    duration_seconds: float | None = None,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    if outcome.disposition is MemoryReconciliationDisposition.FAILED:
        diagnostic_outcome = MemoryDiagnosticOutcome.FAILED
    elif outcome.disposition is MemoryReconciliationDisposition.REPAIRED:
        diagnostic_outcome = MemoryDiagnosticOutcome.SUCCESS
    else:
        diagnostic_outcome = MemoryDiagnosticOutcome.SUCCESS
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=MemoryDiagnosticOperation.RECONCILE,
            phase=MemoryDiagnosticPhase.RECONCILIATION,
            outcome=diagnostic_outcome,
            component=MemoryDiagnosticComponent.RECONCILIATION,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            duration_seconds=duration_seconds,
        )
    )
    for evidence in outcome.projection_evidence:
        if evidence.succeeded:
            if outcome.disposition is MemoryReconciliationDisposition.REPAIRED:
                emitter.emit(
                    MemoryDiagnosticEvent(
                        event_id=emitter.new_event_id(),
                        reference_time_iso=emitter.reference_time_iso(),
                        operation=MemoryDiagnosticOperation.RECONCILE,
                        phase=MemoryDiagnosticPhase.RECONCILIATION,
                        outcome=MemoryDiagnosticOutcome.SUCCESS,
                        component=MemoryDiagnosticComponent.RECONCILIATION,
                        task_id=execution.task_id,
                        run_id=execution.run_id,
                        attempt_id=execution.attempt_id,
                        execution_id=execution.execution_id,
                        tenant_id=tenant_id,
                        user_id=user_id,
                        projection_id=evidence.projection_id,
                    )
                )
            continue
        failure_class = MemoryDiagnosticFailureClass.RECONCILIATION
        if evidence.failure is not None and "unsupported" in evidence.failure.message.lower():
            failure_class = MemoryDiagnosticFailureClass.UNSUPPORTED
        emitter.emit(
            MemoryDiagnosticEvent(
                event_id=emitter.new_event_id(),
                reference_time_iso=emitter.reference_time_iso(),
                operation=MemoryDiagnosticOperation.RECONCILE,
                phase=MemoryDiagnosticPhase.RECONCILIATION,
                outcome=MemoryDiagnosticOutcome.FAILED,
                component=MemoryDiagnosticComponent.RECONCILIATION,
                task_id=execution.task_id,
                run_id=execution.run_id,
                attempt_id=execution.attempt_id,
                execution_id=execution.execution_id,
                tenant_id=tenant_id,
                user_id=user_id,
                projection_id=evidence.projection_id,
                failure_class=failure_class,
            )
        )


def emit_compaction_terminal(
    emitter: MemoryDiagnosticEmitter,
    *,
    tenant_id: str | None,
    user_id: str | None,
    result: CompactionResult,
    duration_seconds: float | None = None,
    execution_correlation: MemoryDiagnosticExecutionCorrelation | None = None,
) -> None:
    failures = len(result.failures)
    created = len(result.created) + len(result.updated)
    if failures and created:
        outcome = MemoryDiagnosticOutcome.PARTIAL
    elif failures:
        outcome = MemoryDiagnosticOutcome.FAILED
    else:
        outcome = MemoryDiagnosticOutcome.SUCCESS
    execution = _event_execution_identity_kwargs(execution_correlation)
    emitter.emit(
        MemoryDiagnosticEvent(
            event_id=emitter.new_event_id(),
            reference_time_iso=emitter.reference_time_iso(),
            operation=MemoryDiagnosticOperation.COMPACT,
            phase=MemoryDiagnosticPhase.TERMINAL,
            outcome=outcome,
            component=MemoryDiagnosticComponent.LONG_HORIZON,
            task_id=execution.task_id,
            run_id=execution.run_id,
            attempt_id=execution.attempt_id,
            execution_id=execution.execution_id,
            tenant_id=tenant_id,
            user_id=user_id,
            duration_seconds=duration_seconds,
            counts=MemoryDiagnosticCounts(
                summaries_created=created,
                failures=failures,
            ),
        )
    )
