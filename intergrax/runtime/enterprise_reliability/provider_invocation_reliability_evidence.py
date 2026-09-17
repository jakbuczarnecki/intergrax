# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical provider-invocation reliability evidence projection (GR-7-A8)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationResult,
    provider_invocation_reconciliation_correlation_id,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryRequest,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityCorrelation,
    ProviderInvocationReliabilityEvidenceObserver,
    ProviderInvocationReliabilityFact,
    ProviderInvocationReliabilityTracePhase,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityResult,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.logging import IntergraxLogging
from intergrax.runtime.enterprise_reliability.reconciliation_execution import (
    ExternalEffectReconciliationProbeRun,
)

if TYPE_CHECKING:
    from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
        ProviderInvocationRecoveryExecutionResult,
    )

_logger = IntergraxLogging.get_logger(__name__, component="enterprise_reliability")


def correlation_from_invocation(
    invocation: ProviderInvocation,
    *,
    tenant_id: str,
    effect_contract_id: str | None = None,
    execution_id: str | None = None,
    attempt_id: str | None = None,
    governance_execution_id: str | None = None,
) -> ProviderInvocationReliabilityCorrelation:
    return ProviderInvocationReliabilityCorrelation(
        tenant_id=tenant_id,
        provider_id=invocation.provider_id,
        operation=invocation.operation,
        invocation_id=invocation.invocation_id,
        task_id=invocation.task_id,
        run_id=invocation.run_id,
        execution_id=execution_id,
        attempt_id=attempt_id,
        idempotency_key=invocation.idempotency_key,
        effect_contract_id=effect_contract_id,
        governance_execution_id=governance_execution_id,
        erl_correlation_id=provider_invocation_reconciliation_correlation_id(invocation),
    )


def emit_provider_invocation_reliability_fact(
    fact: ProviderInvocationReliabilityFact,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    if observer is None:
        return
    try:
        observer.observe_provider_invocation_reliability_fact(fact)
    except Exception:
        _logger.exception(
            "provider invocation reliability evidence observer failed",
            extra={
                "phase": fact.phase.value,
                "invocation_id": fact.correlation.invocation_id,
                "tenant_id": fact.correlation.tenant_id,
            },
        )


def project_governance_authorized(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    governance_execution_id: str,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.GOVERNANCE_AUTHORIZED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            governance_execution_id=governance_execution_id,
        ),
    )


def project_intent_persisted(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str | None,
    recorded_at: datetime,
    execution_id: str | None = None,
    attempt_id: str | None = None,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.INTENT_PERSISTED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            execution_id=execution_id,
            attempt_id=attempt_id,
        ),
    )


def project_intent_persistence_failed(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    recorded_at: datetime,
    detail: str,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.INTENT_PERSISTENCE_FAILED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(invocation, tenant_id=tenant_id),
        provider_mutation_attempted=False,
        provider_mutation_count=0,
        detail=detail[:512],
    )


def project_dispatch_attempted(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str | None,
    recorded_at: datetime,
    provider_mutation_attempted: bool,
    execution_id: str | None = None,
    attempt_id: str | None = None,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.DISPATCH_ATTEMPTED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            execution_id=execution_id,
            attempt_id=attempt_id,
        ),
        provider_mutation_attempted=provider_mutation_attempted,
    )


def project_outcome_persisted(
    *,
    invocation: ProviderInvocation,
    outcome: ProviderInvocationOutcome,
    tenant_id: str,
    effect_contract_id: str | None,
    recorded_at: datetime,
    execution_id: str | None = None,
) -> ProviderInvocationReliabilityFact:
    phase = ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTED
    if outcome.status is ProviderInvocationStatus.UNKNOWN:
        phase = ProviderInvocationReliabilityTracePhase.UNKNOWN_ADMITTED
    return ProviderInvocationReliabilityFact(
        phase=phase,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            execution_id=execution_id,
        ),
        invocation_status=outcome.status,
    )


def project_outcome_persistence_failed(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    recorded_at: datetime,
    detail: str,
    execution_id: str | None = None,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTENCE_FAILED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            execution_id=execution_id,
        ),
        detail=detail[:512],
    )


def project_crash_ambiguity(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str | None,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.CRASH_AMBIGUITY_ADMITTED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        dispatch_state=ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY,
        detail="intent without durable outcome",
    )


def project_repeat_eligibility(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    eligibility: ExternalEffectRepeatEligibilityResult,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.REPEAT_ELIGIBILITY_EVALUATED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        repeat_eligibility_verdict=eligibility.verdict,
        repeat_eligibility_reason=eligibility.reason,
        repeat_policy_id=eligibility.policy_id,
        unknown_posture=eligibility.unknown_posture,
        detail=eligibility.detail,
    )


def project_reconciliation_completed(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    result: ProviderInvocationReconciliationResult,
    probe_run: ExternalEffectReconciliationProbeRun | None,
    plugin_id: str,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    probe_ref: str | None = None
    evidence_ref = result.evidence_ref
    if probe_run is not None and probe_run.attempt_fact is not None:
        probe_ref = probe_run.attempt_fact.probe_ref
        if evidence_ref is None:
            evidence_ref = probe_run.attempt_fact.evidence_ref
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.RECONCILIATION_COMPLETED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        reconciliation_verdict=result.verdict,
        reconciliation_reason=result.reason,
        reconciliation_plugin_id=plugin_id,
        reconciliation_probe_ref=probe_ref,
        evidence_ref=evidence_ref,
        provider_mutation_count=0,
        detail=result.detail,
    )


def project_recovery_decided(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    decision: ProviderInvocationRecoveryDecision,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.RECOVERY_DECIDED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        recovery_action=decision.action,
        recovery_reason=decision.reason,
        recovery_policy_id=decision.policy_id,
        unknown_posture=decision.unknown_posture,
        detail=decision.detail,
    )


def project_recovery_execution(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    execution: ProviderInvocationRecoveryExecutionResult,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    block_reason = (
        execution.block_reason.value if execution.block_reason is not None else None
    )
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.RECOVERY_EXECUTION_COMPLETED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        recovery_action=execution.decision.action,
        recovery_reason=execution.decision.reason,
        recovery_policy_id=execution.decision.policy_id,
        recovery_execution_disposition=execution.disposition.value,
        recovery_block_reason=block_reason,
        provider_mutation_count=execution.provider_mutation_count,
        detail=execution.detail,
    )


def project_repeat_attempt_linked(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    repeat_invocation_id: str,
    recorded_at: datetime,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.REPEAT_ATTEMPT_LINKED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        repeat_invocation_id=repeat_invocation_id,
    )


def project_hitl_escalated(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    continuation_request_id: str | None,
    recorded_at: datetime,
    detail: str,
) -> ProviderInvocationReliabilityFact:
    return ProviderInvocationReliabilityFact(
        phase=ProviderInvocationReliabilityTracePhase.HITL_ESCALATED,
        recorded_at=recorded_at,
        correlation=correlation_from_invocation(
            invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
        ),
        continuation_request_id=continuation_request_id,
        provider_mutation_count=0,
        detail=detail[:512],
    )


def project_recovery_decision_evidence(
    *,
    request: ProviderInvocationRecoveryRequest,
    tenant_id: str,
    decision: ProviderInvocationRecoveryDecision,
    recorded_at: datetime,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    """Emit trace facts for repeat eligibility, crash ambiguity, and recovery decision."""
    invocation = request.invocation
    if invocation is None:
        return
    contract_id = request.effect_contract.contract_id
    eligibility = request.repeat_eligibility
    if eligibility is not None:
        emit_provider_invocation_reliability_fact(
            project_repeat_eligibility(
                invocation=invocation,
                tenant_id=tenant_id,
                effect_contract_id=contract_id,
                eligibility=eligibility,
                recorded_at=recorded_at,
            ),
            observer,
        )
    if request.dispatch_state is ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY:
        emit_provider_invocation_reliability_fact(
            project_crash_ambiguity(
                invocation=invocation,
                tenant_id=tenant_id,
                effect_contract_id=contract_id,
                recorded_at=recorded_at,
            ),
            observer,
        )
    emit_provider_invocation_reliability_fact(
        project_recovery_decided(
            invocation=invocation,
            tenant_id=tenant_id,
            effect_contract_id=contract_id,
            decision=decision,
            recorded_at=recorded_at,
        ),
        observer,
    )


def project_recovery_execution_evidence(
    *,
    invocation: ProviderInvocation,
    tenant_id: str,
    effect_contract_id: str,
    execution: ProviderInvocationRecoveryExecutionResult,
    reconciliation_plugin_id: str | None,
    recorded_at: datetime,
    observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    """Emit trace facts for recovery execution, reconciliation, repeat, and HITL."""
    emit_provider_invocation_reliability_fact(
        project_recovery_execution(
            invocation=invocation,
            tenant_id=tenant_id,
            effect_contract_id=effect_contract_id,
            execution=execution,
            recorded_at=recorded_at,
        ),
        observer,
    )
    reconciliation = execution.reconciliation
    if reconciliation is not None and reconciliation_plugin_id is not None:
        emit_provider_invocation_reliability_fact(
            project_reconciliation_completed(
                invocation=invocation,
                tenant_id=tenant_id,
                effect_contract_id=effect_contract_id,
                result=reconciliation.result,
                probe_run=reconciliation.probe_run,
                plugin_id=reconciliation_plugin_id,
                recorded_at=recorded_at,
            ),
            observer,
        )
    if execution.repeat is not None:
        emit_provider_invocation_reliability_fact(
            project_repeat_attempt_linked(
                invocation=invocation,
                tenant_id=tenant_id,
                effect_contract_id=effect_contract_id,
                repeat_invocation_id=execution.repeat.repeat_invocation_id,
                recorded_at=recorded_at,
            ),
            observer,
        )
    if execution.hitl is not None:
        emit_provider_invocation_reliability_fact(
            project_hitl_escalated(
                invocation=invocation,
                tenant_id=tenant_id,
                effect_contract_id=effect_contract_id,
                continuation_request_id=execution.hitl.governed_continuation_request_id,
                recorded_at=recorded_at,
                detail=execution.hitl.escalation.recovery_reason.value,
            ),
            observer,
        )


__all__ = [
    "correlation_from_invocation",
    "emit_provider_invocation_reliability_fact",
    "project_crash_ambiguity",
    "project_dispatch_attempted",
    "project_governance_authorized",
    "project_hitl_escalated",
    "project_intent_persisted",
    "project_intent_persistence_failed",
    "project_outcome_persisted",
    "project_outcome_persistence_failed",
    "project_recovery_decided",
    "project_recovery_execution",
    "project_recovery_decision_evidence",
    "project_recovery_execution_evidence",
    "project_repeat_attempt_linked",
    "project_repeat_eligibility",
    "project_reconciliation_completed",
]
