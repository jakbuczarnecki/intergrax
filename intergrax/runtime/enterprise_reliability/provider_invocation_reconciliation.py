# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Durable provider UNKNOWN → ERL reconciliation probe orchestration (GR-7-A6)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.plugin_spi import EnterpriseReliabilityPluginGateway
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationPreparation,
    ProviderInvocationReconciliationReason,
    ProviderInvocationReconciliationRequest,
    ProviderInvocationReconciliationResult,
    ProviderInvocationReconciliationVerdict,
    prepare_provider_invocation_reconciliation,
)
from intergrax.contracts.enterprise_reliability.reconciliation import ReconciliationDisposition
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import ExternalEffectEvidence
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ReconciliationExecutionDisposition,
)
from intergrax.runtime.enterprise_reliability.contract_admission import (
    admit_external_effect_unknown_with_contract,
)
from intergrax.runtime.enterprise_reliability.reconciliation_execution import (
    ExternalEffectReconciliationProbeRun,
    execute_external_effect_reconciliation_probe,
)
from intergrax.runtime.enterprise_reliability.reconciliation_orchestration import (
    ReconciliationOrchestrationError,
    plan_external_effect_reconciliation,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityEvidenceObserver,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_evidence import (
    emit_provider_invocation_reliability_fact,
    project_reconciliation_completed,
)


class ProviderInvocationReconciliationRun(BaseModel):
    """Probe bundle for observability — does not mutate provider invocation history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result: ProviderInvocationReconciliationResult
    probe_run: ExternalEffectReconciliationProbeRun | None = None
    evidence: ExternalEffectEvidence | None = None


def reconcile_durable_provider_invocation_unknown(
    request: ProviderInvocationReconciliationRequest,
    *,
    gateway: EnterpriseReliabilityPluginGateway,
    recorded_at: datetime | None = None,
    evidence_observer: ProviderInvocationReliabilityEvidenceObserver | None = None,
) -> ProviderInvocationReconciliationRun:
    """
    Execute one bounded reconciliation probe for a durable UNKNOWN provider attempt.

    Does not repeat provider mutations, recovery, HITL, or host state projection.
    """
    prepared = prepare_provider_invocation_reconciliation(request)
    if isinstance(prepared, ProviderInvocationReconciliationResult):
        run = ProviderInvocationReconciliationRun(result=prepared)
        _emit_reconciliation_evidence(
            request=request,
            run=run,
            recorded_at=recorded_at,
            evidence_observer=evidence_observer,
        )
        return run

    run = _execute_prepared_reconciliation(
        prepared,
        gateway=gateway,
        recorded_at=recorded_at,
    )
    _emit_reconciliation_evidence(
        request=request,
        run=run,
        recorded_at=recorded_at,
        evidence_observer=evidence_observer,
    )
    return run


def _emit_reconciliation_evidence(
    *,
    request: ProviderInvocationReconciliationRequest,
    run: ProviderInvocationReconciliationRun,
    recorded_at: datetime | None,
    evidence_observer: ProviderInvocationReliabilityEvidenceObserver | None,
) -> None:
    if evidence_observer is None or recorded_at is None:
        return
    invocation = request.invocation
    if invocation is None:
        return
    emit_provider_invocation_reliability_fact(
        project_reconciliation_completed(
            invocation=invocation,
            tenant_id=request.tenant_id,
            effect_contract_id=request.effect_contract.contract_id,
            result=run.result,
            probe_run=run.probe_run,
            plugin_id=request.plugin_id,
            recorded_at=recorded_at,
        ),
        evidence_observer,
    )


def _execute_prepared_reconciliation(
    prepared: ProviderInvocationReconciliationPreparation,
    *,
    gateway: EnterpriseReliabilityPluginGateway,
    recorded_at: datetime | None = None,
) -> ProviderInvocationReconciliationRun:
    contract = prepared.effect_contract
    admission = admit_external_effect_unknown_with_contract(
        correlation_id=prepared.correlation_id,
        contract=contract,
    )
    try:
        planning = plan_external_effect_reconciliation(
            admission=admission,
            contract=contract,
            gateway=gateway,
            plugin_id=prepared.plugin_id,
            tenant_id=prepared.tenant_id,
        )
    except ReconciliationOrchestrationError as exc:
        return ProviderInvocationReconciliationRun(
            result=ProviderInvocationReconciliationResult(
                verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
                reason=ProviderInvocationReconciliationReason.SKIPPED_NOT_SCHEDULED,
                invocation_id=prepared.invocation.invocation_id,
                detail=str(exc)[:512],
            ),
        )

    plan = planning.plan
    if plan.disposition is ReconciliationDisposition.ESCALATE_REQUIRED:
        return ProviderInvocationReconciliationRun(
            result=ProviderInvocationReconciliationResult(
                verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
                reason=ProviderInvocationReconciliationReason.ESCALATE_REQUIRED,
                invocation_id=prepared.invocation.invocation_id,
                detail=plan.rationale or plan.disposition.value,
            ),
        )
    if plan.disposition is ReconciliationDisposition.NO_DECLARED_PROBE:
        return ProviderInvocationReconciliationRun(
            result=ProviderInvocationReconciliationResult(
                verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
                reason=ProviderInvocationReconciliationReason.RECONCILIATION_UNSUPPORTED,
                invocation_id=prepared.invocation.invocation_id,
                detail=plan.rationale or plan.disposition.value,
            ),
        )
    if plan.disposition is not ReconciliationDisposition.SCHEDULE_PROBE:
        return ProviderInvocationReconciliationRun(
            result=ProviderInvocationReconciliationResult(
                verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
                reason=ProviderInvocationReconciliationReason.SKIPPED_NOT_SCHEDULED,
                invocation_id=prepared.invocation.invocation_id,
                detail=plan.disposition.value,
            ),
        )

    timestamp = recorded_at or datetime.now(tz=UTC)
    probe_run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id=prepared.tenant_id,
        recorded_at=timestamp,
    )
    result = _result_from_probe_run(
        prepared.invocation.invocation_id,
        probe_run,
    )
    return ProviderInvocationReconciliationRun(
        result=result,
        probe_run=probe_run,
        evidence=probe_run.evidence,
    )


def _result_from_probe_run(
    invocation_id: str,
    probe_run: ExternalEffectReconciliationProbeRun,
) -> ProviderInvocationReconciliationResult:
    execution = probe_run.execution
    if execution.disposition is ReconciliationExecutionDisposition.SKIPPED_NOT_SCHEDULED:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.NOT_AVAILABLE,
            reason=ProviderInvocationReconciliationReason.SKIPPED_NOT_SCHEDULED,
            invocation_id=invocation_id,
            detail=execution.rationale or execution.disposition.value,
        )
    if execution.disposition is ReconciliationExecutionDisposition.PLUGIN_PROBE_UNAVAILABLE:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.PROBE_FAILED,
            reason=ProviderInvocationReconciliationReason.PLUGIN_PROBE_UNAVAILABLE,
            invocation_id=invocation_id,
            detail=execution.rationale or "reconciliation probe executor missing",
        )

    probe_result = execution.probe_result
    if probe_result is None:
        return ProviderInvocationReconciliationResult(
            verdict=ProviderInvocationReconciliationVerdict.PROBE_FAILED,
            reason=ProviderInvocationReconciliationReason.PLUGIN_PROBE_UNAVAILABLE,
            invocation_id=invocation_id,
            detail="probe executed without result",
        )

    verdict = _map_evidence_verdict(probe_result.verdict)
    return ProviderInvocationReconciliationResult(
        verdict=verdict,
        reason=ProviderInvocationReconciliationReason.PROBE_EXECUTED,
        invocation_id=invocation_id,
        evidence_ref=probe_result.evidence_ref,
        detail=probe_result.rationale,
    )


def _map_evidence_verdict(
    verdict: ExternalEffectEvidenceVerdict,
) -> ProviderInvocationReconciliationVerdict:
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        return ProviderInvocationReconciliationVerdict.CONFIRMED_SUCCEEDED
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        return ProviderInvocationReconciliationVerdict.CONFIRMED_FAILED
    return ProviderInvocationReconciliationVerdict.STILL_UNKNOWN


__all__ = [
    "ProviderInvocationReconciliationRun",
    "reconcile_durable_provider_invocation_unknown",
]
