# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconciliation probe execution — plugin gateway, evidence, lifecycle handoff."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.evidence import (
    ExternalEffectEvidenceVerdict,
    classify_external_effect_outcome,
)
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyResolutionKind,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.observability import ReconciliationAttemptFact
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.plugin_spi import EnterpriseReliabilityPluginGateway
from intergrax.contracts.enterprise_reliability.reconciliation import ReconciliationDisposition
from intergrax.contracts.enterprise_reliability.reconciliation_execution import (
    ExternalEffectReconciliationExecution,
    ReconciliationExecutionDisposition,
    ReconciliationExecutionError,
    ReconciliationProbeRequest,
    ReconciliationProbeResult,
    build_reconciliation_probe_request,
)
from intergrax.runtime.enterprise_reliability.reconciliation_orchestration import (
    ExternalEffectReconciliationPlanning,
    ReconciliationOrchestrationError,
)
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import resolve_uncertainty


class ExternalEffectReconciliationProbeRun(BaseModel):
    """Probe execution bundle for downstream resolution and observability emission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    execution: ExternalEffectReconciliationExecution
    attempt_fact: ReconciliationAttemptFact | None = None


def _resolution_kind_for_verdict(
    verdict: ExternalEffectEvidenceVerdict,
) -> UncertaintyResolutionKind | None:
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_SUCCESS:
        return UncertaintyResolutionKind.CONFIRMED_SUCCESS
    if verdict is ExternalEffectEvidenceVerdict.DEFINITIVE_FAILURE:
        return UncertaintyResolutionKind.CONFIRMED_FAILURE
    return None


def execute_external_effect_reconciliation_probe(
    *,
    planning: ExternalEffectReconciliationPlanning,
    gateway: EnterpriseReliabilityPluginGateway,
    tenant_id: str,
    attempt_index: int = 1,
    recorded_at: datetime | None = None,
) -> ExternalEffectReconciliationProbeRun:
    """
    Run one declared reconcile probe through the plugin gateway.

    Provider I/O lives in plugin implementations; core records verdict and evidence ref.
    """
    plan = planning.plan
    if plan.disposition is not ReconciliationDisposition.SCHEDULE_PROBE:
        return ExternalEffectReconciliationProbeRun(
            state=planning.state,
            execution=ExternalEffectReconciliationExecution(
                disposition=ReconciliationExecutionDisposition.SKIPPED_NOT_SCHEDULED,
                plan=plan,
                rationale=plan.disposition.value,
            ),
        )

    assert plan.plugin_id is not None
    try:
        probe_request = build_reconciliation_probe_request(
            plan=plan,
            tenant_id=tenant_id,
            correlation_id=planning.state.correlation_id,
            contract_id=planning.contract_id,
            attempt_index=attempt_index,
        )
    except ReconciliationExecutionError as exc:
        raise ReconciliationOrchestrationError(str(exc)) from exc

    probe_result = gateway.execute_reconciliation_probe(plan.plugin_id, probe_request)
    if probe_result is None:
        return ExternalEffectReconciliationProbeRun(
            state=planning.state,
            execution=ExternalEffectReconciliationExecution(
                disposition=ReconciliationExecutionDisposition.PLUGIN_PROBE_UNAVAILABLE,
                plan=plan,
                probe_request=probe_request,
                rationale="reconciliation_probe_executor_missing",
            ),
        )

    state = _apply_probe_evidence(planning.state, probe_result)
    timestamp = recorded_at or datetime.now(tz=UTC)
    attempt_fact = ReconciliationAttemptFact(
        correlation_id=planning.state.correlation_id,
        contract_id=planning.contract_id,
        probe_ref=probe_request.probe_ref,
        plugin_id=plan.plugin_id,
        attempt_index=attempt_index,
        verdict=probe_result.verdict,
        evidence_ref=probe_result.evidence_ref,
        lifecycle_phase=state.lifecycle_phase,
        recorded_at=timestamp,
    )
    return ExternalEffectReconciliationProbeRun(
        state=state,
        execution=ExternalEffectReconciliationExecution(
            disposition=ReconciliationExecutionDisposition.PROBE_EXECUTED,
            plan=plan,
            probe_request=probe_request,
            probe_result=probe_result,
            rationale=probe_result.rationale,
        ),
        attempt_fact=attempt_fact,
    )


def _apply_probe_evidence(
    state: UncertaintyStateRecord,
    probe_result: ReconciliationProbeResult,
) -> UncertaintyStateRecord:
    if state.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        raise ReconciliationOrchestrationError(
            "probe execution requires UNKNOWN effect outcome",
        )
    resolution_kind = _resolution_kind_for_verdict(probe_result.verdict)
    if resolution_kind is None:
        return state
    resolved_outcome = classify_external_effect_outcome(probe_result.verdict)
    return resolve_uncertainty(
        state,
        resolution_kind=resolution_kind,
        resolved_outcome=resolved_outcome,
    )


__all__ = [
    "ExternalEffectReconciliationProbeRun",
    "execute_external_effect_reconciliation_probe",
]
