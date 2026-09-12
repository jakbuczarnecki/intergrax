# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reconciliation planning orchestration — contracts, lifecycle, plugin gateway."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.effect_contract import (
    ExternalEffectContract,
    UnknownUncertaintyPosture,
)
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPluginGateway,
    EnterpriseReliabilityStrategyContext,
)
from intergrax.contracts.enterprise_reliability.reconciliation import (
    ReconciliationDisposition,
    ReconciliationPlan,
    ReconciliationPlanningError,
    build_reconciliation_plan,
    default_contract_probe_ref,
    evaluate_reconciliation_disposition,
)
from intergrax.runtime.enterprise_reliability.contract_admission import (
    ExternalEffectUnknownAdmission,
)
from intergrax.runtime.enterprise_reliability.uncertainty_lifecycle import (
    advance_uncertainty_lifecycle,
)


class ReconciliationOrchestrationError(ValueError):
    """Admission or plugin advice inconsistent with reconciliation rules."""


class ExternalEffectReconciliationPlanning(BaseModel):
    """UNKNOWN episode prepared for bounded reconcile — plan only, no provider I/O."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    effect_contract: ExternalEffectContract
    unknown_posture: UnknownUncertaintyPosture
    plan: ReconciliationPlan


def _prepare_lifecycle_for_reconciliation(
    state: UncertaintyStateRecord,
) -> UncertaintyStateRecord:
    pending = state
    for target in (
        UncertaintyLifecyclePhase.CONTAINED,
        UncertaintyLifecyclePhase.PENDING_RESOLUTION,
    ):
        if pending.lifecycle_phase is target:
            continue
        pending = advance_uncertainty_lifecycle(pending, target)
    return pending


def _strategy_context(
    *,
    admission: ExternalEffectUnknownAdmission,
    contract: ExternalEffectContract,
    tenant_id: str,
) -> EnterpriseReliabilityStrategyContext:
    return EnterpriseReliabilityStrategyContext(
        tenant_id=tenant_id,
        correlation_id=admission.state.correlation_id,
        contract_id=contract.contract_id,
        effect_outcome=admission.state.effect_outcome,
        lifecycle_phase=UncertaintyLifecyclePhase.PENDING_RESOLUTION,
    )


def plan_external_effect_reconciliation(
    *,
    admission: ExternalEffectUnknownAdmission,
    contract: ExternalEffectContract,
    gateway: EnterpriseReliabilityPluginGateway,
    plugin_id: str,
    tenant_id: str,
) -> ExternalEffectReconciliationPlanning:
    """
    Select reconcile probe via plugin strategy with contract-bound platform defaults.

    Provider I/O runs in Integrations; this step only plans and advances lifecycle.
    """
    if admission.contract_id != contract.contract_id:
        raise ReconciliationOrchestrationError("admission contract_id mismatch")
    if admission.state.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        raise ReconciliationOrchestrationError(
            "reconciliation planning requires UNKNOWN effect outcome",
        )

    disposition = evaluate_reconciliation_disposition(
        unknown_posture=admission.unknown_posture,
        contract=contract,
    )
    if disposition is not ReconciliationDisposition.SCHEDULE_PROBE:
        return ExternalEffectReconciliationPlanning(
            state=admission.state,
            contract_id=contract.contract_id,
            effect_contract=contract,
            unknown_posture=admission.unknown_posture,
            plan=ReconciliationPlan(
                disposition=disposition,
                rationale=disposition.value,
            ),
        )

    prepared = _prepare_lifecycle_for_reconciliation(admission.state)
    context = _strategy_context(
        admission=admission.model_copy(update={"state": prepared}),
        contract=contract,
        tenant_id=tenant_id,
    )
    advice = gateway.evaluate_reconciliation(plugin_id, context)
    try:
        probe_ref = (
            advice.probe_ref
            if advice is not None
            else default_contract_probe_ref(contract)
        )
        plan = build_reconciliation_plan(
            unknown_posture=admission.unknown_posture,
            contract=contract,
            plugin_id=plugin_id,
            probe_ref=probe_ref,
            rationale=advice.rationale if advice is not None else "platform_default_probe",
        )
    except ReconciliationPlanningError as exc:
        raise ReconciliationOrchestrationError(str(exc)) from exc

    return ExternalEffectReconciliationPlanning(
        state=prepared,
        contract_id=contract.contract_id,
        effect_contract=contract,
        unknown_posture=admission.unknown_posture,
        plan=plan,
    )


__all__ = [
    "ExternalEffectReconciliationPlanning",
    "ReconciliationOrchestrationError",
    "plan_external_effect_reconciliation",
]
