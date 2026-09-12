# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recovery lifecycle orchestration — resolution and compensation outcomes, plugin gateway."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionResult,
)
from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPluginGateway,
    EnterpriseReliabilityStrategyContext,
    RecoveryStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
)
from intergrax.contracts.enterprise_reliability.recovery_decision import (
    RecoveryDecision,
    abstained_recovery_strategy_decision,
    missing_recovery_strategy_decision,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability.reconciliation_evidence import (
    assert_reconciliation_evidence_applicable,
)


class RecoveryOrchestrationError(ValueError):
    """Evidence or plugin advice inconsistent with recovery rules."""


class ExternalEffectRecoveryRecommendation(BaseModel):
    """ERL outcome bundle for execution lifecycle handoff and observability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    resolution_decision: ResolutionDecision
    evidence: ExternalEffectEvidence
    compensation_execution: CompensationExecutionResult | None
    recovery_decision: RecoveryDecision


def _recovery_strategy_context(
    *,
    state: UncertaintyStateRecord,
    contract_id: str,
    tenant_id: str,
    evidence: ExternalEffectEvidence,
) -> EnterpriseReliabilityStrategyContext:
    return EnterpriseReliabilityStrategyContext(
        tenant_id=tenant_id,
        correlation_id=state.correlation_id,
        contract_id=contract_id,
        effect_outcome=state.effect_outcome,
        lifecycle_phase=UncertaintyLifecyclePhase.PENDING_RESOLUTION,
        evidence_verdict=evidence.verdict,
        evidence_ref=evidence.evidence_ref,
    )


def recommend_external_effect_recovery_lifecycle(
    *,
    state: UncertaintyStateRecord,
    contract_id: str,
    effect_contract: ExternalEffectContract,
    resolution_decision: ResolutionDecision,
    evidence: ExternalEffectEvidence,
    gateway: EnterpriseReliabilityPluginGateway,
    plugin_id: str,
    tenant_id: str,
    compensation_execution: CompensationExecutionResult | None = None,
) -> ExternalEffectRecoveryRecommendation:
    """
    Select lifecycle recommendation via recovery strategy after resolution and compensation.

    Does not mutate execution lifecycle — execution runtime applies recommendations later.
    """
    try:
        assert_reconciliation_evidence_applicable(state, evidence)
    except Exception as exc:
        raise RecoveryOrchestrationError(str(exc)) from exc
    if evidence.operation_link.contract_id != contract_id:
        raise RecoveryOrchestrationError("evidence contract_id mismatch")
    if effect_contract.contract_id != contract_id:
        raise RecoveryOrchestrationError("effect_contract contract_id mismatch")

    context = _recovery_strategy_context(
        state=state,
        contract_id=contract_id,
        tenant_id=tenant_id,
        evidence=evidence,
    )
    request = RecoveryStrategyEvaluationRequest(
        resolution_decision=resolution_decision,
        compensation_execution=compensation_execution,
        evidence=evidence,
        execution_context=context,
        effect_contract=effect_contract,
    )
    strategy_registered = gateway.recovery_strategy_registered(plugin_id)
    plugin_decision = (
        gateway.evaluate_recovery(plugin_id, request) if strategy_registered else None
    )

    if not strategy_registered:
        recovery_decision = missing_recovery_strategy_decision()
    elif plugin_decision is None:
        recovery_decision = abstained_recovery_strategy_decision()
    else:
        recovery_decision = plugin_decision

    return ExternalEffectRecoveryRecommendation(
        state=state,
        contract_id=contract_id,
        resolution_decision=resolution_decision,
        evidence=evidence,
        compensation_execution=compensation_execution,
        recovery_decision=recovery_decision,
    )


__all__ = [
    "ExternalEffectRecoveryRecommendation",
    "RecoveryOrchestrationError",
    "recommend_external_effect_recovery_lifecycle",
]
