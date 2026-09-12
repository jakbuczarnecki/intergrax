# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Compensation planning orchestration — resolution decision, evidence, plugin gateway."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.compensation import (
    CompensationPlan,
    CompensationPlanningError,
    build_compensation_plan,
)
from intergrax.contracts.enterprise_reliability.compensation_decision import (
    abstained_compensation_strategy_decision,
    missing_compensation_strategy_decision,
)
from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    CompensationStrategyEvaluationRequest,
    EnterpriseReliabilityPluginGateway,
    EnterpriseReliabilityStrategyContext,
)
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability.reconciliation_evidence import (
    assert_reconciliation_evidence_applicable,
)


class CompensationOrchestrationError(ValueError):
    """Evidence or plugin advice inconsistent with compensation rules."""


class ExternalEffectCompensationPlanning(BaseModel):
    """Resolution mandate with a compensation plan — execution runs in later phases only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    resolution_decision: ResolutionDecision
    evidence: ExternalEffectEvidence
    plan: CompensationPlan


def _compensation_strategy_context(
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


def plan_external_effect_compensation(
    *,
    state: UncertaintyStateRecord,
    contract_id: str,
    effect_contract: ExternalEffectContract,
    resolution_decision: ResolutionDecision,
    evidence: ExternalEffectEvidence,
    gateway: EnterpriseReliabilityPluginGateway,
    plugin_id: str,
    tenant_id: str,
) -> ExternalEffectCompensationPlanning:
    """
    Select compensation advice via plugin strategy after resolution mandates compensation.

    Enqueue and external mutations run in later phases; this step only plans.
    """
    try:
        assert_reconciliation_evidence_applicable(state, evidence)
    except Exception as exc:
        raise CompensationOrchestrationError(str(exc)) from exc
    if evidence.operation_link.contract_id != contract_id:
        raise CompensationOrchestrationError("evidence contract_id mismatch")
    if effect_contract.contract_id != contract_id:
        raise CompensationOrchestrationError("effect_contract contract_id mismatch")

    context = _compensation_strategy_context(
        state=state,
        contract_id=contract_id,
        tenant_id=tenant_id,
        evidence=evidence,
    )
    request = CompensationStrategyEvaluationRequest(
        resolution_decision=resolution_decision,
        evidence=evidence,
        execution_context=context,
        effect_contract=effect_contract,
    )
    strategy_registered = gateway.compensation_strategy_registered(plugin_id)
    plugin_decision = (
        gateway.evaluate_compensation(plugin_id, request) if strategy_registered else None
    )

    if not strategy_registered:
        decision = missing_compensation_strategy_decision()
    elif plugin_decision is None:
        decision = abstained_compensation_strategy_decision()
    else:
        decision = plugin_decision
    try:
        plan = build_compensation_plan(
            resolution_decision=resolution_decision,
            plugin_id=plugin_id,
            decision=decision,
            strategy_registered=strategy_registered,
            rationale=decision.rationale,
        )
    except CompensationPlanningError as exc:
        raise CompensationOrchestrationError(str(exc)) from exc

    return ExternalEffectCompensationPlanning(
        state=state,
        contract_id=contract_id,
        resolution_decision=resolution_decision,
        evidence=evidence,
        plan=plan,
    )


__all__ = [
    "CompensationOrchestrationError",
    "ExternalEffectCompensationPlanning",
    "plan_external_effect_compensation",
]
