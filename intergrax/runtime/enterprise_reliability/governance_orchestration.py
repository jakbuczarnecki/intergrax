# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Governance evaluation orchestration — recovery outcomes, plugin gateway, fail closed."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.contracts.enterprise_reliability.compensation_decision import CompensationDecision
from intergrax.contracts.enterprise_reliability.compensation_execution import (
    CompensationExecutionResult,
)
from intergrax.contracts.enterprise_reliability.effect_contract import ExternalEffectContract
from intergrax.contracts.enterprise_reliability.governance_decision import (
    GovernanceDecision,
    abstained_governance_strategy_decision,
    invalid_governance_strategy_decision,
    missing_governance_strategy_decision,
)
from intergrax.contracts.enterprise_reliability.lifecycle import (
    UncertaintyLifecyclePhase,
    UncertaintyStateRecord,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPluginGateway,
    EnterpriseReliabilityStrategyContext,
    GovernanceStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
)
from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability.reconciliation_evidence import (
    assert_reconciliation_evidence_applicable,
)


class GovernanceOrchestrationError(ValueError):
    """Evidence or plugin advice inconsistent with governance rules."""


class ExternalEffectGovernanceEvaluation(BaseModel):
    """ERL outcome bundle for execution handoff and observability."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    resolution_decision: ResolutionDecision
    recovery_decision: RecoveryDecision
    evidence: ExternalEffectEvidence
    compensation_execution: CompensationExecutionResult | None
    compensation_decision: CompensationDecision | None
    governance_decision: GovernanceDecision


def _governance_strategy_context(
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


def evaluate_external_effect_governance(
    *,
    state: UncertaintyStateRecord,
    contract_id: str,
    effect_contract: ExternalEffectContract,
    resolution_decision: ResolutionDecision,
    recovery_decision: RecoveryDecision,
    evidence: ExternalEffectEvidence,
    gateway: EnterpriseReliabilityPluginGateway,
    plugin_id: str,
    tenant_id: str,
    compensation_execution: CompensationExecutionResult | None = None,
    compensation_decision: CompensationDecision | None = None,
) -> ExternalEffectGovernanceEvaluation:
    """
    Evaluate whether proposed lifecycle execution may proceed automatically.

    Does not execute actions, approve requests, or mutate execution lifecycle.
    """
    try:
        assert_reconciliation_evidence_applicable(state, evidence)
    except Exception as exc:
        raise GovernanceOrchestrationError(str(exc)) from exc
    if evidence.operation_link.contract_id != contract_id:
        raise GovernanceOrchestrationError("evidence contract_id mismatch")
    if effect_contract.contract_id != contract_id:
        raise GovernanceOrchestrationError("effect_contract contract_id mismatch")

    context = _governance_strategy_context(
        state=state,
        contract_id=contract_id,
        tenant_id=tenant_id,
        evidence=evidence,
    )
    request = GovernanceStrategyEvaluationRequest(
        recovery_decision=recovery_decision,
        resolution_decision=resolution_decision,
        compensation_execution=compensation_execution,
        compensation_decision=compensation_decision,
        evidence=evidence,
        execution_context=context,
        effect_contract=effect_contract,
    )
    strategy_registered = gateway.governance_strategy_registered(plugin_id)
    plugin_decision = (
        gateway.evaluate_governance(plugin_id, request) if strategy_registered else None
    )

    fail_closed_identity = {
        "tenant_id": tenant_id,
        "correlation_id": state.correlation_id,
        "contract_id": contract_id,
    }
    if not strategy_registered:
        governance_decision = missing_governance_strategy_decision(**fail_closed_identity)
    elif plugin_decision is None:
        governance_decision = abstained_governance_strategy_decision(**fail_closed_identity)
    elif type(plugin_decision) is not GovernanceDecision:
        governance_decision = invalid_governance_strategy_decision(**fail_closed_identity)
    else:
        governance_decision = plugin_decision

    return ExternalEffectGovernanceEvaluation(
        state=state,
        contract_id=contract_id,
        resolution_decision=resolution_decision,
        recovery_decision=recovery_decision,
        evidence=evidence,
        compensation_execution=compensation_execution,
        compensation_decision=compensation_decision,
        governance_decision=governance_decision,
    )


__all__ = [
    "ExternalEffectGovernanceEvaluation",
    "GovernanceOrchestrationError",
    "evaluate_external_effect_governance",
]
