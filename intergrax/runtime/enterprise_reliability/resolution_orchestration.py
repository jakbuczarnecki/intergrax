# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolution planning orchestration — evidence, lifecycle, plugin gateway."""

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
    ResolutionStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.reconciliation_evidence import (
    ExternalEffectEvidence,
)
from intergrax.contracts.enterprise_reliability.resolution import (
    ResolutionDisposition,
    ResolutionPlanningError,
    ResolutionPlan,
    build_resolution_plan,
    evaluate_resolution_disposition,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    abstained_resolution_decision,
    missing_resolution_strategy_decision,
)
from intergrax.contracts.enterprise_reliability.evidence_evaluation import (
    EvidenceEvaluationOutcome,
    EvidenceEvaluationResult,
)
from intergrax.runtime.enterprise_reliability.evidence_evaluation import (
    evaluate_external_effect_evidence,
)
from intergrax.runtime.enterprise_reliability.reconciliation_evidence import (
    assert_reconciliation_evidence_applicable,
)


class ResolutionOrchestrationError(ValueError):
    """Evidence or plugin advice inconsistent with resolution rules."""


class ExternalEffectResolutionPlanning(BaseModel):
    """UNKNOWN episode with a resolution plan — closure runs in execution only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: UncertaintyStateRecord
    contract_id: str
    unknown_posture: UnknownUncertaintyPosture
    evidence: ExternalEffectEvidence
    evidence_evaluation: EvidenceEvaluationResult
    plan: ResolutionPlan


def _resolution_strategy_context(
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


def plan_external_effect_resolution(
    *,
    state: UncertaintyStateRecord,
    contract_id: str,
    effect_contract: ExternalEffectContract,
    unknown_posture: UnknownUncertaintyPosture,
    evidence: ExternalEffectEvidence,
    gateway: EnterpriseReliabilityPluginGateway,
    plugin_id: str,
    tenant_id: str,
) -> ExternalEffectResolutionPlanning:
    """
    Select resolution advice via plugin strategy with evidence-bound platform defaults.

    Recovery actions (resume, compensate, HITL) run in later phases; this step only plans.
    """
    if state.effect_outcome is not ExternalEffectOutcome.UNKNOWN:
        raise ResolutionOrchestrationError(
            "resolution planning requires UNKNOWN effect outcome",
        )
    try:
        assert_reconciliation_evidence_applicable(state, evidence)
    except Exception as exc:
        raise ResolutionOrchestrationError(str(exc)) from exc
    if evidence.operation_link.contract_id != contract_id:
        raise ResolutionOrchestrationError("evidence contract_id mismatch")
    if effect_contract.contract_id != contract_id:
        raise ResolutionOrchestrationError("effect_contract contract_id mismatch")

    evidence_evaluation = evaluate_external_effect_evidence(
        state=state,
        evidence=evidence,
        tenant_id=tenant_id,
        contract_id=contract_id,
    )
    if evidence_evaluation.outcome is EvidenceEvaluationOutcome.CONFLICTING_EVIDENCE:
        raise ResolutionOrchestrationError(
            evidence_evaluation.rationale or "conflicting_evidence",
        )
    if evidence_evaluation.outcome is EvidenceEvaluationOutcome.EVALUATION_FAILED:
        raise ResolutionOrchestrationError(
            evidence_evaluation.rationale or "evaluation_failed",
        )
    if evidence_evaluation.outcome is EvidenceEvaluationOutcome.INSUFFICIENT_EVIDENCE:
        try:
            plan = build_resolution_plan(
                unknown_posture=unknown_posture,
                evidence=evidence,
                plugin_id=plugin_id,
                decision=abstained_resolution_decision(),
                strategy_registered=gateway.resolution_strategy_registered(plugin_id),
            )
        except ResolutionPlanningError as exc:
            raise ResolutionOrchestrationError(str(exc)) from exc
        return ExternalEffectResolutionPlanning(
            state=state,
            contract_id=contract_id,
            unknown_posture=unknown_posture,
            evidence=evidence,
            evidence_evaluation=evidence_evaluation,
            plan=plan,
        )

    posture_disposition = evaluate_resolution_disposition(
        unknown_posture=unknown_posture,
        evidence=evidence,
    )
    if posture_disposition is not ResolutionDisposition.INVOKE_PLUGIN:
        try:
            plan = build_resolution_plan(
                unknown_posture=unknown_posture,
                evidence=evidence,
                plugin_id=plugin_id,
                decision=abstained_resolution_decision(),
                strategy_registered=gateway.resolution_strategy_registered(plugin_id),
            )
        except ResolutionPlanningError as exc:
            raise ResolutionOrchestrationError(str(exc)) from exc
        return ExternalEffectResolutionPlanning(
            state=state,
            contract_id=contract_id,
            unknown_posture=unknown_posture,
            evidence=evidence,
            evidence_evaluation=evidence_evaluation,
            plan=plan,
        )

    context = _resolution_strategy_context(
        state=state,
        contract_id=contract_id,
        tenant_id=tenant_id,
        evidence=evidence,
    )
    request = ResolutionStrategyEvaluationRequest(
        evidence=evidence,
        execution_context=context,
        effect_contract=effect_contract,
    )
    strategy_registered = gateway.resolution_strategy_registered(plugin_id)
    plugin_decision = (
        gateway.evaluate_resolution(plugin_id, request) if strategy_registered else None
    )

    if not strategy_registered:
        decision = missing_resolution_strategy_decision()
    elif plugin_decision is None:
        decision = abstained_resolution_decision()
    else:
        decision = plugin_decision
    try:
        plan = build_resolution_plan(
            unknown_posture=unknown_posture,
            evidence=evidence,
            plugin_id=plugin_id,
            decision=decision,
            strategy_registered=strategy_registered,
            rationale=decision.rationale,
        )
    except ResolutionPlanningError as exc:
        raise ResolutionOrchestrationError(str(exc)) from exc

    return ExternalEffectResolutionPlanning(
        state=state,
        contract_id=contract_id,
        unknown_posture=unknown_posture,
        evidence=evidence,
        evidence_evaluation=evidence_evaluation,
        plan=plan,
    )


__all__ = [
    "ExternalEffectResolutionPlanning",
    "ResolutionOrchestrationError",
    "plan_external_effect_resolution",
]
