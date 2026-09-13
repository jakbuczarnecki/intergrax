"""ERL admission through recovery — platform orchestration with scenario plugins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.enterprise_reliability.evidence_evaluation import EvidenceEvaluationOutcome
from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDecision
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPluginGateway,
    EnterpriseReliabilityStrategyContext,
    ResolutionStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    missing_resolution_strategy_decision,
)
from intergrax.runtime.enterprise_reliability import (
    admit_external_effect_unknown,
    admit_external_effect_unknown_with_contract,
    evaluate_external_effect_evidence,
    evaluate_external_effect_governance,
    execute_external_effect_reconciliation_probe,
    plan_external_effect_reconciliation,
    recommend_external_effect_recovery_lifecycle,
)
from intergrax.runtime.enterprise_reliability.reconciliation_execution import (
    ExternalEffectReconciliationProbeRun,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    scenario_external_effect_contract,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_evidence_evaluator import (
    PaymentEvidenceEvaluatorPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.tracing.port import (
    ScenarioExecutionTracePort,
    ScenarioExecutionTraceStepId,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidenceLookupPort,
)


@dataclass(frozen=True, slots=True)
class EnterpriseReliabilityPhaseResult:
    reconciliation_run: ExternalEffectReconciliationProbeRun
    evidence_evaluation_outcome: EvidenceEvaluationOutcome
    resolution_result: ResolutionDecision
    governance_result: GovernanceDecision
    recovery_result: RecoveryDecision


def run_enterprise_reliability_phase(
    *,
    correlation_id: str,
    tenant_id: str,
    gateway: EnterpriseReliabilityPluginGateway,
    payment_evidence_lookup: PaymentReconciliationEvidenceLookupPort,
    execution_trace: ScenarioExecutionTracePort,
    recorded_at: datetime | None = None,
) -> EnterpriseReliabilityPhaseResult:
    """Admission → reconciliation → evidence → resolution → governance → recovery."""
    contract = scenario_external_effect_contract()
    admission = admit_external_effect_unknown_with_contract(
        correlation_id=correlation_id,
        contract=contract,
    )
    execution_trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.RELIABILITY_CASE_CREATED,
        outcome=admission.state.effect_outcome.value,
        component_identity="intergrax.runtime.enterprise_reliability.admission",
        business_detail={"contract_id": admission.contract_id},
    )
    timestamp = recorded_at or datetime.now(tz=UTC)

    planning = plan_external_effect_reconciliation(
        admission=admission,
        contract=contract,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id=tenant_id,
    )
    reconciliation_run = execute_external_effect_reconciliation_probe(
        planning=planning,
        gateway=gateway,
        tenant_id=tenant_id,
        recorded_at=timestamp,
    )
    probe_verdict = None
    if reconciliation_run.execution.probe_result is not None:
        probe_verdict = reconciliation_run.execution.probe_result.verdict.value
    execution_trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.RECONCILIATION_EXECUTED,
        outcome=reconciliation_run.execution.disposition.value,
        component_identity="intergrax.runtime.enterprise_reliability.reconciliation",
        business_detail={
            "probe_verdict": probe_verdict,
            "plugin_id": SCENARIO_RECONCILIATION_PLUGIN_ID,
        },
    )
    evidence = reconciliation_run.evidence
    if evidence is None:
        raise ValueError("reconciliation_probe_missing_evidence")

    unknown_episode_state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluator = PaymentEvidenceEvaluatorPlugin(_lookup=payment_evidence_lookup)
    evidence_evaluation = evaluate_external_effect_evidence(
        state=unknown_episode_state,
        evidence=evidence,
        tenant_id=tenant_id,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        evaluator_strategy=evaluator,
    )
    execution_trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.EVIDENCE_EVALUATED,
        outcome=evidence_evaluation.outcome.value,
        component_identity="erl_integration.plugins.payment_evidence_evaluator",
        business_detail={"evidence_ref": evidence.evidence_ref},
    )

    strategy_context = EnterpriseReliabilityStrategyContext(
        tenant_id=tenant_id,
        correlation_id=correlation_id,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        effect_outcome=reconciliation_run.state.effect_outcome,
        lifecycle_phase=reconciliation_run.state.lifecycle_phase,
        evidence_verdict=evidence.verdict,
        evidence_ref=evidence.evidence_ref,
    )
    resolution_plugin = gateway.evaluate_resolution(
        SCENARIO_RECONCILIATION_PLUGIN_ID,
        ResolutionStrategyEvaluationRequest(
            evidence=evidence,
            execution_context=strategy_context,
            effect_contract=contract,
        ),
    )
    resolution_result = resolution_plugin or missing_resolution_strategy_decision()
    execution_trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.RESOLUTION_DECIDED,
        outcome=resolution_result.action.value,
        component_identity="erl_integration.plugins.payment_resolution_strategy",
        business_detail={"evidence_ref": evidence.evidence_ref},
    )

    recovery_bundle = recommend_external_effect_recovery_lifecycle(
        state=unknown_episode_state,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        effect_contract=contract,
        resolution_decision=resolution_result,
        evidence=evidence,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id=tenant_id,
    )

    governance_evaluation = evaluate_external_effect_governance(
        state=unknown_episode_state,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        effect_contract=contract,
        resolution_decision=resolution_result,
        recovery_decision=recovery_bundle.recovery_decision,
        evidence=evidence,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id=tenant_id,
    )
    execution_trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.GOVERNANCE_EVALUATED,
        outcome=governance_evaluation.governance_decision.disposition.value,
        component_identity="erl_integration.plugins.payment_governance_policy",
        business_detail={},
    )
    execution_trace.emit_lifecycle_step(
        ScenarioExecutionTraceStepId.RECOVERY_EXECUTED,
        outcome=recovery_bundle.recovery_decision.action.value,
        component_identity="erl_integration.plugins.payment_recovery_strategy",
        business_detail={},
    )

    return EnterpriseReliabilityPhaseResult(
        reconciliation_run=reconciliation_run,
        evidence_evaluation_outcome=evidence_evaluation.outcome,
        resolution_result=resolution_result,
        governance_result=governance_evaluation.governance_decision,
        recovery_result=recovery_bundle.recovery_decision,
    )
