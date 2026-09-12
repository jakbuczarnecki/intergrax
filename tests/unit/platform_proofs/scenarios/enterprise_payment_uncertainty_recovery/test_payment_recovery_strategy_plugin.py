# © Artur Czarnecki. All rights reserved.

"""Payment recovery strategy plugin — SPI, business actions, and orchestration integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    RecoveryLifecycleAction,
    RecoveryStrategy,
    ResolutionPlatformAction,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityStrategyContext,
    RecoveryStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryDecision
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    recommend_external_effect_recovery_lifecycle,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_recovery_action import (
    InMemoryPaymentRecoveryActionPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    scenario_external_effect_contract,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_recovery_action import (
    PaymentRecoveryBusinessAction,
    PaymentRecoveryExecutionStatus,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_recovery_decision import (
    decide_payment_recovery,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_recovery_strategy import (
    PaymentRecoveryStrategyPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.wiring import (
    register_scenario_reconciliation_plugins,
)
from tests.unit.platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.test_reconciliation_plugin_integration import (
    _run_reconciliation,
    _seed_lookup,
)

pytestmark = pytest.mark.unit


def _recovery_request(
    *,
    correlation_id: str,
    variant_id: str,
    resolution_action: ResolutionPlatformAction,
    resolution_rationale: str = "recovery_test",
) -> RecoveryStrategyEvaluationRequest:
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None
    return RecoveryStrategyEvaluationRequest(
        resolution_decision=ResolutionDecision(
            action=resolution_action,
            rationale=resolution_rationale,
        ),
        compensation_execution=None,
        evidence=run.evidence,
        execution_context=EnterpriseReliabilityStrategyContext(
            tenant_id="tenant-lab",
            correlation_id=correlation_id,
            contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
            effect_outcome=run.state.effect_outcome,
            lifecycle_phase=run.state.lifecycle_phase,
            evidence_verdict=run.evidence.verdict,
            evidence_ref=run.evidence.evidence_ref,
        ),
        effect_contract=scenario_external_effect_contract(),
    )


def test_continue_recovery_resumes_fulfillment_with_success() -> None:
    correlation_id = "corr-rec-continue"
    action_port = InMemoryPaymentRecoveryActionPort()
    plugin = PaymentRecoveryStrategyPlugin(_action_port=action_port)
    request = _recovery_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
        resolution_action=ResolutionPlatformAction.CONTINUE,
    )
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.action is RecoveryLifecycleAction.CONTINUE
    assert len(action_port.executions) == 1
    assert action_port.executions[0].action is PaymentRecoveryBusinessAction.RESUME_FULFILLMENT
    assert action_port.executions[0].status is PaymentRecoveryExecutionStatus.SUCCESS


def test_stop_recovery_releases_reservation() -> None:
    correlation_id = "corr-rec-stop"
    action_port = InMemoryPaymentRecoveryActionPort()
    plugin = PaymentRecoveryStrategyPlugin(_action_port=action_port)
    request = _recovery_request(
        correlation_id=correlation_id,
        variant_id="payment_failed_after_unknown",
        resolution_action=ResolutionPlatformAction.STOP,
    )
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.action is RecoveryLifecycleAction.TERMINATE
    assert action_port.executions[0].action is PaymentRecoveryBusinessAction.RELEASE_RESERVATION
    assert action_port.executions[0].status is PaymentRecoveryExecutionStatus.SUCCESS


def test_escalation_recovery_creates_operational_follow_up() -> None:
    correlation_id = "corr-rec-escalate"
    action_port = InMemoryPaymentRecoveryActionPort()
    plugin = PaymentRecoveryStrategyPlugin(_action_port=action_port)
    request = _recovery_request(
        correlation_id=correlation_id,
        variant_id="payment_truth_unavailable",
        resolution_action=ResolutionPlatformAction.ESCALATE,
        resolution_rationale="payment_truth_unavailable",
    )
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.action is RecoveryLifecycleAction.ESCALATE
    assert action_port.executions[0].action is PaymentRecoveryBusinessAction.OPERATIONAL_FOLLOW_UP
    assert action_port.executions[0].status is PaymentRecoveryExecutionStatus.ESCALATED


def test_unresolved_truth_recovery_waits_with_follow_up() -> None:
    correlation_id = "corr-rec-wait"
    action_port = InMemoryPaymentRecoveryActionPort()
    recovery_decision, execution = decide_payment_recovery(
        resolution_decision=ResolutionDecision(
            action=ResolutionPlatformAction.UNKNOWN,
            rationale="resolution_strategy_abstained",
        ),
        action_port=action_port,
        tenant_id="tenant-lab",
        correlation_id=correlation_id,
    )
    assert recovery_decision.action is RecoveryLifecycleAction.WAIT
    assert execution.status is PaymentRecoveryExecutionStatus.WAITING


def test_recommend_recovery_via_gateway_with_payment_bundle() -> None:
    correlation_id = "corr-rec-gateway"
    variant_id = "payment_completed_after_unknown"
    reality_lookup = _seed_lookup(variant_id, correlation_id)
    action_port = InMemoryPaymentRecoveryActionPort()
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    register_scenario_reconciliation_plugins(
        registry,
        reality_lookup,
        payment_recovery_action_port=action_port,
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    _planning, run = _run_reconciliation(reality_lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    recommendation = recommend_external_effect_recovery_lifecycle(
        state=state,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        effect_contract=scenario_external_effect_contract(),
        resolution_decision=ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="payment_confirmed_continue_business_process",
        ),
        evidence=run.evidence,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id="tenant-lab",
    )
    assert recommendation.recovery_decision.action is RecoveryLifecycleAction.CONTINUE
    assert action_port.last_correlation_id == correlation_id


def test_plugin_implements_recovery_strategy_protocol() -> None:
    plugin = PaymentRecoveryStrategyPlugin(_action_port=InMemoryPaymentRecoveryActionPort())
    assert isinstance(plugin, RecoveryStrategy)


def test_plugin_abstains_for_non_scenario_evidence_refs() -> None:
    correlation_id = "corr-rec-abstain"
    request = _recovery_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
        resolution_action=ResolutionPlatformAction.CONTINUE,
    )
    foreign_evidence = request.evidence.model_copy(
        update={"evidence_ref": "evidence://foreign/probe/1"},
    )
    request = RecoveryStrategyEvaluationRequest(
        resolution_decision=request.resolution_decision,
        compensation_execution=None,
        evidence=foreign_evidence,
        execution_context=request.execution_context,
        effect_contract=request.effect_contract,
    )
    plugin = PaymentRecoveryStrategyPlugin(_action_port=InMemoryPaymentRecoveryActionPort())
    assert plugin.evaluate(request) is None


def test_intergrax_has_no_import_dependency_on_payment_recovery_plugin() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    intergrax_root = repo_root / "intergrax"
    needle = "payment_recovery_strategy"
    violations: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        if needle in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(repo_root)))
    assert not violations


def test_recovery_decision_types_are_platform_only() -> None:
    correlation_id = "corr-rec-types"
    request = _recovery_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
        resolution_action=ResolutionPlatformAction.CONTINUE,
    )
    plugin = PaymentRecoveryStrategyPlugin(_action_port=InMemoryPaymentRecoveryActionPort())
    decision = plugin.evaluate(request)
    assert type(decision) is RecoveryDecision
