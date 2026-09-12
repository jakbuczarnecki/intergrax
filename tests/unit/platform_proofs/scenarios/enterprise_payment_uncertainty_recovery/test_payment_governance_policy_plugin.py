# © Artur Czarnecki. All rights reserved.

"""Payment governance policy plugin — SPI, policy rules, and orchestration integration."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from intergrax.contracts.enterprise_reliability import (
    GovernanceDisposition,
    GovernanceStrategy,
    RecoveryDecision,
    RecoveryLifecycleAction,
    ResolutionPlatformAction,
)
from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDecision
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityStrategyContext,
    GovernanceStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import ResolutionDecision
from intergrax.runtime.enterprise_reliability import (
    EnterpriseReliabilityPluginGatewayImpl,
    InMemoryEnterpriseReliabilityPluginRegistry,
    admit_external_effect_unknown,
    evaluate_external_effect_governance,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_governance_context_lookup import (
    InMemoryPaymentGovernanceBusinessContextLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    scenario_external_effect_contract,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentEnterpriseGovernancePolicy,
    PaymentGovernanceBusinessContext,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_governance_policy import (
    PaymentGovernancePolicyPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.wiring import (
    register_scenario_reconciliation_plugins,
)
from tests.unit.platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.test_reconciliation_plugin_integration import (
    _run_reconciliation,
    _seed_lookup,
)

pytestmark = pytest.mark.unit

_DEFAULT_POLICY = PaymentEnterpriseGovernancePolicy(
    human_approval_threshold_amount=Decimal("10000.00"),
    currency="EUR",
)


def _governance_plugin(
    *,
    correlation_id: str,
    amount: Decimal | None,
    currency: str = "EUR",
) -> PaymentGovernancePolicyPlugin:
    lookup = InMemoryPaymentGovernanceBusinessContextLookup()
    if amount is not None:
        lookup.seed(
            PaymentGovernanceBusinessContext(
                correlation_id=correlation_id,
                payment_amount=amount,
                currency=currency,
            ),
        )
    return PaymentGovernancePolicyPlugin(_lookup=lookup, _policy=_DEFAULT_POLICY)


def _governance_request(
    *,
    correlation_id: str,
    variant_id: str,
    resolution_action: ResolutionPlatformAction = ResolutionPlatformAction.CONTINUE,
) -> GovernanceStrategyEvaluationRequest:
    lookup = _seed_lookup(variant_id, correlation_id)
    _planning, run = _run_reconciliation(lookup, correlation_id=correlation_id)
    assert run.evidence is not None
    return GovernanceStrategyEvaluationRequest(
        recovery_decision=RecoveryDecision(
            action=RecoveryLifecycleAction.CONTINUE,
            rationale="resume_after_erl",
        ),
        resolution_decision=ResolutionDecision(
            action=resolution_action,
            rationale="resolution_for_governance_test",
        ),
        compensation_execution=None,
        compensation_decision=None,
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


def test_standard_payment_governance_allows_continuation() -> None:
    correlation_id = "corr-gov-standard"
    request = _governance_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
    )
    plugin = _governance_plugin(correlation_id=correlation_id, amount=Decimal("2500.00"))
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.disposition is GovernanceDisposition.ALLOW
    assert "within_auto_allow" in decision.rationale


def test_high_value_payment_requires_approval() -> None:
    correlation_id = "corr-gov-high"
    request = _governance_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
    )
    plugin = _governance_plugin(correlation_id=correlation_id, amount=Decimal("12500.00"))
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.disposition is GovernanceDisposition.APPROVAL_REQUIRED
    assert decision.hitl_requirement is not None
    assert "high_value" in decision.hitl_requirement.requirement_ref


def test_missing_governance_information_fails_closed() -> None:
    correlation_id = "corr-gov-missing"
    request = _governance_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
    )
    plugin = _governance_plugin(correlation_id=correlation_id, amount=None)
    decision = plugin.evaluate(request)
    assert decision is not None
    assert decision.disposition is GovernanceDisposition.APPROVAL_REQUIRED
    assert decision.hitl_requirement is not None


def test_evaluate_governance_via_gateway_with_payment_bundle() -> None:
    correlation_id = "corr-gov-gateway"
    variant_id = "payment_completed_after_unknown"
    reality_lookup = _seed_lookup(variant_id, correlation_id)
    governance_lookup = InMemoryPaymentGovernanceBusinessContextLookup()
    governance_lookup.seed(
        PaymentGovernanceBusinessContext(
            correlation_id=correlation_id,
            payment_amount=Decimal("12500.00"),
            currency="EUR",
        ),
    )
    registry = InMemoryEnterpriseReliabilityPluginRegistry()
    register_scenario_reconciliation_plugins(
        registry,
        reality_lookup,
        payment_governance_lookup=governance_lookup,
        payment_governance_policy=_DEFAULT_POLICY,
    )
    gateway = EnterpriseReliabilityPluginGatewayImpl(registry)
    _planning, run = _run_reconciliation(reality_lookup, correlation_id=correlation_id)
    assert run.evidence is not None

    state = admit_external_effect_unknown(correlation_id=correlation_id)
    evaluation = evaluate_external_effect_governance(
        state=state,
        contract_id=SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
        effect_contract=scenario_external_effect_contract(),
        resolution_decision=ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="continue_after_reconciliation",
        ),
        recovery_decision=RecoveryDecision(
            action=RecoveryLifecycleAction.CONTINUE,
            rationale="resume",
        ),
        evidence=run.evidence,
        gateway=gateway,
        plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
        tenant_id="tenant-lab",
    )
    assert evaluation.governance_decision.disposition is GovernanceDisposition.APPROVAL_REQUIRED


def test_plugin_implements_governance_strategy_protocol() -> None:
    plugin = _governance_plugin(correlation_id="corr-proto", amount=Decimal("1.00"))
    assert isinstance(plugin, GovernanceStrategy)


def test_plugin_abstains_for_non_scenario_evidence_refs() -> None:
    correlation_id = "corr-gov-abstain"
    request = _governance_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
    )
    foreign_evidence = request.evidence.model_copy(
        update={"evidence_ref": "evidence://foreign/probe/1"},
    )
    request = GovernanceStrategyEvaluationRequest(
        recovery_decision=request.recovery_decision,
        resolution_decision=request.resolution_decision,
        compensation_execution=None,
        compensation_decision=None,
        evidence=foreign_evidence,
        execution_context=request.execution_context,
        effect_contract=request.effect_contract,
    )
    plugin = _governance_plugin(correlation_id=correlation_id, amount=Decimal("2500.00"))
    assert plugin.evaluate(request) is None


def test_intergrax_has_no_import_dependency_on_payment_governance_plugin() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    intergrax_root = repo_root / "intergrax"
    needle = "payment_governance_policy"
    violations: list[str] = []
    for path in intergrax_root.rglob("*.py"):
        if needle in path.read_text(encoding="utf-8"):
            violations.append(str(path.relative_to(repo_root)))
    assert not violations


def test_governance_decision_types_are_platform_only() -> None:
    correlation_id = "corr-gov-types"
    request = _governance_request(
        correlation_id=correlation_id,
        variant_id="payment_completed_after_unknown",
    )
    plugin = _governance_plugin(correlation_id=correlation_id, amount=Decimal("2500.00"))
    decision = plugin.evaluate(request)
    assert type(decision) is GovernanceDecision
