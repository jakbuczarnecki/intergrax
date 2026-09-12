"""Scenario payment governance policy — ``GovernanceStrategy`` implementation."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.governance_decision import GovernanceDecision
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    GovernanceStrategyEvaluationRequest,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EVIDENCE_REF_PREFIX,
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_PAYMENT_GOVERNANCE_STRATEGY_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    SCENARIO_RECONCILIATION_PLUGIN_OWNER,
    SCENARIO_RECONCILIATION_PLUGIN_VERSION,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    PaymentGovernanceContextLookupError,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentEnterpriseGovernancePolicy,
    PaymentGovernanceBusinessContext,
    PaymentGovernanceBusinessContextLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_governance_decision import (
    decide_payment_governance,
)


def _applies_to_request(request: GovernanceStrategyEvaluationRequest) -> bool:
    if request.effect_contract.contract_id != SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID:
        return False
    return request.evidence.evidence_ref.startswith(SCENARIO_EVIDENCE_REF_PREFIX)


@dataclass(frozen=True, slots=True)
class PaymentGovernancePolicyPlugin:
    """
    Replaceable payment governance policy behind platform ``GovernanceStrategy`` SPI.

    Evaluates enterprise thresholds and risk metadata; does not execute payments
    or interact with reconciliation / evidence collection internals.
    """

    _lookup: PaymentGovernanceBusinessContextLookupPort
    _policy: PaymentEnterpriseGovernancePolicy

    @property
    def plugin_id(self) -> str:
        return SCENARIO_RECONCILIATION_PLUGIN_ID

    @property
    def version(self) -> str:
        return SCENARIO_RECONCILIATION_PLUGIN_VERSION

    @property
    def descriptor(self) -> EnterpriseReliabilityPluginDescriptor:
        return EnterpriseReliabilityPluginDescriptor(
            plugin_id=SCENARIO_RECONCILIATION_PLUGIN_ID,
            version=SCENARIO_RECONCILIATION_PLUGIN_VERSION,
            owner=SCENARIO_RECONCILIATION_PLUGIN_OWNER,
            capability_kind=EnterpriseReliabilityCapabilityKind.GOVERNANCE,
            capabilities=(SCENARIO_PAYMENT_GOVERNANCE_STRATEGY_ID,),
            tenant_scope=None,
            priority=0,
        )

    def evaluate(
        self,
        request: GovernanceStrategyEvaluationRequest,
    ) -> GovernanceDecision | None:
        if not _applies_to_request(request):
            return None

        ctx = request.execution_context
        try:
            business_context = self._lookup.lookup_by_correlation_id(ctx.correlation_id)
        except PaymentGovernanceContextLookupError:
            business_context = PaymentGovernanceBusinessContext(
                correlation_id=ctx.correlation_id,
                payment_amount=None,
                currency="",
                customer_risk_tier=None,
            )

        return decide_payment_governance(
            business_context=business_context,
            policy=self._policy,
            resolution_action=request.resolution_decision.action,
            tenant_id=ctx.tenant_id,
            contract_id=ctx.contract_id,
            correlation_id=ctx.correlation_id,
        )
