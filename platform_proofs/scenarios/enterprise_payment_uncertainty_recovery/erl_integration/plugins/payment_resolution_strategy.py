"""Scenario payment resolution strategy — ``ResolutionStrategy`` implementation."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.evidence import ExternalEffectEvidenceVerdict
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    ResolutionStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    ResolutionPlatformAction,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EVIDENCE_REF_PREFIX,
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_PAYMENT_RESOLUTION_STRATEGY_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    SCENARIO_RECONCILIATION_PLUGIN_OWNER,
    SCENARIO_RECONCILIATION_PLUGIN_VERSION,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.failures import (
    PaymentReconciliationEvidenceLookupError,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidenceLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_resolution_decision import (
    decide_payment_resolution,
)


def _applies_to_request(request: ResolutionStrategyEvaluationRequest) -> bool:
    if request.effect_contract.contract_id != SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID:
        return False
    return request.evidence.evidence_ref.startswith(SCENARIO_EVIDENCE_REF_PREFIX)


@dataclass(frozen=True, slots=True)
class PaymentResolutionStrategyPlugin:
    """
    Replaceable payment business policy behind platform ``ResolutionStrategy`` SPI.

    Registered on the scenario reconciliation plugin id so gateway resolution
    calls use the same bundle as reconciliation probes.
    """

    _lookup: PaymentReconciliationEvidenceLookupPort

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
            capability_kind=EnterpriseReliabilityCapabilityKind.RESOLUTION,
            capabilities=(SCENARIO_PAYMENT_RESOLUTION_STRATEGY_ID,),
            tenant_scope=None,
            priority=0,
        )

    def evaluate(
        self,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None:
        if not _applies_to_request(request):
            return None

        ctx = request.execution_context
        try:
            payment = self._lookup.lookup_by_correlation_id(ctx.correlation_id)
        except PaymentReconciliationEvidenceLookupError:
            if request.evidence.verdict is ExternalEffectEvidenceVerdict.INSUFFICIENT:
                return ResolutionDecision(
                    action=ResolutionPlatformAction.ESCALATE,
                    rationale="payment_truth_unavailable_without_reconciliation_bundle",
                )
            return ResolutionDecision(
                action=ResolutionPlatformAction.ESCALATE,
                rationale="payment_evidence_lookup_failed",
            )

        return decide_payment_resolution(
            evidence=request.evidence,
            payment=payment,
            tenant_id=ctx.tenant_id,
            contract_id=ctx.contract_id,
            correlation_id=ctx.correlation_id,
        )
