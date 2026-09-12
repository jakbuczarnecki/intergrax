"""Scenario payment recovery strategy — ``RecoveryStrategy`` implementation."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    RecoveryStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.recovery_decision import RecoveryDecision

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_EVIDENCE_REF_PREFIX,
    SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID,
    SCENARIO_PAYMENT_RECOVERY_STRATEGY_ID,
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    SCENARIO_RECONCILIATION_PLUGIN_OWNER,
    SCENARIO_RECONCILIATION_PLUGIN_VERSION,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_recovery_action import (
    PaymentRecoveryActionPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.mapping.payment_recovery_decision import (
    decide_payment_recovery,
)


def _applies_to_request(request: RecoveryStrategyEvaluationRequest) -> bool:
    if request.effect_contract.contract_id != SCENARIO_EXTERNAL_EFFECT_CONTRACT_ID:
        return False
    return request.evidence.evidence_ref.startswith(SCENARIO_EVIDENCE_REF_PREFIX)


@dataclass(frozen=True, slots=True)
class PaymentRecoveryStrategyPlugin:
    """
    Replaceable payment recovery policy behind platform ``RecoveryStrategy`` SPI.

    Translates resolution outcomes into enterprise workflow actions via
    ``PaymentRecoveryActionPort`` and returns only platform ``RecoveryDecision``.
    """

    _action_port: PaymentRecoveryActionPort

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
            capability_kind=EnterpriseReliabilityCapabilityKind.RECOVERY,
            capabilities=(SCENARIO_PAYMENT_RECOVERY_STRATEGY_ID,),
            tenant_scope=None,
            priority=0,
        )

    def evaluate(
        self,
        request: RecoveryStrategyEvaluationRequest,
    ) -> RecoveryDecision | None:
        if not _applies_to_request(request):
            return None

        ctx = request.execution_context
        recovery_decision, _execution = decide_payment_recovery(
            resolution_decision=request.resolution_decision,
            action_port=self._action_port,
            tenant_id=ctx.tenant_id,
            correlation_id=ctx.correlation_id,
        )
        return recovery_decision
