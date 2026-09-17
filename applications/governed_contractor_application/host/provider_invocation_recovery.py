# © Artur Czarnecki. All rights reserved.

"""Host composition — provider invocation controlled recovery (GR-7-A7)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRequest,
    provider_invocation_reconciliation_correlation_id,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryDispatchState,
    ProviderInvocationRecoveryPolicy,
    ProviderInvocationRecoveryRequest,
)
from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
    ExternalEffectRepeatEligibilityRequest,
    ExternalEffectRepeatEligibilityResult,
    ExternalEffectRepeatPolicy,
    evaluate_external_effect_repeat_eligibility,
)
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.runtime.enterprise_reliability.default_provider_invocation_recovery_policy import (
    default_fail_closed_provider_invocation_recovery_policy,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionPorts,
    ProviderInvocationRecoveryExecutionResult,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)

from applications.governed_contractor_application.host.provider_invocation_reconciliation import (
    GovernedExternalWorkProviderReconciliation,
)

_ACTION_FOR_OPERATION = {
    "external_work.create_work": ACTION_CREATE_EXTERNAL_WORK,
    "create_work": ACTION_CREATE_EXTERNAL_WORK,
    "external_work.accept_quote": ACTION_ACCEPT_QUOTE,
    "submit_quote_acceptance": ACTION_ACCEPT_QUOTE,
    "external_work.cancel_work": ACTION_CANCEL_EXTERNAL_WORK,
    "cancel_work": ACTION_CANCEL_EXTERNAL_WORK,
}


@dataclass(frozen=True, slots=True)
class GovernedExternalWorkProviderRecovery:
    """Recovery decision + execution stack for governed external-work host."""

    reconciliation_stack: GovernedExternalWorkProviderReconciliation
    recovery_policy: ProviderInvocationRecoveryPolicy

    @classmethod
    def build(
        cls,
        reconciliation_stack: GovernedExternalWorkProviderReconciliation,
        *,
        recovery_policy: ProviderInvocationRecoveryPolicy | None = None,
    ) -> GovernedExternalWorkProviderRecovery:
        return cls(
            reconciliation_stack=reconciliation_stack,
            recovery_policy=(
                recovery_policy
                if recovery_policy is not None
                else default_fail_closed_provider_invocation_recovery_policy()
            ),
        )

    def _contract_for(self, invocation: ProviderInvocation, capabilities: ExternalWorkProviderCapabilities):
        action = _ACTION_FOR_OPERATION.get(invocation.operation)
        if action is None:
            raise ValueError(f"unsupported provider operation: {invocation.operation}")
        contract = external_work_effect_contract_for_action(action, capabilities)
        if contract.operation_key != invocation.operation:
            return contract.model_copy(update={"operation_key": invocation.operation})
        return contract

    def build_recovery_request(
        self,
        *,
        invocation: ProviderInvocation,
        outcome: ProviderInvocationOutcome | None,
        capabilities: ExternalWorkProviderCapabilities,
        repeat_eligibility: ExternalEffectRepeatEligibilityResult | None = None,
        repeat_policy: ExternalEffectRepeatPolicy | None = None,
    ) -> ProviderInvocationRecoveryRequest:
        contract = self._contract_for(invocation, capabilities)
        eligibility = repeat_eligibility
        if eligibility is None and outcome is not None:
            eligibility = evaluate_external_effect_repeat_eligibility(
                ExternalEffectRepeatEligibilityRequest(
                    invocation=invocation,
                    outcome=outcome,
                    effect_contract=contract,
                ),
                policy=repeat_policy,
            )

        dispatch_state = ProviderInvocationRecoveryDispatchState.OUTCOME_RECORDED
        if outcome is None:
            dispatch_state = ProviderInvocationRecoveryDispatchState.CRASH_AMBIGUITY

        return ProviderInvocationRecoveryRequest(
            invocation=invocation,
            outcome=outcome,
            dispatch_state=dispatch_state,
            effect_contract=contract,
            repeat_eligibility=eligibility,
        )

    def decide(
        self,
        request: ProviderInvocationRecoveryRequest,
    ) -> ProviderInvocationRecoveryDecision:
        return decide_provider_invocation_recovery(
            request,
            policy=self.recovery_policy,
        )

    def execute(
        self,
        request: ProviderInvocationRecoveryRequest,
        *,
        decision: ProviderInvocationRecoveryDecision,
        tenant_id: str,
        ports: ProviderInvocationRecoveryExecutionPorts,
        recorded_at: datetime | None = None,
    ) -> ProviderInvocationRecoveryExecutionResult:
        _ = recorded_at
        from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
            ProviderInvocationRecoveryAction,
        )

        reconciliation_request: ProviderInvocationReconciliationRequest | None = None
        invocation = request.invocation
        outcome = request.outcome
        if (
            invocation is not None
            and outcome is not None
            and decision.action is ProviderInvocationRecoveryAction.RECONCILE
        ):
            reconciliation_request = ProviderInvocationReconciliationRequest(
                invocation=invocation,
                outcome=outcome,
                effect_contract=request.effect_contract,
                tenant_id=tenant_id,
                plugin_id=self.reconciliation_stack.plugin_id,
                provider_id=invocation.provider_id,
            )
            correlation_id = provider_invocation_reconciliation_correlation_id(invocation)
            external_task_id = invocation.external_task_id
            if external_task_id is not None and external_task_id.strip():
                from external_contractor_adapter.external_work_reconciliation_plugin import (
                    external_task_correlation_from_invocation,
                )

                self.reconciliation_stack.correlation_registry.bind(
                    correlation_id=correlation_id,
                    contract_id=request.effect_contract.contract_id,
                    correlation=external_task_correlation_from_invocation(
                        task_id=invocation.task_id,
                        run_id=invocation.run_id,
                        provider_id=invocation.provider_id,
                        external_task_id=external_task_id.strip(),
                        correlation_id=invocation.correlation_id,
                        idempotency_key=invocation.idempotency_key,
                    ),
                )
        execution_ports = ProviderInvocationRecoveryExecutionPorts(
            gateway=(
                ports.gateway
                if ports.gateway is not None
                else self.reconciliation_stack.gateway
            ),
            repeat=ports.repeat,
            hitl=ports.hitl,
        )
        return execute_provider_invocation_recovery(
            request,
            decision=decision,
            ports=execution_ports,
            reconciliation_request=reconciliation_request,
        )


__all__ = [
    "GovernedExternalWorkProviderRecovery",
]
