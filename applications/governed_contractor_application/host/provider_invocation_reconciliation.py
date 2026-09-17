# © Artur Czarnecki. All rights reserved.

"""Host composition — durable provider UNKNOWN reconciliation (GR-7-A6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.external_work_reconciliation_plugin import (
    EXTERNAL_WORK_RECONCILIATION_PLUGIN_ID,
    ExternalWorkReconciliationCorrelationRegistry,
    ExternalWorkReconciliationPlugin,
    external_task_correlation_from_invocation,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPlugin,
    EnterpriseReliabilityPluginRegistry,
)
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration
from intergrax.contracts.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRequest,
    provider_invocation_reconciliation_correlation_id,
)
from intergrax.contracts.external_work_provider_capabilities import (
    ExternalWorkProviderCapabilities,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
)
from intergrax.runtime.enterprise_reliability.plugin_gateway import (
    EnterpriseReliabilityPluginGatewayImpl,
)
from intergrax.runtime.enterprise_reliability.plugin_registry import (
    InMemoryEnterpriseReliabilityPluginRegistry,
)
from intergrax.runtime.enterprise_reliability.provider_invocation_reconciliation import (
    ProviderInvocationReconciliationRun,
    reconcile_durable_provider_invocation_unknown,
)

_ACTION_FOR_OPERATION = {
    "external_work.create_work": ACTION_CREATE_EXTERNAL_WORK,
    "external_work.accept_quote": ACTION_ACCEPT_QUOTE,
    "external_work.cancel_work": ACTION_CANCEL_EXTERNAL_WORK,
}


@dataclass(frozen=True, slots=True)
class GovernedExternalWorkProviderReconciliation:
    """Injectable reconciliation stack for governed external-work host."""

    gateway: EnterpriseReliabilityPluginGatewayImpl
    correlation_registry: ExternalWorkReconciliationCorrelationRegistry
    plugin_id: str = EXTERNAL_WORK_RECONCILIATION_PLUGIN_ID

    @classmethod
    def build(
        cls,
        integration: ExternalWorkIntegration,
        *,
        resolution_plugin: EnterpriseReliabilityPlugin | None = None,
    ) -> GovernedExternalWorkProviderReconciliation:
        registry: EnterpriseReliabilityPluginRegistry = (
            InMemoryEnterpriseReliabilityPluginRegistry()
        )
        correlation_registry = ExternalWorkReconciliationCorrelationRegistry()
        registry.register(
            ExternalWorkReconciliationPlugin(
                _integration=integration,
                _correlation_registry=correlation_registry,
            ),
        )
        if resolution_plugin is not None:
            registry.register(resolution_plugin)
        return cls(
            gateway=EnterpriseReliabilityPluginGatewayImpl(registry),
            correlation_registry=correlation_registry,
        )

    def reconcile_unknown(
        self,
        *,
        invocation: ProviderInvocation,
        outcome: ProviderInvocationOutcome,
        capabilities: ExternalWorkProviderCapabilities,
        tenant_id: str,
        recorded_at: datetime | None = None,
    ) -> ProviderInvocationReconciliationRun:
        action = _ACTION_FOR_OPERATION.get(invocation.operation)
        if action is None:
            raise ValueError(f"unsupported provider operation: {invocation.operation}")
        contract = external_work_effect_contract_for_action(action, capabilities)
        correlation_id = provider_invocation_reconciliation_correlation_id(invocation)
        external_task_id = invocation.external_task_id
        if external_task_id is not None and external_task_id.strip():
            self.correlation_registry.bind(
                correlation_id=correlation_id,
                contract_id=contract.contract_id,
                correlation=external_task_correlation_from_invocation(
                    task_id=invocation.task_id,
                    run_id=invocation.run_id,
                    provider_id=invocation.provider_id,
                    external_task_id=external_task_id.strip(),
                    correlation_id=invocation.correlation_id,
                    idempotency_key=invocation.idempotency_key,
                ),
            )
        request = ProviderInvocationReconciliationRequest(
            invocation=invocation,
            outcome=outcome,
            effect_contract=contract,
            tenant_id=tenant_id,
            plugin_id=self.plugin_id,
            provider_id=invocation.provider_id,
        )
        return reconcile_durable_provider_invocation_unknown(
            request,
            gateway=self.gateway,
            recorded_at=recorded_at,
        )


__all__ = [
    "GovernedExternalWorkProviderReconciliation",
]
