"""Register scenario reconciliation plugins on an ERL registry."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityCapabilityKind,
    EnterpriseReliabilityPluginDescriptor,
    EnterpriseReliabilityPluginRegistry,
    ResolutionStrategyEvaluationRequest,
)
from intergrax.contracts.enterprise_reliability.resolution_decision import (
    ResolutionDecision,
    ResolutionPlatformAction,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.constants import (
    SCENARIO_RECONCILIATION_PLUGIN_ID,
    SCENARIO_RECONCILIATION_PLUGIN_OWNER,
    SCENARIO_RECONCILIATION_PLUGIN_VERSION,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealityLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.external_reality_reconciliation import (
    ScenarioExternalRealityReconciliationPlugin,
)


def register_scenario_reconciliation_plugins(
    registry: EnterpriseReliabilityPluginRegistry,
    lookup: ExternalRealityLookupPort,
) -> ScenarioExternalRealityReconciliationPlugin:
    """Install reconciliation probe + minimal resolution strategy for qualification flows."""
    reconcile = ScenarioExternalRealityReconciliationPlugin(_lookup=lookup)
    registry.register(reconcile)
    registry.register(_ScenarioResolutionPlugin())
    return reconcile


@dataclass(frozen=True, slots=True)
class _ScenarioResolutionPlugin:
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
            capabilities=("scenario_default_resolution",),
            tenant_scope=None,
            priority=0,
        )

    def evaluate(
        self,
        request: ResolutionStrategyEvaluationRequest,
    ) -> ResolutionDecision | None:
        return ResolutionDecision(
            action=ResolutionPlatformAction.CONTINUE,
            rationale="scenario_continue_after_reconciliation",
        )
