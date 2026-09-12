"""Register scenario reconciliation plugins on an ERL registry."""

from __future__ import annotations

from decimal import Decimal

from intergrax.contracts.enterprise_reliability.plugin_spi import (
    EnterpriseReliabilityPluginRegistry,
)

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.adapters.in_memory_payment_evidence_lookup import (
    InMemoryPaymentReconciliationEvidenceLookup,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealityLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentEnterpriseGovernancePolicy,
    PaymentGovernanceBusinessContextLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidenceLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.external_reality_reconciliation import (
    ScenarioExternalRealityReconciliationPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_governance_policy import (
    PaymentGovernancePolicyPlugin,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.plugins.payment_resolution_strategy import (
    PaymentResolutionStrategyPlugin,
)


def register_scenario_reconciliation_plugins(
    registry: EnterpriseReliabilityPluginRegistry,
    lookup: ExternalRealityLookupPort,
    *,
    payment_evidence_lookup: PaymentReconciliationEvidenceLookupPort | None = None,
    payment_governance_lookup: PaymentGovernanceBusinessContextLookupPort | None = None,
    payment_governance_policy: PaymentEnterpriseGovernancePolicy | None = None,
) -> ScenarioExternalRealityReconciliationPlugin:
    """Install reconciliation probe + payment resolution and optional governance plugins."""
    reconcile = ScenarioExternalRealityReconciliationPlugin(_lookup=lookup)
    registry.register(reconcile)
    registry.register(
        PaymentResolutionStrategyPlugin(
            _lookup=payment_evidence_lookup or InMemoryPaymentReconciliationEvidenceLookup(),
        ),
    )
    if payment_governance_lookup is not None:
        policy = payment_governance_policy or PaymentEnterpriseGovernancePolicy(
            human_approval_threshold_amount=Decimal("10000.00"),
            currency="EUR",
        )
        registry.register(
            PaymentGovernancePolicyPlugin(
                _lookup=payment_governance_lookup,
                _policy=policy,
            ),
        )
    return reconcile
