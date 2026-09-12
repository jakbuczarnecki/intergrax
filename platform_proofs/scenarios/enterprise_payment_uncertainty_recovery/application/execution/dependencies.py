"""Explicit execution dependencies — replaceable ports, no global registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from intergrax.contracts.enterprise_reliability.plugin_spi import EnterpriseReliabilityPluginGateway

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.application.composition.dependencies import (
    ApplicationDependencies,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.contracts.provisioning.ports import (
    ScenarioProvisioningPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.external_reality_lookup import (
    ExternalRealityLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_governance_context import (
    PaymentGovernanceBusinessContextLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_reconciliation_evidence import (
    PaymentReconciliationEvidenceLookupPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.erl_integration.contracts.payment_recovery_action import (
    PaymentRecoveryActionPort,
)
from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.external_payment.adapters.in_memory_persistence import (
    InMemoryExternalRealityStore,
)


@dataclass(frozen=True, slots=True)
class EnterprisePaymentExecutionDependencies:
    dataset_package_root: Path
    provisioning_port: ScenarioProvisioningPort
    application: ApplicationDependencies
    external_reality_store: InMemoryExternalRealityStore
    reality_lookup: ExternalRealityLookupPort
    payment_evidence_lookup: PaymentReconciliationEvidenceLookupPort
    payment_governance_lookup: PaymentGovernanceBusinessContextLookupPort | None
    payment_recovery_action_port: PaymentRecoveryActionPort
    erl_gateway: EnterpriseReliabilityPluginGateway
