# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Gcp Secret Manager secrets store."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.gcp_secret_manager.bundle import (
    create_gcp_secret_manager_secrets_store_integration,
)
from intergrax.integrations.providers.secrets_store.gcp_secret_manager.integration import (
    GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
    GcpSecretManagerSecretsStoreIntegration,
    GcpSecretManagerSecretsStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="secrets_store",
    provider_id=GCP_SECRET_MANAGER_SECRETS_STORE_PROVIDER_ID,
    integration_class=GcpSecretManagerSecretsStoreIntegration,
    contract_class=SecretsStoreIntegrationContract,
    contract_factory=create_gcp_secret_manager_secrets_store_integration,
    display_name="Gcp Secret Manager",
    config_class=GcpSecretManagerSecretsStoreIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
