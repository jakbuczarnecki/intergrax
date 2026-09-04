# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Vault secrets store."""

from __future__ import annotations

from intergrax.integrations.providers.secrets_store.vault.bundle import (
    create_vault_secrets_store_integration,
)
from intergrax.integrations.providers.secrets_store.vault.integration import (
    VAULT_SECRETS_STORE_PROVIDER_ID,
    VaultSecretsStoreIntegration,
    VaultSecretsStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.security import SecretsStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="secrets_store",
    provider_id=VAULT_SECRETS_STORE_PROVIDER_ID,
    integration_class=VaultSecretsStoreIntegration,
    contract_class=SecretsStoreIntegrationContract,
    contract_factory=create_vault_secrets_store_integration,
    display_name="Vault",
    config_class=VaultSecretsStoreIntegrationConfig,
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
