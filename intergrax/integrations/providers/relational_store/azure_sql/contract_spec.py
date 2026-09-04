# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Azure Sql relational store."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.azure_sql.bundle import (
    create_azure_sql_relational_store_integration,
)
from intergrax.integrations.providers.relational_store.azure_sql.integration import (
    AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
    AzureSqlRelationalStoreIntegration,
    AzureSqlRelationalStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="relational_store",
    provider_id=AZURE_SQL_RELATIONAL_STORE_PROVIDER_ID,
    integration_class=AzureSqlRelationalStoreIntegration,
    contract_class=RelationalStoreIntegrationContract,
    contract_factory=create_azure_sql_relational_store_integration,
    display_name="Azure Sql",
    config_class=AzureSqlRelationalStoreIntegrationConfig,
    capabilities=(
        PlatformIntegrationCapability.CONNECT,
        PlatformIntegrationCapability.READ,
        PlatformIntegrationCapability.WRITE,
        PlatformIntegrationCapability.HEALTH_CHECK,
    ),
    security_posture=PlatformIntegrationSecurityPosture(),
    supports_runtime_binding=True,
    supports_health_check=True,
    metadata={"source": "explicit_provider_declaration"},
)

CONTRACT_SPECS = (CONTRACT_SPEC,)

__all__ = ["CONTRACT_SPEC", "CONTRACT_SPECS"]
