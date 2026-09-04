# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Cloud Sql relational store."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.cloud_sql.bundle import (
    create_cloud_sql_relational_store_integration,
)
from intergrax.integrations.providers.relational_store.cloud_sql.integration import (
    CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
    CloudSqlRelationalStoreIntegration,
    CloudSqlRelationalStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="relational_store",
    provider_id=CLOUD_SQL_RELATIONAL_STORE_PROVIDER_ID,
    integration_class=CloudSqlRelationalStoreIntegration,
    contract_class=RelationalStoreIntegrationContract,
    contract_factory=create_cloud_sql_relational_store_integration,
    display_name="Cloud Sql",
    config_class=CloudSqlRelationalStoreIntegrationConfig,
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
