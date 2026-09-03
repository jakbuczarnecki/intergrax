# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Postgresql relational store."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.postgresql.bundle import (
    create_postgresql_relational_store_integration,
)
from intergrax.integrations.providers.relational_store.postgresql.integration import (
    POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID,
    PostgresqlRelationalStoreIntegration,
    PostgresqlRelationalStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="relational_store",
    provider_id=POSTGRESQL_RELATIONAL_STORE_PROVIDER_ID,
    integration_class=PostgresqlRelationalStoreIntegration,
    contract_class=RelationalStoreIntegrationContract,
    contract_factory=create_postgresql_relational_store_integration,
    display_name="Postgresql",
    config_class=PostgresqlRelationalStoreIntegrationConfig,
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
