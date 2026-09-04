# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Duckdb relational store."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.duckdb.bundle import (
    create_duckdb_relational_store_integration,
)
from intergrax.integrations.providers.relational_store.duckdb.integration import (
    DUCKDB_RELATIONAL_STORE_PROVIDER_ID,
    DuckdbRelationalStoreIntegration,
    DuckdbRelationalStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="relational_store",
    provider_id=DUCKDB_RELATIONAL_STORE_PROVIDER_ID,
    integration_class=DuckdbRelationalStoreIntegration,
    contract_class=RelationalStoreIntegrationContract,
    contract_factory=create_duckdb_relational_store_integration,
    display_name="Duckdb",
    config_class=DuckdbRelationalStoreIntegrationConfig,
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
