# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Motherduck relational store."""

from __future__ import annotations

from intergrax.integrations.providers.relational_store.motherduck.bundle import (
    create_motherduck_relational_store_integration,
)
from intergrax.integrations.providers.relational_store.motherduck.integration import (
    MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID,
    MotherduckRelationalStoreIntegration,
    MotherduckRelationalStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.data import RelationalStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="relational_store",
    provider_id=MOTHERDUCK_RELATIONAL_STORE_PROVIDER_ID,
    integration_class=MotherduckRelationalStoreIntegration,
    contract_class=RelationalStoreIntegrationContract,
    contract_factory=create_motherduck_relational_store_integration,
    display_name="Motherduck",
    config_class=MotherduckRelationalStoreIntegrationConfig,
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
