# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Inmemory vector store."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.inmemory.bundle import (
    create_inmemory_vector_store_integration,
)
from intergrax.integrations.providers.vector_store.inmemory.integration import (
    INMEMORY_VECTOR_STORE_PROVIDER_ID,
    InmemoryVectorStoreIntegration,
    InmemoryVectorStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="vector_store",
    provider_id=INMEMORY_VECTOR_STORE_PROVIDER_ID,
    integration_class=InmemoryVectorStoreIntegration,
    contract_class=VectorStoreIntegrationContract,
    contract_factory=create_inmemory_vector_store_integration,
    display_name="Inmemory",
    config_class=InmemoryVectorStoreIntegrationConfig,
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
