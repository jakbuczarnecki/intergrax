# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Qdrant vector store."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.qdrant.bundle import (
    create_qdrant_vector_store_integration,
)
from intergrax.integrations.providers.vector_store.qdrant.integration import (
    QDRANT_VECTOR_STORE_PROVIDER_ID,
    QdrantVectorStoreIntegration,
    QdrantVectorStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="vector_store",
    provider_id=QDRANT_VECTOR_STORE_PROVIDER_ID,
    integration_class=QdrantVectorStoreIntegration,
    contract_class=VectorStoreIntegrationContract,
    contract_factory=create_qdrant_vector_store_integration,
    display_name="Qdrant",
    config_class=QdrantVectorStoreIntegrationConfig,
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
