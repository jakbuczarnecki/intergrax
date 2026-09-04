# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Milvus vector store."""

from __future__ import annotations

from intergrax.integrations.providers.vector_store.milvus.bundle import (
    create_milvus_vector_store_integration,
)
from intergrax.integrations.providers.vector_store.milvus.integration import (
    MILVUS_VECTOR_STORE_PROVIDER_ID,
    MilvusVectorStoreIntegration,
    MilvusVectorStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import VectorStoreIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="vector_store",
    provider_id=MILVUS_VECTOR_STORE_PROVIDER_ID,
    integration_class=MilvusVectorStoreIntegration,
    contract_class=VectorStoreIntegrationContract,
    contract_factory=create_milvus_vector_store_integration,
    display_name="Milvus",
    config_class=MilvusVectorStoreIntegrationConfig,
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
