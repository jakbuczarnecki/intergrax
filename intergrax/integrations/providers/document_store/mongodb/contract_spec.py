# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for MongoDB document store."""

from __future__ import annotations

from intergrax.integrations.providers.document_store.mongodb.bundle import (
    create_mongodb_document_store_integration,
)
from intergrax.integrations.providers.document_store.mongodb.integration import (
    MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    MongoDBDocumentStoreIntegration,
    MongoDBDocumentStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.document_store import DocumentStoreVendorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="document_store",
    provider_id=MONGODB_DOCUMENT_STORE_PROVIDER_ID,
    integration_class=MongoDBDocumentStoreIntegration,
    contract_class=DocumentStoreVendorIntegrationContract,
    contract_factory=create_mongodb_document_store_integration,
    display_name="MongoDB",
    config_class=MongoDBDocumentStoreIntegrationConfig,
    capabilities=(
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
