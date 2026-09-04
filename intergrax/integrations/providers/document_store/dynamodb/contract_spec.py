# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for DynamoDB document store."""

from __future__ import annotations

from intergrax.integrations.providers.document_store.dynamodb.bundle import (
    create_dynamodb_document_store_integration,
)
from intergrax.integrations.providers.document_store.dynamodb.integration import (
    DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
    DynamoDBDocumentStoreIntegration,
    DynamoDBDocumentStoreIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.document_store import DocumentStoreVendorIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="document_store",
    provider_id=DYNAMODB_DOCUMENT_STORE_PROVIDER_ID,
    integration_class=DynamoDBDocumentStoreIntegration,
    contract_class=DocumentStoreVendorIntegrationContract,
    contract_factory=create_dynamodb_document_store_integration,
    display_name="DynamoDB",
    config_class=DynamoDBDocumentStoreIntegrationConfig,
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
