# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Azure Blob object storage."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.azure_blob.bundle import (
    create_azure_blob_object_storage_integration,
)
from intergrax.integrations.providers.object_storage.azure_blob.integration import (
    AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID,
    AzureBlobObjectStorageIntegration,
    AzureBlobObjectStorageIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="object_storage",
    provider_id=AZURE_BLOB_OBJECT_STORAGE_PROVIDER_ID,
    integration_class=AzureBlobObjectStorageIntegration,
    contract_class=ObjectStorageIntegrationContract,
    contract_factory=create_azure_blob_object_storage_integration,
    display_name="Azure Blob",
    config_class=AzureBlobObjectStorageIntegrationConfig,
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
