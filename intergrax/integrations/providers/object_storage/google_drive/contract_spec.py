# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Google Drive object storage."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.google_drive.bundle import (
    create_google_drive_object_storage_integration,
)
from intergrax.integrations.providers.object_storage.google_drive.integration import (
    GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID,
    GoogleDriveObjectStorageIntegration,
    GoogleDriveObjectStorageIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="object_storage",
    provider_id=GOOGLE_DRIVE_OBJECT_STORAGE_PROVIDER_ID,
    integration_class=GoogleDriveObjectStorageIntegration,
    contract_class=ObjectStorageIntegrationContract,
    contract_factory=create_google_drive_object_storage_integration,
    display_name="Google Drive",
    config_class=GoogleDriveObjectStorageIntegrationConfig,
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
