# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Backblaze B2 object storage."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.backblaze_b2.bundle import (
    create_backblaze_b2_object_storage_integration,
)
from intergrax.integrations.providers.object_storage.backblaze_b2.integration import (
    BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
    BackblazeB2ObjectStorageIntegration,
    BackblazeB2ObjectStorageIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="object_storage",
    provider_id=BACKBLAZE_B2_OBJECT_STORAGE_PROVIDER_ID,
    integration_class=BackblazeB2ObjectStorageIntegration,
    contract_class=ObjectStorageIntegrationContract,
    contract_factory=create_backblaze_b2_object_storage_integration,
    display_name="Backblaze B2",
    config_class=BackblazeB2ObjectStorageIntegrationConfig,
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
