# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit contract declaration for Cloudflare R2 object storage."""

from __future__ import annotations

from intergrax.integrations.providers.object_storage.cloudflare_r2.bundle import (
    create_cloudflare_r2_object_storage_integration,
)
from intergrax.integrations.providers.object_storage.cloudflare_r2.integration import (
    CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID,
    CloudflareR2ObjectStorageIntegration,
    CloudflareR2ObjectStorageIntegrationConfig,
)
from intergrax.integrations.registry.contract_spec import declare_integration_contract
from intergrax.runtime.integrations.categories.storage import ObjectStorageIntegrationContract
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationSecurityPosture,
)

CONTRACT_SPEC = declare_integration_contract(
    category="object_storage",
    provider_id=CLOUDFLARE_R2_OBJECT_STORAGE_PROVIDER_ID,
    integration_class=CloudflareR2ObjectStorageIntegration,
    contract_class=ObjectStorageIntegrationContract,
    contract_factory=create_cloudflare_r2_object_storage_integration,
    display_name="Cloudflare R2",
    config_class=CloudflareR2ObjectStorageIntegrationConfig,
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
